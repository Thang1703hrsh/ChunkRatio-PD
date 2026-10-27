"""Teacher-displacement diagnostic before any student training (method §§7,10.2; NARVAL pilot).

A DIAGNOSTIC, not a result: it estimates d_s(z) = log T+(z|s) - log T0(z|s) under rho rollouts,
the empirical Jensen gap log mean exp(d) - mean d, and the sibling spread omega_s (eq. 15a).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .artifacts import ShardStore, atomic_bytes, digest, write_json
from .collection import collect_prefixes
from .config import collection_seed
from .sampling import generate_batch, stream, terminal
from .scoring import continuation_increments


def gap_statistics(values: list[float], seed: int, name: str, bootstrap: int = 200) -> dict[str,Any]:
    """Moments of d, empirical Jensen gap with bootstrap interval and rare-tail flag."""
    d = torch.tensor(values,dtype=torch.float64)
    weights = (d-d.max()).exp()
    gap = float(d.max()+weights.mean().log()-d.mean())
    indices = torch.randint(len(d),(bootstrap,len(d)),generator=stream(seed,'displacement-bootstrap/'+name))
    samples = d[indices]
    replicas = torch.logsumexp(samples,dim=1)-math.log(len(d))-samples.mean(dim=1)
    low,high = torch.quantile(replicas,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
    share = float(weights.max()/weights.sum())
    return {'n':len(d),'mean':float(d.mean()),'variance':float(d.var(unbiased=True)) if len(d) > 1 else 0.,
            'min':float(d.min()),'max':float(d.max()),'range':float(d.max()-d.min()),
            'jensen_gap':gap,'jensen_gap_bootstrap_95':[low,high],'max_exponential_weight_share':share,
            'rare_tail_warning':share > .5}


def rollout_returns(models: dict[str,nn.Module], prefix: list[int], remaining: int, eos: int,
                    seed: int, name: str, replicates: int) -> tuple[list[float],list[int]]:
    """R complete rho continuations (collection sampling) scored by teacher forcing under T+ and T0."""
    rngs = [stream(seed,f'displacement/{name}/{r}') for r in range(replicates)]
    suffixes = generate_batch(models['student_ref'],[prefix]*replicates,[remaining]*replicates,eos,rngs)
    values = []
    for suffix in suffixes:
        plus = continuation_increments(models['teacher_plus'],prefix,suffix,eos).cpu().double()
        base = continuation_increments(models['teacher_base'],prefix,suffix,eos).cpu().double()
        values.append(float((plus-base).sum()))
    return values,[len(s) for s in suffixes]


def state_report(models: dict[str,nn.Module], state: dict[str,Any], cfg: dict[str,Any], eos: int,
                 replicates: int) -> dict[str,Any]:
    """Statistics at one state and the omega_s spread of gaps over its K=4 sibling endpoints."""
    seed, horizon = collection_seed(cfg), cfg['horizon']
    values,lengths = rollout_returns(models,state['prefix_ids'],horizon-state['n'],eos,seed,state['id'],replicates)
    report = {'state_id':state['id'],'prompt_id':state['prompt_id'],'n':state['n'],
              'mean_suffix_length':sum(lengths)/len(lengths),**gap_statistics(values,seed,state['id'])}
    siblings: dict[tuple[int,...],dict[str,Any]] = {}
    for source in ('student_ref','teacher_plus'):
        rngs = [stream(seed,f"displacement-candidate/{state['id']}/{source}/{i}") for i in range(2)]
        chunks = generate_batch(models[source],[state['prefix_ids']]*2,[horizon-state['n']]*2,eos,rngs,cfg['chunk_length'])
        for chunk in chunks:
            siblings.setdefault(tuple(chunk),{'tokens':chunk,'sources':[]})['sources'].append(source)
    endpoints = []
    for chunk,row in siblings.items():
        if terminal(state['response_ids']+list(chunk),horizon,eos):
            endpoints.append({**row,'terminal':True,'jensen_gap':0.})
            continue
        name = digest([state['id'],list(chunk)])
        d,_ = rollout_returns(models,state['prefix_ids']+list(chunk),horizon-state['n']-len(chunk),eos,seed,name,replicates)
        endpoints.append({**row,'terminal':False,**gap_statistics(d,seed,name)})
    gaps = [e['jensen_gap'] for e in endpoints]
    report['sibling_endpoints'] = endpoints
    report['omega_s'] = max(gaps)-min(gaps) if len(gaps) > 1 else None
    return report


def diagnose(cfg: dict[str,Any], n_prompts: int, replicates: int, output: Path) -> dict[str,Any]:
    """Run on the first N eligible train prompts (sorted ids) and their pre-registered states."""
    from .models import load_models
    from .pipeline import eligible_prompts
    if n_prompts <= 0 or replicates < 2:
        raise ValueError('need at least one prompt and two replicates')
    models,_,tokenizers,info = load_models(cfg)
    prompts,ineligible = eligible_prompts(cfg,models,tokenizers)
    chosen = [p for p in prompts if p['split'] == 'train'][:n_prompts]
    store = ShardStore(output/'prefixes',{'collection_seed':collection_seed(cfg),'horizon':cfg['horizon'],
                                          'chunk_length':cfg['chunk_length'],'states_per_prompt':cfg.get('states_per_prompt')})
    states = collect_prefixes(chosen,models['student_ref'],{**cfg,'variant':'crpd_main'},info['eos'],store)
    states = [s for s in states if s['prompt_id'] in {p['id'] for p in chosen}]
    reports = [state_report(models,state,cfg,info['eos'],replicates) for state in states]
    summary = {'label':'DIAGNOSTIC ONLY: displacement/Jensen-gap estimates, not a result',
               'prompts':len(chosen),'states':len(reports),'replicates':replicates,
               'ineligible_prompts':ineligible,'collection_seed':collection_seed(cfg),
               'mean_abs_d_mean':sum(abs(r['mean']) for r in reports)/len(reports),
               'mean_jensen_gap':sum(r['jensen_gap'] for r in reports)/len(reports),
               'mean_omega_s':_mean([r['omega_s'] for r in reports if r['omega_s'] is not None]),
               'rare_tail_states':sum(r['rare_tail_warning'] for r in reports),'states_detail':reports}
    write_json(output/'displacement.json',summary)
    omega = summary['mean_omega_s']
    lines = ['# Displacement diagnostic (not a result)','',
             f"{summary['prompts']} prompts, {summary['states']} states, R={replicates} rho rollouts per state and endpoint.",'',
             '| Quantity | Mean over states |','|---|---:|',
             f"| abs mean d_s | {summary['mean_abs_d_mean']:.4g} |",
             f"| Jensen gap log mean exp d - mean d | {summary['mean_jensen_gap']:.4g} |",
             f"| omega_s (sibling gap spread, eq. 15a) | {'n/a' if omega is None else format(omega,'.4g')} |",
             f"| states with rare-tail warning (one sample > 50% of exp weight) | {summary['rare_tail_states']} |",'',
             'Finite-sample estimates; a small R can miss rare high-weight suffixes.']
    atomic_bytes(output/'displacement.md',('\n'.join(lines)+'\n').encode())
    return {k:v for k,v in summary.items() if k != 'states_detail'}


def _mean(values: list[float]) -> float | None:
    """Mean or None for an empty list."""
    return sum(values)/len(values) if values else None
