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


def generate_rows(model: nn.Module, rows: list[tuple[list[int],int,str]], eos: int, seed: int,
                  batch_rows: int, chunk_length: int | None = None) -> list[list[int]]:
    """(prefix, remaining, stream name) rows sampled in batches; each row draws only from its own
    named stream, so the tokens do not depend on how rows are grouped (§5.3)."""
    out: list[list[int]] = []
    for start in range(0,len(rows),batch_rows):
        group = rows[start:start+batch_rows]
        out += generate_batch(model,[p for p,_,_ in group],[r for _,r,_ in group],eos,
                              [stream(seed,name) for _,_,name in group],chunk_length)
    return out


def rollout_returns_many(models: dict[str,nn.Module], jobs: list[tuple[list[int],int,str]], eos: int,
                         seed: int, replicates: int, batch_rows: int) -> list[tuple[list[float],list[int]]]:
    """For each (prefix, remaining, name) job: R complete rho continuations scored under T+ and T0."""
    rows = [(prefix,remaining,f'displacement/{name}/{r}') for prefix,remaining,name in jobs for r in range(replicates)]
    suffixes = generate_rows(models['student_ref'],rows,eos,seed,batch_rows)
    values = []
    for (prefix,_,_),suffix in zip(rows,suffixes):
        plus = continuation_increments(models['teacher_plus'],prefix,suffix,eos).cpu().double()
        base = continuation_increments(models['teacher_base'],prefix,suffix,eos).cpu().double()
        values.append(float((plus-base).sum()))
    return [(values[j*replicates:(j+1)*replicates],[len(s) for s in suffixes[j*replicates:(j+1)*replicates]])
            for j in range(len(jobs))]


def state_reports(models: dict[str,nn.Module], states: list[dict[str,Any]], cfg: dict[str,Any], eos: int,
                  replicates: int) -> list[dict[str,Any]]:
    """Per state: statistics and the omega_s spread of gaps over its K=4 sibling endpoints.

    Sampling is batched across states (one row per stream), which keeps the GPU busy without
    changing any token: a per-state loop of R-row batches left it about 60% idle.
    """
    seed, horizon = collection_seed(cfg), cfg['horizon']
    batch_rows = cfg.get('generation_batch_size') or 16
    chunks = {source:generate_rows(models[source],
                                   [(s['prefix_ids'],horizon-s['n'],f"displacement-candidate/{s['id']}/{source}/{i}")
                                    for s in states for i in range(2)],eos,seed,batch_rows,cfg['chunk_length'])
              for source in ('student_ref','teacher_plus')}
    siblings: list[dict[tuple[int,...],dict[str,Any]]] = []
    jobs = []
    for index,state in enumerate(states):
        found: dict[tuple[int,...],dict[str,Any]] = {}
        for source in ('student_ref','teacher_plus'):
            for chunk in chunks[source][2*index:2*index+2]:
                found.setdefault(tuple(chunk),{'tokens':chunk,'sources':[]})['sources'].append(source)
        siblings.append(found)
        jobs.append((state['prefix_ids'],horizon-state['n'],state['id']))
        for chunk in found:
            if not terminal(state['response_ids']+list(chunk),horizon,eos):
                jobs.append((state['prefix_ids']+list(chunk),horizon-state['n']-len(chunk),digest([state['id'],list(chunk)])))
    results = iter(rollout_returns_many(models,jobs,eos,seed,replicates,batch_rows))
    reports = []
    for state,found in zip(states,siblings):
        values,lengths = next(results)
        report = {'state_id':state['id'],'prompt_id':state['prompt_id'],'n':state['n'],
                  'mean_suffix_length':sum(lengths)/len(lengths),**gap_statistics(values,seed,state['id'])}
        endpoints = []
        for chunk,row in found.items():
            if terminal(state['response_ids']+list(chunk),horizon,eos):
                endpoints.append({**row,'terminal':True,'jensen_gap':0.})
                continue
            d,_ = next(results)
            endpoints.append({**row,'terminal':False,**gap_statistics(d,seed,digest([state['id'],list(chunk)]))})
        gaps = [e['jensen_gap'] for e in endpoints]
        report['sibling_endpoints'] = endpoints
        report['omega_s'] = max(gaps)-min(gaps) if len(gaps) > 1 else None
        reports.append(report)
    return reports


def state_report(models: dict[str,nn.Module], state: dict[str,Any], cfg: dict[str,Any], eos: int,
                 replicates: int) -> dict[str,Any]:
    """Statistics at one state and the omega_s spread of gaps over its K=4 sibling endpoints."""
    return state_reports(models,[state],cfg,eos,replicates)[0]


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
    reports = state_reports(models,states,cfg,info['eos'],replicates)
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
