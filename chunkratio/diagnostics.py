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

from .artifacts import ShardStore, atomic_bytes, digest, stage_key, write_json
from .collection import collect_prefixes
from .config import collection_seed
from .sampling import generate_batch, stream, terminal
from .scoring import continuation_increments, fingerprint


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
    from .pipeline import code_revision, eligible_prompts
    if n_prompts <= 0 or replicates < 2:
        raise ValueError('need at least one prompt and two replicates')
    models,_,tokenizers,info = load_models(cfg)
    prompts,ineligible = eligible_prompts(cfg,models,tokenizers)
    chosen = [p for p in prompts if p['split'] == 'train'][:n_prompts]
    identities = {role:fingerprint(model) for role,model in models.items()}
    revision = code_revision()
    # Same identity as the pipeline's stage keys (§10): a changed rho, tokenizer, prompt set or code
    # collects afresh next to the old shards instead of resuming prefixes it did not generate.
    metadata = {'models':identities,'tokenizer':info,'split_hash':digest(prompts),'code_revision':revision,
                'collection_seed':collection_seed(cfg),'horizon':cfg['horizon'],
                'chunk_length':cfg['chunk_length'],'states_per_prompt':cfg.get('states_per_prompt')}
    store = ShardStore(output/stage_key('prefixes',metadata),metadata)
    states = collect_prefixes(chosen,models['student_ref'],{**cfg,'variant':'crpd_main'},info['eos'],store)
    states = [s for s in states if s['prompt_id'] in {p['id'] for p in chosen}]
    reports = state_reports(models,states,cfg,info['eos'],replicates)
    summary = {'label':'DIAGNOSTIC ONLY: displacement/Jensen-gap estimates, not a result',
               'prompts':len(chosen),'states':len(reports),'replicates':replicates,
               'ineligible_prompts':ineligible,'collection_seed':collection_seed(cfg),
               'models':identities,'tokenizer':info,'code_revision':revision,
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


def _source(candidate: dict[str,Any]) -> str:
    """rho, teacher or both, from the roles that proposed a candidate or endpoint."""
    sources = set(candidate['sources'] if 'sources' in candidate else candidate['seed_source'])
    return 'both' if len(sources) > 1 else 'rho' if sources == {'student_ref'} else 'teacher'


def _corr_r2(predictions: list[float], labels: list[float]) -> dict[str,float | None]:
    """Pearson correlation around this set's own means, and R^2 of the given predictions (no refit) against
    the labels' own mean; None when either side is constant. Per-state centring of the whole sample does not
    centre a subset (one endpoint source, a bootstrap draw), so the means are always recomputed here."""
    if len(predictions) < 2:
        return {'corr':None,'r2':None}
    mp,my = sum(predictions)/len(predictions),sum(labels)/len(labels)
    spp = sum((p-mp)**2 for p in predictions)
    syy = sum((y-my)**2 for y in labels)
    if spp == 0 or syy == 0:
        return {'corr':None,'r2':None}
    spy = sum((p-mp)*(y-my) for p,y in zip(predictions,labels))
    return {'corr':spy/math.sqrt(spp*syy),'r2':1-sum((p-y)**2 for p,y in zip(predictions,labels))/syy}


def correction_audit(pairs: list[dict[str,Any]], corrections: list[dict[str,Any]], head: nn.Module | None,
                     features: Any, seed: int, bootstrap: int = 200,
                     parents: dict[str,str] | None = None) -> dict[str,Any]:
    """Sibling view of the endpoint correction mu_hat; a diagnostic only (head objective and selection unchanged).

    Pair targets use t_c - t_d, so what matters is mu_hat(sc) - mu_hat(sd), not the level of mu_hat. Pairs:
    spread of mu_hat differences against delta_T differences and the measure-weighted share of pairs whose
    preferred side mu_hat flips, by proposing sources. head_val corrections: endpoints extending the same
    state, centred per state, compare mu_hat with their single-rollout returns (correlation, R^2) by endpoint
    source, with prompt-bootstrap 95% intervals. One rollout per endpoint makes these lower bounds.
    """
    def spread(values: list[float]) -> float | None:
        """Population standard deviation."""
        if not values:
            return None
        mean = sum(values)/len(values)
        return math.sqrt(sum((v-mean)**2 for v in values)/len(values))
    groups: dict[str,list[tuple[float,float,float]]] = {}
    for pair in pairs:
        delta = [pair[s]['logp_teacher_plus']-pair[s]['logp_teacher_base'] for s in ('c','d')]
        mu = [pair[s].get('endpoint_mu',0.) for s in ('c','d')]
        kind = '+'.join(sorted(_source(pair[s]) for s in ('c','d')))
        for key in ('all',kind):
            groups.setdefault(key,[]).append((delta[0]-delta[1],mu[0]-mu[1],pair['weight']))
    report: dict[str,Any] = {'pairs':{}}
    for key,rows in sorted(groups.items()):
        weight = sum(w for *_,w in rows)
        flipped = sum(w for d,m,w in rows if d*(d+m) < 0)
        report['pairs'][key] = {'n':len(rows),'sd_delta_difference':spread([d for d,_,_ in rows]),
                                'sd_mu_difference':spread([m for _,m,_ in rows]),
                                'flip_share':flipped/weight if weight else None}
    if head is None:
        report['endpoints'] = None
        return report
    # Endpoint -> parent state: the longest prefix seed of the same prompt whose ids start the endpoint's.
    prefixes: dict[str,list[dict[str,Any]]] = {}
    for row in corrections:
        if row['seed'].get('seed_source') == 'prefix':
            prefixes.setdefault(row['seed']['prompt_id'],[]).append(row['seed'])
    siblings: dict[str,list[tuple[str,str,float,float]]] = {}
    with torch.no_grad():
        for row in corrections:
            seed_row = row['seed']
            if seed_row.get('seed_source') == 'prefix' or not row['labels']:
                continue
            if parents is not None:
                # Candidate records give each endpoint's state directly (an endpoint seed's id is its
                # candidate's id), so endpoint-only correction (no prefix seeds) is audited too.
                parent_id = parents.get(seed_row['id'])
                if parent_id is None:
                    continue
            else:
                ids = seed_row['prefix_ids']
                found = [s for s in prefixes.get(seed_row['prompt_id'],[]) if len(s['prefix_ids']) < len(ids)
                         and ids[:len(s['prefix_ids'])] == s['prefix_ids']]
                if not found:
                    continue
                parent_id = max(found,key=lambda s:len(s['prefix_ids']))['id']
            first = features(row)[:1].float()
            device = next(head.parameters()).device
            mu = float(head(first.to(device),torch.tensor(row['budgets'][:1],device=device)))
            siblings.setdefault(parent_id,[]).append((seed_row['prompt_id'],_source(seed_row),mu,row['labels'][0]))
    centred = []
    for members in siblings.values():
        if len(members) < 2:
            continue
        mean_mu = sum(m for *_,m,_ in members)/len(members)
        mean_y = sum(y for *_,y in members)/len(members)
        centred += [(prompt,source,mu-mean_mu,y-mean_y) for prompt,source,mu,y in members]
    rng = torch.Generator().manual_seed(seed+919)
    prompts = sorted({p for p,*_ in centred})
    report['endpoints'] = {'states':sum(len(m) >= 2 for m in siblings.values()),'endpoints':len(centred),
                           'note':'single rollout per endpoint: correlations understate how well mu_hat tracks the true value'}
    for key in ('all','rho','teacher','both'):
        rows = [r for r in centred if key == 'all' or r[1] == key]
        stats = {'n':len(rows),**_corr_r2([r[2] for r in rows],[r[3] for r in rows])}
        by_prompt: dict[str,list[tuple[str,str,float,float]]] = {}
        for r in rows:
            by_prompt.setdefault(r[0],[]).append(r)
        if stats['corr'] is not None and len(prompts) > 1:
            draws = []
            for _ in range(bootstrap):
                sample = [prompts[i] for i in torch.randint(len(prompts),(len(prompts),),generator=rng).tolist()]
                chosen = [r for p in sample for r in by_prompt.get(p,[])]
                value = _corr_r2([r[2] for r in chosen],[r[3] for r in chosen])['corr']
                if value is not None:
                    draws.append(value)
            draws.sort()
            if draws:
                stats['corr_95'] = [draws[int(.025*(len(draws)-1))],draws[int(.975*(len(draws)-1))]]
        report['endpoints'][key] = stats
    return report
