"""Frozen endpoint corrections and complete unordered sibling pairs (§§8–9)."""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

import torch
from torch import nn

from .artifacts import digest
from .math import target_score
from .sampling import generate, stream
from .scoring import continuation_increments, state_features
from .value_head import MeanHead


@torch.no_grad()
def exponential_audit(prefix: list[int], remaining: int, models: dict[str,nn.Module],
                      cfg: dict[str,Any], eos: int, job: str) -> dict[str,Any]:
    """Replicated log-mean-exp estimate, delta-method SE and bootstrap uncertainty (§9)."""
    rng = stream(cfg['seed'],'exponential-audit/'+job)
    values, tokens = [],0
    for _ in range(cfg['audit_replicates']):
        suffix = generate(models['student_ref'],prefix,remaining,eos,rng)
        a = continuation_increments(models['teacher_plus'],prefix,suffix,eos)
        b = continuation_increments(models['teacher_base'],prefix,suffix,eos)
        values.append(float((a-b).double().sum()))
        tokens += len(suffix)
    d = torch.tensor(values,dtype=torch.float64)
    shifted = (d-d.max()).exp()
    phi = float(d.max()+shifted.mean().log())
    se = float(shifted.std(unbiased=True)/len(d)**.5/shifted.mean())
    bootstrap = stream(cfg['seed'],'audit-bootstrap/'+job)
    indices = torch.randint(len(d),(256,len(d)),generator=bootstrap)
    replicas = torch.logsumexp(d[indices],dim=1)-math.log(len(d))
    interval = torch.quantile(replicas,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
    return {'phi_logmeanexp':phi,'mu_mc':float(d.mean()),'delta_method_se':se,
            'bootstrap_95':interval,'replicates':len(d),'sample_returns':values,
            'warning':'finite-sample diagnostic; rare exponential tails may be missed',
            'generated_tokens':{'student_ref':tokens},'scored_tokens':{'teacher_plus':tokens,'teacher_base':tokens}}


def build_targets(records: list[dict[str,Any]], head: MeanHead | None, models: dict[str,nn.Module],
                  cfg: dict[str,Any], eos: int, route: str) -> dict[str,Any]:
    """Cache scores independent of alpha; weights cover eligible train prompts (§8)."""
    output, audits = [],[]
    grouped: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for record in records:
        if record['skipped']:
            continue
        state = record['state']
        candidates = []
        for original in record['candidates']:
            row = dict(original)
            bypass = route != 'reference_mean' or cfg['variant'] in ('crpd_raw','crpd_suffix') or row['terminal']
            mu = 0.
            if not bypass:
                assert head is not None and not any(p.requires_grad for p in head.parameters())
                f = state_features(models['student_ref'],[state['prefix_ids']+row['tokens']])
                budget = torch.tensor([(cfg['horizon']-state['n']-len(row['tokens']))/cfg['horizon']],device=f.device)
                with torch.no_grad():
                    mu = head(f,budget).item()
            row['endpoint_mu'] = mu
            row['target_score'] = target_score(*(torch.tensor([row[k]],dtype=torch.float64) for k in
                                                 ('logp_teacher_plus','logp_teacher_base','endpoint_mu')),
                                               torch.tensor([row['terminal']])).item()
            candidates.append(row)
            if cfg['variant'] == 'crpd_audit' and not row['terminal'] and len(audits) < cfg.get('audit_endpoints',8):
                audits.append({'candidate_id':row['id'],'mu_hat':mu,**exponential_audit(
                    state['prefix_ids']+row['tokens'],cfg['horizon']-state['n']-len(row['tokens']),
                    models,cfg,eos,row['id'])})
        result = {'state':state,'candidates':candidates}
        output.append(result)
        if state['split'] == 'train':
            grouped[state['prompt_id']].append(result)
    pairs = []
    for prompt_states in grouped.values():
        for record in prompt_states:
            cs = record['candidates']
            for c,d in itertools.combinations(cs,2):
                pairs.append({'id':digest([c['id'],d['id']]),'state':record['state'],'c':c,'d':d,
                              'weight':1/len(grouped)/len(prompt_states)/math.comb(len(cs),2)})
    if pairs:
        assert abs(sum(p['weight'] for p in pairs)-1) < 1e-10
    return {'records':output,'pairs':pairs,'audits':audits,'route':route}
