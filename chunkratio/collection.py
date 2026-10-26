"""Canonical split/prefix/candidate/correction records with resume (§§5–7,10)."""
from __future__ import annotations

import contextlib
import copy
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .artifacts import ShardStore, digest, read_jsonl
from .math import suffix_returns
from .sampling import generate, stream, terminal
from .scoring import continuation_increments, score_batch, trajectory_features


@contextlib.contextmanager
def record_errors(kind: str, record_id: str) -> Iterator[None]:
    """Non-finite or invalid scores fail with the record that produced them (§6.1)."""
    try:
        yield
    except AssertionError as exc:
        raise ValueError(f'nonfinite or invalid score in {kind} {record_id}') from exc


def prepare_prompts(records: list[dict[str,Any]], seed: int) -> list[dict[str,Any]]:
    """Group duplicate canonical token prompts before generation; reject split leakage (§7.1)."""
    groups: dict[str,dict[str,Any]] = {}
    for original in records:
        row = copy.deepcopy(original)
        assert row['prompt_ids'] and all(isinstance(i,int) and i >= 0 for i in row['prompt_ids'])
        key = digest(row['prompt_ids'])
        # Explicit duplicate-group IDs can link differently serialized equivalent prompts.
        group = row.get('group_id',key)
        if 'split' not in row:
            bucket = int(digest([seed,group])[:8],16)/2**32
            row['split'] = 'train' if bucket < .8 else ('head_val' if bucket < .9 else 'select')
        if row['split'] not in ('train','head_val','select','test'):
            raise ValueError('invalid split')
        for existing in groups.values():
            if (existing['group_id'] == group or existing['prompt_ids'] == row['prompt_ids']) and existing['split'] != row['split']:
                raise ValueError('duplicate prompt group crosses splits')
        row['group_id'] = group
        row['id'] = key
        if key in groups:
            if row.get('chosen_ids') != groups[key].get('chosen_ids') or row.get('rejected_ids') != groups[key].get('rejected_ids'):
                raise ValueError('duplicate prompt with conflicting anchor pair')
        else:
            groups[key] = row
    return sorted(groups.values(),key=lambda r:r['id'])


def collect_prefixes(prompts: list[dict[str,Any]], rho: nn.Module, cfg: dict[str,Any],
                     eos: int, store: ShardStore) -> list[dict[str,Any]]:
    """Frozen-rho pool at every causal L boundary, including initial state (§7.2)."""
    length = 1 if cfg['variant'] == 'crpd_token' else cfg['chunk_length']
    for prompt in prompts:
        if prompt['split'] not in ('train','head_val'):
            continue
        if store.get(prompt['id']) is not None:
            continue
        rng = stream(cfg['seed'],'prefix/'+prompt['id'])
        before = rng.get_state().tolist()
        response = generate(rho,prompt['prompt_ids'],cfg['horizon'],eos,rng)
        states = []
        for n in range(0,len(response)+1,length):
            if terminal(response[:n],cfg['horizon'],eos):
                continue
            states.append({'id':digest([prompt['id'],response[:n]]),'prompt_id':prompt['id'],
                           'split':prompt['split'],'prefix_ids':prompt['prompt_ids']+response[:n],
                           'response_ids':response[:n],'n':n})
        store.put(prompt['id'],{'prompt':prompt,'response':response,'states':states,
                               'rng_before':before,'rng_after':rng.get_state().tolist(),
                               'generated_tokens':{'student_ref':len(response)}})
    return [s for record in store.records() for s in record['states']]


def collect_candidates(states: list[dict[str,Any]], models: dict[str,nn.Module], cfg: dict[str,Any],
                       eos: int, store: ShardStore) -> list[dict[str,Any]]:
    """Two rho + two teacher siblings, exact dedup/provenance, skip <2 (§7.2)."""
    length = 1 if cfg['variant'] == 'crpd_token' else cfg['chunk_length']
    for state in states:
        if store.get(state['id']) is not None:
            continue
        with record_errors('candidate state',state['id']):
            candidates: dict[tuple[int,...],dict[str,Any]] = {}
            counts = {'student_ref':0,'teacher_plus':0}
            rng_states = {}
            for source in ('student_ref','teacher_plus'):
                rng = stream(cfg['seed'],'candidate/'+state['id']+'/'+source)
                before = rng.get_state().tolist()
                for _ in range(2):
                    tokens = generate(models[source],state['prefix_ids'],cfg['horizon']-state['n'],eos,rng,
                                      None if cfg['variant'] == 'crpd_suffix' else length)
                    counts[source] += len(tokens)
                    key = tuple(tokens)
                    if key in candidates:
                        candidates[key]['sources'].append(source)
                    else:
                        candidates[key] = {'id':digest([state['id'],tokens]),'tokens':tokens,'sources':[source],
                                           'terminal':terminal(state['response_ids']+tokens,cfg['horizon'],eos)}
                rng_states[source] = {'before':before,'after':rng.get_state().tolist()}
            rows = list(candidates.values())
            for row in rows:
                for source,model in models.items():
                    row['logp_'+source] = score_batch(model,[state['prefix_ids']],[row['tokens']],eos).item()
            scored = {source:sum(len(row['tokens']) for row in rows) for source in models}
            store.put(state['id'],{'state':state,'candidates':rows,'skipped':len(rows)<2,
                                   'rng':rng_states,'generated_tokens':counts,'scored_tokens':scored})
    return store.records()


def correction_seeds(candidates: list[dict[str,Any]], cfg: dict[str,Any]) -> list[dict[str,Any]]:
    """Preselect prefixes and endpoint Bernoulli sample before observing returns (§7.3)."""
    seeds = {}
    for record in candidates:
        state = record['state']
        seeds[state['id']] = {**state,'seed_source':'prefix'}
        for candidate in record['candidates']:
            if candidate['terminal']:
                continue
            fraction = int(digest([cfg['seed'],'endpoint-selection',candidate['id']])[:12],16)/16**12
            if fraction < cfg['endpoint_fraction']:
                endpoint = {**state,'id':candidate['id'],'prefix_ids':state['prefix_ids']+candidate['tokens'],
                            'response_ids':state['response_ids']+candidate['tokens'],
                            'n':state['n']+len(candidate['tokens']),'seed_source':candidate['sources']}
                seeds[endpoint['id']] = endpoint
    return sorted(seeds.values(),key=lambda x:x['id'])


def collect_correction(candidates: list[dict[str,Any]], models: dict[str,nn.Module],
                       cfg: dict[str,Any], eos: int, store: ShardStore) -> list[dict[str,Any]]:
    """Complete suffixes, teacher-forced returns, detached causal rho features (§7.3)."""
    teacher_direction = cfg['variant'] == 'crpd_mu_teacher'
    source = 'teacher_plus' if teacher_direction else 'student_ref'
    for seed in correction_seeds(candidates,cfg):
        for replica in range(cfg['correction_replicates']):
            job_id = digest([seed['id'],replica,source])
            if store.get(job_id) is not None:
                continue
            with record_errors('correction job',job_id):
                rng = stream(cfg['seed'],'correction/'+job_id)
                before = rng.get_state().tolist()
                tokens = generate(models[source],seed['prefix_ids'],cfg['horizon']-seed['n'],eos,rng)
                numerator = 'student_ref' if teacher_direction else 'teacher_plus'
                a = continuation_increments(models[numerator],seed['prefix_ids'],tokens,eos)
                b = continuation_increments(models['teacher_base'],seed['prefix_ids'],tokens,eos)
                increments = a-b
                labels = suffix_returns(increments)
                # Binary attachment written first; the JSON commit records its content hash.
                features = trajectory_features(models['student_ref'],seed['prefix_ids'],tokens)
                features_sha256 = store.put_tensor(job_id,'features',features)
                budgets = [(cfg['horizon']-seed['n']-t)/cfg['horizon'] for t in range(len(tokens))]
                store.put(job_id,{'id':job_id,'seed':seed,'tokens':tokens,'features_sha256':features_sha256,
                                  'features_shape':list(features.shape),'budgets':budgets,
                                  'increments':increments.cpu().tolist(),'labels':labels.cpu().tolist(),
                                  'terminal_target':0.,'rng_before':before,'rng_after':rng.get_state().tolist(),
                                  'generated_tokens':{source:len(tokens)},
                                  'scored_tokens':{numerator:len(tokens),'teacher_base':len(tokens)}})
    return store.records()


def correction_features(store: ShardStore) -> Callable[[dict[str,Any]],torch.Tensor]:
    """Lazy hash-verified loader for a correction row's detached rho features (§§6.2,10)."""
    return lambda row: store.get_tensor(row['id'],'features',row['features_sha256'])
