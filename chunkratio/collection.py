"""Canonical split/prefix/candidate/correction records with resume (§§5–7,10)."""
from __future__ import annotations

import contextlib
import copy
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .artifacts import ShardStore, digest, read_jsonl, write_json
from .config import collection_seed
from .math import suffix_returns
from .sampling import generate_batch, rng_digest, stream, terminal
from .scoring import continuation_increments_batch, length_groups, score_batch, trajectory_features_batch


@contextlib.contextmanager
def record_errors(kind: str, record_id: str) -> Iterator[None]:
    """Non-finite or invalid scores fail with the record that produced them (§6.1)."""
    try:
        yield
    except AssertionError as exc:
        raise ValueError(f'nonfinite or invalid score in {kind} {record_id}') from exc


@contextlib.contextmanager
def timed(totals: dict[str,float], phase: str) -> Iterator[None]:
    """Add the block's CUDA-synchronized wall seconds to totals[phase]; measurement only."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        totals[phase] = totals.get(phase,0.)+time.perf_counter()-start


def report_timing(cfg: dict[str,Any], stage: str, totals: dict[str,float], jobs: int) -> None:
    """Per-phase seconds of this invocation, printed and written beside the run's outputs."""
    if not jobs:
        return
    report = {'stage':stage,'jobs_this_invocation':jobs,'seconds':{k:round(v,2) for k,v in totals.items()}}
    print(json.dumps({'timing':report}),flush=True)
    if cfg.get('output'):
        write_json(Path(cfg['output'])/f'timing_{stage}.json',report)


def prepare_prompts(records: list[dict[str,Any]], seed: int) -> list[dict[str,Any]]:
    """Group duplicate canonical token prompts before generation; reject split leakage (§7.1).

    Rows that share token ids or a group_id belong to one duplicate group, transitively over the whole
    input (union-find): prompt X in groups A and B links A and B, so B's other prompts must share X's
    split. Every alias stays in the group before any row is deduplicated.
    """
    rows = [copy.deepcopy(original) for original in records]
    parent: dict[tuple[str,str],tuple[str,str]] = {}
    def find(node: tuple[str,str]) -> tuple[str,str]:
        """Root of a node, halving the path as it goes."""
        parent.setdefault(node,node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for row in rows:
        assert row['prompt_ids'] and all(isinstance(i,int) and i >= 0 for i in row['prompt_ids'])
        row['id'] = digest(row['prompt_ids'])
        # Explicit duplicate-group IDs can link differently serialized equivalent prompts.
        row['group_id'] = row.get('group_id',row['id'])
        parent[find(('prompt',row['id']))] = find(('group',row['group_id']))
    aliases: dict[tuple[str,str],set[str]] = {}
    for row in rows:
        aliases.setdefault(find(('prompt',row['id'])),set()).add(row['group_id'])
    splits: dict[tuple[str,str],str] = {}
    for row in rows:
        root = find(('prompt',row['id']))
        if 'split' not in row:
            # One hash per duplicate group (its smallest alias; the row's own id when it has no duplicates).
            bucket = int(digest([seed,min(aliases[root])])[:8],16)/2**32
            row['split'] = 'train' if bucket < .8 else ('head_val' if bucket < .9 else 'select')
        if row['split'] not in ('train','head_val','select','test'):
            raise ValueError('invalid split')
        if splits.setdefault(root,row['split']) != row['split']:
            raise ValueError('duplicate prompt group crosses splits')
    groups: dict[str,dict[str,Any]] = {}
    for row in rows:
        key = row['id']
        if key in groups:
            if row.get('chosen_ids') != groups[key].get('chosen_ids') or row.get('rejected_ids') != groups[key].get('rejected_ids'):
                raise ValueError('duplicate prompt with conflicting anchor pair')
        else:
            groups[key] = row
    return sorted(groups.values(),key=lambda r:r['id'])


def batches(items: list[Any], cfg: dict[str,Any], cost: Callable[[Any],int] | None = None) -> list[list[Any]]:
    """Job groups; results never depend on the grouping (§10).

    Fixed-size by default. With `generation_tokens_per_batch` (opt-in) and a per-item token cost (rows it
    adds x the widest sequence it can reach), items are sorted by cost and cut so rows x largest cost stays
    within the budget, at most `generation_max_rows` per group: many short jobs share a batch, long ones
    do not pad them. KV-cache memory follows the budget instead of the longest prompt.
    """
    budget = cfg.get('generation_tokens_per_batch')
    if budget and cost is not None:
        limit = cfg.get('generation_max_rows',512)
        groups = []
        for group in length_groups([cost(item) for item in items],budget):
            groups += [group[i:i+limit] for i in range(0,len(group),limit)]
        return [[items[i] for i in group] for group in groups]
    size = cfg.get('generation_batch_size',8)
    return [items[i:i+size] for i in range(0,len(items),size)]


def select_states(positions: list[int], prompt_id: str, seed: int, k: int | None) -> list[int]:
    """Uniform state subsample without replacement; n=0 always kept (pre-registered, §7.2).

    Depends only on the schedule positions, the prompt id and the seed; never on returns or scores.
    """
    assert positions and positions[0] == 0 and (k is None or k >= 1)
    if k is None or len(positions) <= k:
        return list(positions)
    rest = positions[1:]
    order = torch.randperm(len(rest),generator=stream(seed,'state-select/'+prompt_id)).tolist()
    return [0]+sorted(rest[i] for i in order[:k-1])


def collect_prefixes(prompts: list[dict[str,Any]], rho: nn.Module, cfg: dict[str,Any],
                     eos: int, store: ShardStore) -> list[dict[str,Any]]:
    """Frozen-rho pool at causal L boundaries, including the initial state (§7.2)."""
    length = 1 if cfg['variant'] == 'crpd_token' else cfg['chunk_length']
    seed = collection_seed(cfg)
    # Similar prompt lengths share a batch (less left padding); every prompt has its own stream and the
    # store reads shards by id, so neither the order nor the group size changes any record.
    # crpd_raw and crpd_suffix use no head, so `collection_splits: ['train']` skips the head_val prompts. Each
    # prompt has its own stream and shard, so the train records are the same either way (DECISIONS 87).
    splits = tuple(cfg.get('collection_splits',('train','head_val')))
    pending = sorted((p for p in prompts if p['split'] in splits and store.get(p['id']) is None),
                     key=lambda p:(len(p['prompt_ids']),p['id']))
    size = cfg.get('prefix_generation_batch_size') or cfg.get('generation_batch_size',8)
    for group in batches(pending,{**cfg,'generation_batch_size':size},lambda p:len(p['prompt_ids'])+cfg['horizon']):
        rngs = [stream(seed,'prefix/'+prompt['id']) for prompt in group]
        befores = [rng_digest(rng) for rng in rngs]
        responses = generate_batch(rho,[p['prompt_ids'] for p in group],[cfg['horizon']]*len(group),eos,rngs,
                                   gpu_sampling=cfg.get('gpu_sampling',False))
        for prompt,response,rng,before in zip(group,responses,rngs,befores):
            positions = [n for n in range(0,len(response)+1,length) if not terminal(response[:n],cfg['horizon'],eos)]
            selected = select_states(positions,prompt['id'],seed,cfg.get('states_per_prompt'))
            states = [{'id':digest([prompt['id'],response[:n]]),'prompt_id':prompt['id'],
                       'split':prompt['split'],'prefix_ids':prompt['prompt_ids']+response[:n],
                       'response_ids':response[:n],'n':n} for n in selected]
            store.put(prompt['id'],{'prompt':prompt,'response':response,'states':states,
                                   'schedule_positions':positions,'selected_positions':selected,
                                   'rng_before':before,'rng_after':rng_digest(rng),
                                   'generated_tokens':{'student_ref':len(response)}})
    # Only the configured splits, so a shared store that also holds head_val shards gives the same states.
    return [s for record in store.records() for s in record['states'] if s['split'] in splits]


def collect_candidates(states: list[dict[str,Any]], models: dict[str,nn.Module], cfg: dict[str,Any],
                       eos: int, store: ShardStore) -> list[dict[str,Any]]:
    """K/2 rho + K/2 teacher siblings (K=4, or K=2 compact), exact dedup/provenance, skip <2 (§7.2).

    Draw i of a source uses the same stream for every K, so the K=2 siblings are the first draws of K=4.
    """
    length = None if cfg['variant'] == 'crpd_suffix' else (1 if cfg['variant'] == 'crpd_token' else cfg['chunk_length'])
    seed = collection_seed(cfg)
    per_source = cfg['candidates']//2
    # Length-sorted groups pad less; each draw keeps its own stream, so records do not change.
    pending = sorted((s for s in states if store.get(s['id']) is None),key=lambda s:(len(s['prefix_ids']),s['id']))
    totals: dict[str,float] = {}
    reach = lambda s: per_source*(len(s['prefix_ids'])+(length or cfg['horizon']-s['n']))
    for group in batches(pending,cfg,reach):
        with record_errors('candidate state',','.join(s['id'] for s in group)):
            draws: dict[str,list[tuple[list[int],dict[str,Any]]]] = {}
            for source in ('student_ref','teacher_plus'):
                jobs = [(state,i) for state in group for i in range(per_source)]
                rngs = [stream(seed,f"candidate/{state['id']}/{source}/{i}") for state,i in jobs]
                befores = [rng_digest(rng) for rng in rngs]
                with timed(totals,'generate_'+source):
                    tokens = generate_batch(models[source],[state['prefix_ids'] for state,_ in jobs],
                                            [cfg['horizon']-state['n'] for state,_ in jobs],eos,rngs,length,
                                            gpu_sampling=cfg.get('gpu_sampling',False))
                draws[source] = [(t,{'before':b,'after':rng_digest(r)}) for t,b,r in zip(tokens,befores,rngs)]
        prepared = []
        for index,state in enumerate(group):
            with record_errors('candidate state',state['id']):
                candidates: dict[tuple[int,...],dict[str,Any]] = {}
                counts = {'student_ref':0,'teacher_plus':0}
                rng_states = {}
                for source in ('student_ref','teacher_plus'):
                    rows = draws[source][per_source*index:per_source*(index+1)]
                    rng_states[source] = [rng for _,rng in rows]
                    for tokens,_ in rows:
                        counts[source] += len(tokens)
                        key = tuple(tokens)
                        if key in candidates:
                            candidates[key]['sources'].append(source)
                        else:
                            candidates[key] = {'id':digest([state['id'],tokens]),'tokens':tokens,'sources':[source],
                                               'terminal':terminal(state['response_ids']+tokens,cfg['horizon'],eos)}
                rows = list(candidates.values())
                if cfg.get('batch_candidate_scoring',False) or cfg.get('shared_prefix_scoring',False):
                    prepared.append({'state':state,'candidates':rows,'skipped':len(rows)<2,
                                     'rng':rng_states,'generated_tokens':counts,
                                     'scored_tokens':{source:sum(len(row['tokens']) for row in rows) for source in models}})
                    continue
                # One batched forward per model for the state's siblings (batched = unbatched, §6.1).
                for source,model in models.items():
                    with timed(totals,'score_'+source):
                        scores = score_batch(model,[state['prefix_ids']]*len(rows),[row['tokens'] for row in rows],eos,
                                             score_eos=cfg.get('score_end_token',True))
                    for row,score in zip(rows,scores.tolist()):
                        row['logp_'+source] = score
                scored = {source:sum(len(row['tokens']) for row in rows) for source in models}
                with timed(totals,'write'):
                    store.put(state['id'],{'state':state,'candidates':rows,'skipped':len(rows)<2,
                                           'rng':rng_states,'generated_tokens':counts,'scored_tokens':scored})
        if prepared:
            from .fast import score_candidate_records, score_candidate_siblings
            with record_errors('candidate score group',','.join(s['id'] for s in group)):
                with timed(totals,'score_cross_state'):
                    # shared_prefix_scoring runs each state's prefix once per model for all its siblings.
                    scorer = score_candidate_siblings if cfg.get('shared_prefix_scoring',False) else score_candidate_records
                    if cfg.get('shared_prefix_scoring',False):
                        scorer(prepared,models,eos,cfg.get('score_tokens_per_batch',8192),cfg.get('score_max_states',48),
                               cfg.get('score_end_token',True))
                    else:
                        scorer(prepared,models,eos,cfg.get('score_tokens_per_batch',8192),cfg.get('score_end_token',True))
                with timed(totals,'write'):
                    for record in prepared:
                        store.put(record['state']['id'],record)
    report_timing(cfg,'collect-candidates',totals,len(pending))
    # Only the requested states: a shared store may also hold other splits' shards (DECISIONS 87).
    wanted = {s['id'] for s in states}
    return [r for r in store.records() if r['state']['id'] in wanted]


def correction_seeds(candidates: list[dict[str,Any]], cfg: dict[str,Any]) -> list[dict[str,Any]]:
    """Preselect prefixes and endpoint Bernoulli sample before observing returns (§7.3).

    correction_seeds='endpoints' (compact) drops the prefix seeds: the head is queried only at
    candidate endpoints, so only those states are rolled out.
    """
    seeds = {}
    prefixes = cfg.get('correction_seeds','prefix_and_endpoints') == 'prefix_and_endpoints'
    for record in candidates:
        state = record['state']
        if prefixes:
            seeds[state['id']] = {**state,'seed_source':'prefix'}
        for candidate in record['candidates']:
            if candidate['terminal']:
                continue
            fraction = int(digest([collection_seed(cfg),'endpoint-selection',candidate['id']])[:12],16)/16**12
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
    numerator = 'student_ref' if teacher_direction else 'teacher_plus'
    jobs = [(seed,digest([seed['id'],replica,source])) for seed in correction_seeds(candidates,cfg)
            for replica in range(cfg['correction_replicates'])]
    # Similar prefix lengths share a generation batch (less left padding); every job's tokens come
    # from its own stream, so the grouping does not change them.
    pending = sorted((job for job in jobs if store.get(job[1]) is None),key=lambda job:(len(job[0]['prefix_ids']),job[1]))
    budget = cfg.get('score_tokens_per_batch',8192)
    totals: dict[str,float] = {}
    # Correction is generation-bound with free GPU memory, so it may use its own (larger) group size.
    size = cfg.get('correction_generation_batch_size') or cfg.get('generation_batch_size',8)
    for group in batches(pending,{**cfg,'generation_batch_size':size},lambda job:len(job[0]['prefix_ids'])+cfg['horizon']-job[0]['n']):
        with record_errors('correction job',','.join(job_id for _,job_id in group)):
            rngs = [stream(collection_seed(cfg),'correction/'+job_id) for _,job_id in group]
            befores = [rng_digest(rng) for rng in rngs]
            prefixes = [seed['prefix_ids'] for seed,_ in group]
            with timed(totals,'generate_'+source):
                suffixes = generate_batch(models[source],prefixes,[cfg['horizon']-seed['n'] for seed,_ in group],eos,rngs,
                                          gpu_sampling=cfg.get('gpu_sampling',False))
            # Teacher returns and rho features for the whole group in length-sorted microbatches.
            with timed(totals,'score_'+numerator):
                numerators = continuation_increments_batch(models[numerator],prefixes,suffixes,eos,budget)
            with timed(totals,'score_teacher_base'):
                bases = continuation_increments_batch(models['teacher_base'],prefixes,suffixes,eos,budget)
            with timed(totals,'features_student_ref'):
                trajectories = trajectory_features_batch(models['student_ref'],prefixes,suffixes,budget)
        write_start = time.perf_counter()
        for (seed,job_id),tokens,rng,before,a,b,features in zip(group,suffixes,rngs,befores,numerators,bases,trajectories):
            with record_errors('correction job',job_id):
                increments = a.cpu()-b.cpu()
                labels = suffix_returns(increments)
                # Binary attachment written first; the JSON commit records its content hash.
                features_sha256 = store.put_tensor(job_id,'features',features)
                budgets = [(cfg['horizon']-seed['n']-t)/cfg['horizon'] for t in range(len(tokens))]
                store.put(job_id,{'id':job_id,'seed':seed,'tokens':tokens,'features_sha256':features_sha256,
                                  'features_shape':list(features.shape),'budgets':budgets,
                                  'increments':increments.cpu().tolist(),'labels':labels.cpu().tolist(),
                                  'terminal_target':0.,'rng_before':before,'rng_after':rng_digest(rng),
                                  'generated_tokens':{source:len(tokens)},
                                  'scored_tokens':{numerator:len(tokens),'teacher_base':len(tokens)}})
        # Labels, the features' copy to host and shard writes.
        totals['labels_and_write'] = totals.get('labels_and_write',0.)+time.perf_counter()-write_start
    report_timing(cfg,'collect-correction',totals,len(pending))
    return store.records()


def correction_features(store: ShardStore, max_bytes: int = 0) -> Callable[[dict[str,Any]],torch.Tensor]:
    """Lazy hash-verified loader for a correction row's detached rho features (§§6.2,10)."""
    from collections import OrderedDict
    cached: OrderedDict[tuple[str,str],torch.Tensor] = OrderedDict()
    used = 0
    def load(row: dict[str,Any]) -> torch.Tensor:
        """Verify an immutable attachment once while it remains in the bounded CPU cache."""
        nonlocal used
        key = (row['id'],row['features_sha256'])
        if key in cached:
            cached.move_to_end(key)
            return cached[key]
        value = store.get_tensor(row['id'],'features',row['features_sha256'])
        size = value.numel()*value.element_size()
        if 0 < size <= max_bytes:
            while cached and used+size > max_bytes:
                _,old = cached.popitem(last=False)
                used -= old.numel()*old.element_size()
            cached[key] = value
            used += size
        return value
    return load
