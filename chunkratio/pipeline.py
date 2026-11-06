"""Staged orchestration and manifests for the fixed offline CRPD pipeline (§10)."""
from __future__ import annotations

import hashlib
import os
import importlib.metadata
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .artifacts import ShardStore, cache_revision, code_revision, digest, read_jsonl, source_digest, stage_key, write_json
from .config import collection_seed
from .collection import (collect_candidates, collect_correction, collect_prefixes, correction_features,
                         prepare_prompts)
from .cost import ForwardMeter
from .diagnostics import correction_audit
from .evaluation import reward_selector
from .math import diagnostics, routing
from .models import load_models
from .sampling import generate, stream
from .scoring import fingerprint, length_groups, score_batch
from .targets import build_targets
from .training import ChunkRatioTrainer
from .value_head import fit_head, save_tensor


def eligible_prompts(cfg: dict[str,Any], models: dict[str,Any], tokenizers: list[Any]) -> tuple[list[dict[str,Any]],list[str]]:
    """Split before generation and drop prompts whose prompt + H exceeds a model context (§§5.2,7.1)."""
    prompts = prepare_prompts(read_jsonl(Path(cfg['prompts'])),collection_seed(cfg))
    contexts = [getattr(model.config,'max_position_embeddings',None) or getattr(model.config,'n_positions',None)
                for model in models.values()]
    if any(context is None for context in contexts):
        raise ValueError('every model needs a verified context length')
    for prompt in prompts:
        if max(prompt['prompt_ids']) >= len(tokenizers[0]):
            raise ValueError('canonical prompt token outside shared vocabulary')
    # Pre-generation eligibility rule (§5.2): prompt + H must fit every model; never truncate later.
    # An opt-in max_prompt_tokens also drops rare very long prompts: right padding gives every row of
    # a batch the longest width, so they set peak GPU memory. The rule never reads an outcome.
    limit = cfg.get('max_prompt_tokens') or float('inf')
    ineligible = sorted(p['id'] for p in prompts if len(p['prompt_ids'])+cfg['horizon'] > min(contexts) or len(p['prompt_ids']) > limit)
    prompts = [p for p in prompts if p['id'] not in set(ineligible)]
    if not prompts:
        raise ValueError('no prompt fits prompt length + H within every model context')
    return prompts,ineligible


def pair_diagnostics(trainer: ChunkRatioTrainer, pairs: list[dict[str,Any]], cfg: dict[str,Any]) -> dict[str,float]:
    """Measure-weighted excess BCE, logit error and saturation of the trained student (DECISIONS 88).

    Every state's candidates that appear in a pair are scored once per model (trainer.candidate_logps:
    the training rule for precision, end token, shared prefix and cached rho), instead of twice per pair;
    x10 has 7,108 pairs over 4,770 candidates. Each pair then gets pair_loss/diagnostics from those
    scores with its hierarchical weight. Runs after training, so no gradient or target changes.
    """
    from .math import pair_loss
    weighted = {'excess_bce':0.,'logit_error':0.,'saturation':0.}
    if not pairs:
        return weighted
    states: dict[str,tuple[dict[str,Any],dict[str,dict[str,Any]]]] = {}
    for pair in pairs:
        entry = states.setdefault(pair['state']['id'],(pair['state'],{}))
        for side in ('c','d'):
            entry[1].setdefault(pair[side]['id'],pair[side])
    items = [(state,list(candidates.values())) for state,candidates in states.values()]
    most = max(len(candidates) for _,candidates in items)
    widths = [len(state['prefix_ids'])+max(len(c['tokens']) for c in candidates) for state,candidates in items]
    # Chunk rows per forward as in the former pair grouping (two rows per pair, twice the training budget,
    # or diagnostic_batch_size pairs); the scores do not depend on the grouping beyond rounding.
    budget = cfg.get('policy_tokens_per_microbatch')
    if budget:
        groups = length_groups(widths,max(1,4*budget//most))
    else:
        per = max(1,2*cfg.get('diagnostic_batch_size',32)//most)
        groups = [list(range(i,min(i+per,len(items)))) for i in range(0,len(items),per)]
    scores: dict[tuple[str,str],tuple[float,float]] = {}
    for group in groups:
        chunk = [items[i] for i in group]
        for (state,candidates),(student,reference) in zip(chunk,trainer.candidate_logps(chunk)):
            for candidate,s,r in zip(candidates,student,reference):
                scores[(state['id'],candidate['id'])] = (s,r)
    dtype = torch.float64 if next(trainer.student.parameters()).dtype == torch.float64 else torch.float32
    column = lambda side,j: torch.tensor([scores[(p['state']['id'],p[side]['id'])][j] for p in pairs],dtype=dtype)
    target = lambda side: torch.tensor([p[side]['target_score'] for p in pairs],dtype=dtype)
    losses,margin,u = pair_loss(column('c',0),column('d',0),column('c',1),column('d',1),target('c'),target('d'),cfg['alpha'])
    metrics = diagnostics(losses,margin,u)
    weights = torch.tensor([p['weight'] for p in pairs],dtype=torch.float64)
    return {key:float((metrics[key].double()*weights).sum()) for key in weighted}


def pipeline(cfg: dict[str,Any], until: str = 'evaluate') -> dict[str,Any]:
    """Execute requested stage and prerequisites, reusing compatible completed shards (§10)."""
    started = time.perf_counter()
    # Wall seconds per stage of this invocation (diagnostics cost.stage_seconds); timing only.
    stage_seconds: dict[str,float] = {}
    clock = [started]
    def lap(name: str) -> None:
        """Add the time since the previous lap to `name`."""
        now = time.perf_counter()
        stage_seconds[name] = stage_seconds.get(name,0.)+now-clock[0]
        clock[0] = now
    models,student,tokenizers,info = load_models(cfg)
    lap('load_models')
    meter = ForwardMeter({**models,'student':student})
    eos = info['eos']
    identities = {role:fingerprint(model) for role,model in models.items()}
    lap('fingerprint')
    route = routing(identities['teacher_plus'],identities['teacher_base'],identities['student_ref'])
    prompts,ineligible = eligible_prompts(cfg,models,tokenizers)
    output = Path(cfg['output'])
    output.mkdir(parents=True,exist_ok=True)
    # A shared cache_root lets variants of one pair reuse every stage whose key matches (§10).
    cache = Path(cfg['cache_root']) if cfg.get('cache_root') else output/'cache'
    root = Path(__file__).resolve().parents[1]
    documents = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((root/'reference').glob('*')) if p.suffix in ('.md','.py','.json')}
    revision = code_revision()
    manifest = {'schema':1,'method_version':'1.1','spec_revision':2,'code_revision':revision,
                'python':platform.python_version(),'torch':torch.__version__,'documents':documents,
                'config':cfg,'models':identities,'tokenizer':info,'split_hash':digest(prompts),
                'ineligible_prompts':{'rule':'len(prompt_ids)+H > min model context or len(prompt_ids) > max_prompt_tokens',
                                      'max_prompt_tokens':cfg.get('max_prompt_tokens'),'ids':ineligible},
                'route':route,'precision':str(next(student.parameters()).dtype),'feature_dtype':'float32',
                'terminal_convention':'sampled EOS included once; H after prompt; no forced EOS',
                'seed_streams':['split','prefix','candidate/student_ref','candidate/teacher_plus',
                                'endpoint-selection','correction','head-init','policy-sampler','audit'],
                'synthetic':cfg.get('synthetic',False)}
    manifest['dependencies'] = {}
    for package in ('torch','transformers','datasets','numpy','pytest','PyYAML'):
        try:
            manifest['dependencies'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest['dependencies'][package] = None
    write_json(output/'manifest.json',manifest)
    write_json(output/'resolved_config.json',cfg)
    if until == 'validate-config':
        return manifest
    # Collection, head and target keys carry the source/library identity, not the git commit (DECISIONS 87):
    # a commit alone no longer forces a recollection; the manifest still records the full code revision.
    common = {'models':identities,'tokenizer':info,'horizon':cfg['horizon'],'precision':manifest['precision'],
              'split_hash':manifest['split_hash'],'collection_seed':collection_seed(cfg),'schema':1,
              'cache_revision':cache_revision(),'rng_record':'sha256',
              'documents':documents,'feature_convention':'last-valid-token/final-layer/budget'}
    # The sampler decides which tokens are drawn, so every collection key carries it. The GPU rule enters
    # the keys only when set, so default runs keep their stage keys; batch grouping never enters them.
    if cfg.get('gpu_sampling',False):
        common['sampler'] = 'gpu-inverse-cdf-fp64'
    length = 1 if cfg['variant'] == 'crpd_token' else cfg['chunk_length']
    dependencies = {**common,'schedule':length,'states_per_prompt':cfg.get('states_per_prompt')}
    def store(stage: str, metadata: dict[str,Any]) -> ShardStore:
        """Stage namespace follows only its upstream dependencies (§10)."""
        return ShardStore(cache/stage_key(stage,metadata),metadata)
    if route == 'no_alignment_update':
        # Routing precedes any correction collection and ANY optimization.
        if not cfg.get('synthetic',False):
            student.save_pretrained(output/'student')
            tokenizers[2].save_pretrained(output/'student')
        save_tensor(output/'student.pt',{'state_dict':student.state_dict(),'synthetic':cfg.get('synthetic',False)})
        result = {'route':route,'optimizer_steps':0,'collection_skipped':True}
        write_json(output/'diagnostics.json',result)
        return result
    # A collection shard holds part of the prompts, so a sharded worker stops at its collection stage; the
    # run that builds targets and trains is started without a shard once every worker has finished.
    if cfg.get('collection_shard') and until not in ('collect-prefixes','collect-candidates','collect-correction'):
        raise ValueError('a collection_shard worker runs collect-prefixes, collect-candidates or collect-correction only')
    prefix_store = store('prefixes',dependencies)
    lap('setup')
    states = collect_prefixes(prompts,models['student_ref'],cfg,eos,prefix_store)
    lap('collect_prefixes')
    if until == 'collect-prefixes':
        return {'states':len(states),'route':route}
    candidate_deps = {**dependencies,'candidate_rule':'suffix' if cfg['variant'] == 'crpd_suffix' else length,
                      'K':cfg['candidates']}
    # Unscored end-of-turn tokens change every stored chunk score, so they enter the candidate key (and
    # every key built on it); the protocol rule keeps its keys.
    if not cfg.get('score_end_token',True):
        candidate_deps['end_token'] = 'unscored'
    candidate_store = store('candidates',candidate_deps)
    records = collect_candidates(states,models,cfg,eos,candidate_store)
    lap('collect_candidates')
    if until == 'collect-candidates':
        return {'states':len(records),'skipped':sum(r['skipped'] for r in records),'route':route}
    bypass_head = route == 'exact_zero_residual' or cfg['variant'] in ('crpd_raw','crpd_suffix')
    head,head_meta,corrections = None,{'hash':'exact-zero'},[]
    if not bypass_head:
        correction_deps = {**candidate_deps,'direction':'teacher' if cfg['variant'] == 'crpd_mu_teacher' else 'reference',
                           'endpoint_fraction':cfg['endpoint_fraction'],'replicates':cfg['correction_replicates']}
        # Compact settings enter the keys only when set, so protocol runs keep their stage keys.
        if cfg.get('correction_seeds','prefix_and_endpoints') != 'prefix_and_endpoints':
            correction_deps['seeds'] = cfg['correction_seeds']
        correction_store = store('correction',correction_deps)
        corrections = collect_correction(records,models,cfg,eos,correction_store)
        if until == 'collect-correction':
            return {'trajectories':len(corrections),'route':route}
        head_deps = {**correction_deps,'epochs':cfg['head_epochs'],'lr':cfg['head_learning_rate'],
                     'batch':cfg.get('head_batch_size')}
        if cfg.get('head_positions','all') != 'all':
            head_deps['positions'] = cfg['head_positions']
        head_path = cache/stage_key('head',head_deps)/'head.pt'
        head,head_meta = fit_head(corrections,cfg,head_path,correction_features(
            correction_store,cfg.get('head_feature_cache_mb',0)*1024**2))
    lap('correction_and_head')
    if until in ('collect-correction','fit-head'):
        return {'head':head_meta,'route':route,'bypassed':bypass_head}
    targets = build_targets(records,head,models,cfg,eos,route)
    target_deps = {**candidate_deps,'head':head_meta['hash'],'variant':cfg['variant'],
                   'audit_replicates':cfg.get('audit_replicates',0)}
    # gamma enters the target key only when set, so gamma = 1 keeps the existing keys (the policy key
    # carries the whole config, gamma included).
    if cfg.get('teacher_gap_scale',1.0) != 1:
        target_deps['teacher_gap_scale'] = cfg['teacher_gap_scale']
    # The pair rule enters the target key only when it filters, so 'all' keeps every existing key.
    if cfg.get('pair_sources','all') != 'all':
        target_deps['pair_sources'] = cfg['pair_sources']
    target_path = cache/stage_key('targets',target_deps)
    write_json(target_path/'targets.json',targets)
    write_json(output/'target_index.json',{'path':str(target_path/'targets.json'),'head_hash':head_meta['hash']})
    if until == 'build-targets':
        return {'pairs':len(targets['pairs']),'head':head_meta,'route':route}
    # Targets are built, so the frozen teachers are idle from here on: park them in host memory to
    # free GPU room (two FP32 1.7B teachers = 13.6 GB) for the selection RM and the student optimizer.
    # Their fingerprints are still checked after training.
    # Opt-in release_teachers (DECISIONS 94): check their fingerprints now and drop them instead, so a process
    # holds no teacher at all while it trains (two FP32 8B teachers = 64 GB of host RAM per process otherwise).
    # Nothing after this point reads a teacher.
    released = []
    if cfg.get('release_teachers',False):
        for role in ('teacher_plus','teacher_base'):
            assert fingerprint(models[role]) == identities[role],'frozen model parameters changed'
            del models[role]
            released.append(role)
        import gc
        gc.collect()
    else:
        for role in ('teacher_plus','teacher_base'):
            models[role].to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lap('build_targets')
    selector,selection_cost = (None,{}) if cfg.get('synthetic',False) else reward_selector(cfg,prompts,tokenizers[2])
    trainer = ChunkRatioTrainer(student,models['student_ref'],targets,prompts,cfg,eos,selector)
    policy_deps = {'code_revision':revision,'targets':digest(targets),
                   'config':{k:v for k,v in cfg.items() if k not in ('output','steps')}}
    policy_path = cache/stage_key('policy',policy_deps)/'resume.pt'
    lap('setup')
    policy = trainer.train(policy_path)
    lap('train_policy')
    write_json(output/'policy_index.json',{'path':str(policy_path),'code_revision':revision})
    frozen_after = {role:fingerprint(model) for role,model in models.items()}
    assert {role:identities[role] for role in models} == frozen_after,'frozen model parameters changed'
    assert set(frozen_after)|set(released) == set(identities)
    if head is not None:
        assert fingerprint(head) == head_meta['hash'] and all(p.grad is None for p in head.parameters())
    lap('fingerprint')
    # Export only the student; pretrained models use their ordinary deployment format.
    if not cfg.get('synthetic',False):
        student.save_pretrained(output/'student')
        tokenizers[2].save_pretrained(output/'student')
    if cfg.get('save_student_state_dict',True) or cfg.get('synthetic',False):
        save_tensor(output/'student.pt',{'state_dict':student.state_dict(),'synthetic':cfg.get('synthetic',False)})
    lap('export_student')
    if until == 'train-policy':
        return policy
    weighted = pair_diagnostics(trainer,targets['pairs'],cfg)
    # Endpoint -> state from the candidate records, so endpoint-only correction is audited as well.
    parents = {c['id']:r['state']['id'] for r in records for c in r['candidates']}
    sibling_audit = correction_audit(targets['pairs'],[r for r in corrections if r['seed']['split'] == 'head_val'],head,
                                     correction_features(correction_store) if head is not None else None,
                                     collection_seed(cfg),parents=parents)
    # Student mass on each state's candidates; with batch_candidate_scoring the rows of many states share
    # length-sorted forwards (FP32 rounding differences only), otherwise one forward per state.
    rows = [(r,c) for r in targets['records'] for c in r['candidates']]
    def chunk_logps(model: torch.nn.Module) -> list[float]:
        """Actual chunk log-probs (end-of-turn token scored once), whatever the training score rule."""
        logps: list[float] = [0.]*len(rows)
        if cfg.get('batch_candidate_scoring',False):
            lengths = [len(r['state']['prefix_ids'])+len(c['tokens']) for r,c in rows]
            for group in length_groups(lengths,cfg.get('score_tokens_per_batch',8192)):
                scores = score_batch(model,[rows[i][0]['state']['prefix_ids'] for i in group],
                                     [rows[i][1]['tokens'] for i in group],eos,score_eos=True).double().tolist()
                for i,score in zip(group,scores):
                    logps[i] = score
        else:
            offset = 0
            for record in targets['records']:
                cs = record['candidates']
                scores = score_batch(model,[record['state']['prefix_ids']]*len(cs),[c['tokens'] for c in cs],eos,
                                     score_eos=True).double().tolist()
                logps[offset:offset+len(cs)] = scores
                offset += len(cs)
        return logps
    student_logps = chunk_logps(student)
    # The cached rho scores follow score_end_token; with it off they omit the end token, so rho is
    # rescored under the same rule as the student (student = rho then gives equal masses).
    reference_logps = (chunk_logps(models['student_ref']) if not cfg.get('score_end_token',True) else
                       [c['logp_student_ref'] for _,c in rows])
    observed_mass, offset = [],0
    for record in targets['records']:
        cs = record['candidates']
        span = slice(offset,offset+len(cs))
        offset += len(cs)
        observed_mass.append({'state_id':record['state']['id'],
                              'student_candidate_mass':sum(math.exp(v) for v in student_logps[span]),
                              'reference_candidate_mass':sum(math.exp(v) for v in reference_logps[span]),
                              'end_token_scored':True})
    generated_counts,scored_counts = {},{}
    lap('post_training_diagnostics')
    # Cost counts cover the configured splits only, like the states and candidates (DECISIONS 87).
    splits = tuple(cfg.get('collection_splits',('train','head_val')))
    prefix_records = [r for r in prefix_store.records() if r['prompt']['split'] in splits]
    all_cost_records = prefix_records+records+corrections+targets['audits']
    for record in all_cost_records:
        for field,destination in (('generated_tokens',generated_counts),('scored_tokens',scored_counts)):
            for role,count in record.get(field,{}).items():
                destination[role] = destination.get(role,0)+count
    result = {'synthetic':cfg.get('synthetic',False),'route':route,'variant':cfg['variant'],'policy':policy,
              'teachers_released_after_targets':released,
              'head':head_meta,'pair_diagnostics':weighted,'correction_audit':sibling_audit,'candidate_coverage':observed_mass,
              'audit':targets['audits'],'counts':{'states':len(states),'pairs':len(targets['pairs']),
                                                'correction_trajectories':len(corrections)},
              'cost':{'collection_generated_tokens':generated_counts,'collection_scored_tokens':scored_counts,
                      'wall_seconds_this_invocation':time.perf_counter()-started,'stage_seconds':stage_seconds,
                      'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                      'gpu_hours':None,'forward_input_tokens_this_invocation':meter.counts,
                      'selection_this_invocation':selection_cost,
                      'note':'forward counts include repeated prefix prefill; GPU-hours require hardware profiling'}}
    meter.close()
    write_json(output/'diagnostics.json',result)
    return result
