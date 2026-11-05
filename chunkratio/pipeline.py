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

from .artifacts import ShardStore, code_revision, digest, read_jsonl, source_digest, stage_key, write_json
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


def pipeline(cfg: dict[str,Any], until: str = 'evaluate') -> dict[str,Any]:
    """Execute requested stage and prerequisites, reusing compatible completed shards (§10)."""
    started = time.perf_counter()
    models,student,tokenizers,info = load_models(cfg)
    meter = ForwardMeter({**models,'student':student})
    eos = info['eos']
    identities = {role:fingerprint(model) for role,model in models.items()}
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
    common = {'models':identities,'tokenizer':info,'horizon':cfg['horizon'],'precision':manifest['precision'],
              'split_hash':manifest['split_hash'],'collection_seed':collection_seed(cfg),'schema':1,'code_revision':revision,
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
    prefix_store = store('prefixes',dependencies)
    states = collect_prefixes(prompts,models['student_ref'],cfg,eos,prefix_store)
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
    if until in ('collect-correction','fit-head'):
        return {'head':head_meta,'route':route,'bypassed':bypass_head}
    targets = build_targets(records,head,models,cfg,eos,route)
    target_deps = {**candidate_deps,'head':head_meta['hash'],'variant':cfg['variant'],
                   'audit_replicates':cfg.get('audit_replicates',0)}
    # gamma enters the target key only when set, so gamma = 1 keeps the existing keys (the policy key
    # carries the whole config, gamma included).
    if cfg.get('teacher_gap_scale',1.0) != 1:
        target_deps['teacher_gap_scale'] = cfg['teacher_gap_scale']
    target_path = cache/stage_key('targets',target_deps)
    write_json(target_path/'targets.json',targets)
    write_json(output/'target_index.json',{'path':str(target_path/'targets.json'),'head_hash':head_meta['hash']})
    if until == 'build-targets':
        return {'pairs':len(targets['pairs']),'head':head_meta,'route':route}
    # Targets are built, so the frozen teachers are idle from here on: park them in host memory to
    # free GPU room (two FP32 1.7B teachers = 13.6 GB) for the selection RM and the student optimizer.
    # Their fingerprints are still checked after training.
    for role in ('teacher_plus','teacher_base'):
        models[role].to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    selector,selection_cost = (None,{}) if cfg.get('synthetic',False) else reward_selector(cfg,prompts,tokenizers[2])
    trainer = ChunkRatioTrainer(student,models['student_ref'],targets,prompts,cfg,eos,selector)
    policy_deps = {'code_revision':revision,'targets':digest(targets),
                   'config':{k:v for k,v in cfg.items() if k not in ('output','steps')}}
    policy_path = cache/stage_key('policy',policy_deps)/'resume.pt'
    policy = trainer.train(policy_path)
    write_json(output/'policy_index.json',{'path':str(policy_path),'code_revision':revision})
    frozen_after = {role:fingerprint(model) for role,model in models.items()}
    assert identities == frozen_after,'frozen model parameters changed'
    if head is not None:
        assert fingerprint(head) == head_meta['hash'] and all(p.grad is None for p in head.parameters())
    # Export only the student; pretrained models use their ordinary deployment format.
    if not cfg.get('synthetic',False):
        student.save_pretrained(output/'student')
        tokenizers[2].save_pretrained(output/'student')
    if cfg.get('save_student_state_dict',True) or cfg.get('synthetic',False):
        save_tensor(output/'student.pt',{'state_dict':student.state_dict(),'synthetic':cfg.get('synthetic',False)})
    if until == 'train-policy':
        return policy
    # Measure-weighted pair diagnostics in microbatches (each pair keeps its hierarchical weight).
    weighted = {'excess_bce':0.,'logit_error':0.,'saturation':0.}
    size = cfg.get('diagnostic_batch_size',32)
    pairs = targets['pairs']
    # With a token budget the no-grad pass groups pairs by length like training does (twice the training
    # budget: no activations are kept for backward); the weighted sums do not depend on the grouping.
    budget = cfg.get('policy_tokens_per_microbatch')
    widths = [len(p['state']['prefix_ids'])+max(len(p['c']['tokens']),len(p['d']['tokens'])) for p in pairs]
    groups = ([[pairs[i] for i in g] for g in length_groups(widths,2*budget)] if budget else
              [pairs[offset:offset+size] for offset in range(0,len(pairs),size)])
    with torch.no_grad():
        for chunk in groups:
            _,metrics = trainer.pair_terms(chunk)
            weights = torch.tensor([p['weight'] for p in chunk],dtype=torch.float64)
            for key in weighted:
                weighted[key] += float((metrics[key].detach().double().cpu()*weights).sum())
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
    all_cost_records = prefix_store.records()+records+corrections+targets['audits']
    for record in all_cost_records:
        for field,destination in (('generated_tokens',generated_counts),('scored_tokens',scored_counts)):
            for role,count in record.get(field,{}).items():
                destination[role] = destination.get(role,0)+count
    result = {'synthetic':cfg.get('synthetic',False),'route':route,'variant':cfg['variant'],'policy':policy,
              'head':head_meta,'pair_diagnostics':weighted,'correction_audit':sibling_audit,'candidate_coverage':observed_mass,
              'audit':targets['audits'],'counts':{'states':len(states),'pairs':len(targets['pairs']),
                                                'correction_trajectories':len(corrections)},
              'cost':{'collection_generated_tokens':generated_counts,'collection_scored_tokens':scored_counts,
                      'wall_seconds_this_invocation':time.perf_counter()-started,
                      'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                      'gpu_hours':None,'forward_input_tokens_this_invocation':meter.counts,
                      'selection_this_invocation':selection_cost,
                      'note':'forward counts include repeated prefix prefill; GPU-hours require hardware profiling'}}
    meter.close()
    write_json(output/'diagnostics.json',result)
    return result
