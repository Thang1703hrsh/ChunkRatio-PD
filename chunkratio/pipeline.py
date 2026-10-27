"""Staged orchestration and manifests for the fixed offline CRPD pipeline (§10)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .artifacts import ShardStore, digest, read_jsonl, stage_key, write_json
from .config import collection_seed
from .collection import (collect_candidates, collect_correction, collect_prefixes, correction_features,
                         prepare_prompts)
from .cost import ForwardMeter
from .evaluation import reward_selector
from .math import diagnostics, routing
from .models import load_models
from .sampling import generate, stream
from .scoring import fingerprint, score_batch
from .targets import build_targets
from .training import ChunkRatioTrainer
from .value_head import fit_head, save_tensor


def pipeline(cfg: dict[str,Any], until: str = 'evaluate') -> dict[str,Any]:
    """Execute requested stage and prerequisites, reusing compatible completed shards (§10)."""
    started = time.perf_counter()
    models,student,tokenizers,info = load_models(cfg)
    meter = ForwardMeter({**models,'student':student})
    eos = info['eos']
    identities = {role:fingerprint(model) for role,model in models.items()}
    route = routing(identities['teacher_plus'],identities['teacher_base'],identities['student_ref'])
    prompts = prepare_prompts(read_jsonl(Path(cfg['prompts'])),collection_seed(cfg))
    contexts = [getattr(model.config,'max_position_embeddings',None) or getattr(model.config,'n_positions',None)
                for model in models.values()]
    if any(context is None for context in contexts):
        raise ValueError('every model needs a verified context length')
    for prompt in prompts:
        if max(prompt['prompt_ids']) >= len(tokenizers[0]):
            raise ValueError('canonical prompt token outside shared vocabulary')
    # Pre-generation eligibility rule (§5.2): prompt + H must fit every model; never truncate later.
    ineligible = sorted(p['id'] for p in prompts if len(p['prompt_ids'])+cfg['horizon'] > min(contexts))
    prompts = [p for p in prompts if p['id'] not in set(ineligible)]
    if not prompts:
        raise ValueError('no prompt fits prompt length + H within every model context')
    output = Path(cfg['output'])
    output.mkdir(parents=True,exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    documents = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((root/'reference').glob('*')) if p.suffix in ('.md','.py','.json')}
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    manifest = {'schema':1,'method_version':'1.1','spec_revision':2,'code_revision':revision,
                'python':platform.python_version(),'torch':torch.__version__,'documents':documents,
                'config':cfg,'models':identities,'tokenizer':info,'split_hash':digest(prompts),
                'ineligible_prompts':{'rule':'len(prompt_ids)+H > min model context','ids':ineligible},
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
    length = 1 if cfg['variant'] == 'crpd_token' else cfg['chunk_length']
    dependencies = {**common,'schedule':length,'states_per_prompt':cfg.get('states_per_prompt')}
    def store(stage: str, metadata: dict[str,Any]) -> ShardStore:
        """Stage namespace follows only its upstream dependencies (§10)."""
        return ShardStore(output/'cache'/stage_key(stage,metadata),metadata)
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
    candidate_deps = {**dependencies,'candidate_rule':'suffix' if cfg['variant'] == 'crpd_suffix' else length,'K':4}
    candidate_store = store('candidates',candidate_deps)
    records = collect_candidates(states,models,cfg,eos,candidate_store)
    if until == 'collect-candidates':
        return {'states':len(records),'skipped':sum(r['skipped'] for r in records),'route':route}
    bypass_head = route == 'exact_zero_residual' or cfg['variant'] in ('crpd_raw','crpd_suffix')
    head,head_meta,corrections = None,{'hash':'exact-zero'},[]
    if not bypass_head:
        correction_deps = {**candidate_deps,'direction':'teacher' if cfg['variant'] == 'crpd_mu_teacher' else 'reference',
                           'endpoint_fraction':cfg['endpoint_fraction'],'replicates':cfg['correction_replicates']}
        correction_store = store('correction',correction_deps)
        corrections = collect_correction(records,models,cfg,eos,correction_store)
        if until == 'collect-correction':
            return {'trajectories':len(corrections),'route':route}
        head_path = output/'cache'/stage_key('head',{**correction_deps,'epochs':cfg['head_epochs'],'lr':cfg['head_learning_rate']})/'head.pt'
        head,head_meta = fit_head(corrections,cfg,head_path,correction_features(correction_store))
    if until in ('collect-correction','fit-head'):
        return {'head':head_meta,'route':route,'bypassed':bypass_head}
    targets = build_targets(records,head,models,cfg,eos,route)
    target_path = output/'cache'/stage_key('targets',{**candidate_deps,'head':head_meta['hash'],
                'variant':cfg['variant'],'audit_replicates':cfg.get('audit_replicates',0)})
    write_json(target_path/'targets.json',targets)
    write_json(output/'target_index.json',{'path':str(target_path/'targets.json'),'head_hash':head_meta['hash']})
    if until == 'build-targets':
        return {'pairs':len(targets['pairs']),'head':head_meta,'route':route}
    selector,selection_cost = (None,{}) if cfg.get('synthetic',False) else reward_selector(cfg,prompts,tokenizers[2])
    trainer = ChunkRatioTrainer(student,models['student_ref'],targets,prompts,cfg,eos,selector)
    policy_deps = {'code_revision':revision,'targets':digest(targets),
                   'config':{k:v for k,v in cfg.items() if k not in ('output','steps')}}
    policy_path = output/'cache'/stage_key('policy',policy_deps)/'resume.pt'
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
    save_tensor(output/'student.pt',{'state_dict':student.state_dict(),'synthetic':cfg.get('synthetic',False)})
    if until == 'train-policy':
        return policy
    weighted = {'excess_bce':0.,'logit_error':0.,'saturation':0.}
    with torch.no_grad():
        for pair in targets['pairs']:
            _,diag = trainer.batch_loss([pair])
            for key,value in diag.items():
                weighted[key] += value*pair['weight']
    observed_mass = []
    for record in targets['records']:
        cs = record['candidates']
        prefix = record['state']['prefix_ids']
        p = score_batch(student,[prefix]*len(cs),[c['tokens'] for c in cs],eos).double().exp().sum().item()
        observed_mass.append({'state_id':record['state']['id'],'student_candidate_mass':p,
                              'reference_candidate_mass':sum(__import__('math').exp(c['logp_student_ref']) for c in cs)})
    generated_counts,scored_counts = {},{}
    all_cost_records = prefix_store.records()+records+corrections+targets['audits']
    for record in all_cost_records:
        for field,destination in (('generated_tokens',generated_counts),('scored_tokens',scored_counts)):
            for role,count in record.get(field,{}).items():
                destination[role] = destination.get(role,0)+count
    result = {'synthetic':cfg.get('synthetic',False),'route':route,'variant':cfg['variant'],'policy':policy,
              'head':head_meta,'pair_diagnostics':weighted,'candidate_coverage':observed_mass,
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
