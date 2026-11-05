"""Isolated, resumable x10 experiment for the pinned HDA pair (not a full-data run).

Run ``python -m chunkratio.fast_run --help``. No publication or upload operations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .artifacts import atomic_bytes, code_revision, digest, read_json, read_jsonl, write_json
from .config import validate


def file_sha256(path: Path) -> str:
    """Stream an artifact hash without copying model files into memory."""
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def fast_config(base: dict[str,Any], root: Path, prompts: Path, variant: str) -> dict[str,Any]:
    """Retain the pinned models, split, scoring precision and teacher provenance."""
    if variant not in ('crpd_main','crpd_raw'):
        raise ValueError('fast x10 supports main and raw only')
    cfg = copy.deepcopy(base)
    cfg.update(variant=variant,output=(root/variant).as_posix(),cache_root=(root/'cache').as_posix(),
               prompts=prompts.as_posix(),candidates=2,states_per_prompt=2,
               correction_seeds='endpoints',endpoint_fraction=1.,head_positions='boundary',
               correction_replicates=1,steps=100,batch_size=64,selection_interval=50,
               selection_baseline=True,selection_fallback_to_reference=False,
               alpha=.03,learning_rate=1e-6,lr_schedule='cosine',warmup_ratio=.1,
               generation_batch_size=48,policy_microbatch_size=6,selection_generation_batch_size=16,
               batch_candidate_scoring=True,score_tokens_per_batch=8192,head_feature_cache_mb=3072,
               save_student_state_dict=False)
    cfg.setdefault('pilot_assumptions',{})['fast_x10'] = (
        'K=2, 2 states/prompt; endpoint-only R=1 correction, boundary-only head loss; '
        '100 steps x 64 pairs; fixed alpha=.03, lr=1e-6; validation selects checkpoints, '
        'test benchmarks never select; cross-state scoring can differ by FP32 rounding.')
    validate(cfg)
    return cfg


def output_fingerprints(output: Path) -> dict[str,dict[str,Any]]:
    """Bind a completion marker to the actual exported student and diagnostics."""
    files = [output/'diagnostics.json',output/'resolved_config.json',output/'manifest.json']
    files += sorted((output/'student').glob('*'))
    if not (output/'student/config.json').is_file() or not any(
            p.suffix in ('.safetensors','.bin') for p in files):
        raise ValueError('complete pretrained student export required')
    return {p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':file_sha256(p)}
            for p in files if p.is_file()}


def completed(cfg: dict[str,Any]) -> bool:
    """Refuse stale or modified exports; an interrupted run has no completion marker."""
    output = Path(cfg['output'])
    marker = output/'fast_completed.json'
    if not marker.exists():
        return False
    result = read_json(marker)
    if result['config_hash'] != digest(cfg) or result['code_revision'] != code_revision():
        raise ValueError('completed output belongs to another config/source; use an isolated output')
    if result['artifacts'] != output_fingerprints(output):
        raise ValueError('completed artifact was changed or corrupted')
    return True


def run(base_config: Path, root: Path, variant: str = 'both') -> dict[str,Any]:
    """One process per variant; all upstream collection stages execute once or resume."""
    if root.is_absolute():
        raise ValueError('output root must be relative')
    base = read_json(base_config)
    from .pilot import HDA_CHECKPOINTS
    for role,entry in HDA_CHECKPOINTS.items():
        actual = base['models'][role]
        if (actual['path'],actual['revision']) != (entry['repo'],entry['revision']):
            raise ValueError('fast x10 must use the current pinned HDA trio')
    source = Path(base['prompts'])
    rows = read_jsonl(source)
    counts = {split:sum(row.get('split') == split for row in rows) for split in ('train','head_val','select')}
    if counts != {'train':640,'head_val':160,'select':64}:
        raise ValueError(f'expected x10 split counts 640/160/64, got {counts}')
    root.mkdir(parents=True,exist_ok=True)
    prompts = root/'prompts.jsonl'
    if prompts.exists() and file_sha256(prompts) != file_sha256(source):
        raise ValueError('prompt snapshot changed; use a new run directory')
    if not prompts.exists():
        atomic_bytes(prompts,source.read_bytes())
    recipe = {'schema':1,'label':'CRPD fast x10 v1; compact deviation, not protocol K=4',
              'code_revision':code_revision(),'base_config_hash':digest(base),
              'prompt_sha256':file_sha256(prompts),'splits':counts,'variants':['crpd_main','crpd_raw'],
              'teacher_reference_verified':base.get('teacher_dpo_reference_verified',False),
              'teacher_reference_assumption':base.get('teacher_dpo_reference_assumption')}
    recipe_path = root/'recipe.json'
    if recipe_path.exists() and read_json(recipe_path) != recipe:
        raise ValueError('recipe/source changed; use a new run directory')
    write_json(recipe_path,recipe)
    results = {}
    from .pipeline import pipeline
    import gc
    import torch
    for name in (('crpd_main','crpd_raw') if variant == 'both' else (variant,)):
        cfg = fast_config(base,root,prompts,name)
        config_path = root/f'{name}.json'
        write_json(config_path,cfg)
        if completed(cfg):
            results[name] = read_json(Path(cfg['output'])/'diagnostics.json')
            print(json.dumps({'fast_run':'already_completed','variant':name}),flush=True)
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        print(json.dumps({'fast_run':'start','variant':name,'config':str(config_path)}),flush=True)
        result = pipeline(cfg,'evaluate')
        output = Path(cfg['output'])
        write_json(output/'fast_completed.json',{
            'schema':1,'config_hash':digest(cfg),'code_revision':code_revision(),
            'wall_seconds':time.perf_counter()-start,'artifacts':output_fingerprints(output)})
        results[name] = result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({'fast_run':'completed','variant':name,
                          'policy':{k:v for k,v in result['policy'].items() if k != 'history'},
                          'counts':result['counts']}),flush=True)
    return results


def main() -> None:
    """Run the bounded x10 recipe from the Windows/Linux command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('both','crpd_main','crpd_raw'),default='both')
    args = parser.parse_args()
    run(args.base_config,args.output,args.variant)


if __name__ == '__main__':
    main()
