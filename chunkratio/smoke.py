"""No-download tiny CPU stages and every declared variant (§12)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from .artifacts import read_json, write_json
from .config import STAGES, VARIANTS
from .models import load_models
from .pipeline import pipeline
from .sampling import generate, stream
from .scoring import fingerprint, score_batch


def smoke(output: Path, template: dict[str,Any]) -> dict[str,Any]:
    """Run all stages/variants, resume checks, standalone export/reload and generation (§12)."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = {}
    for variant in VARIANTS:
        cfg = copy.deepcopy(template)
        cfg.update(variant=variant,output=str(output/variant),device='cpu',synthetic=True)
        for stage in STAGES:
            pipeline(cfg,stage)
        result = read_json(Path(cfg['output'])/'diagnostics.json')
        _,fresh,_,info = load_models(cfg)
        state = torch.load(Path(cfg['output'])/'student.pt',map_location='cpu',weights_only=True)
        fresh.load_state_dict(state['state_dict'])
        before = fingerprint(fresh)
        restored = copy.deepcopy(fresh)
        restored.load_state_dict(state['state_dict'])
        prefixes,chunks = [[1,2]],[[3,0]]
        torch.testing.assert_close(score_batch(fresh,prefixes,chunks,info['eos']),
                                   score_batch(restored,prefixes,chunks,info['eos']),atol=0,rtol=0)
        del fresh,state
        response = generate(restored,[1,2],cfg['horizon'],info['eos'],stream(5,'standalone-export'))
        assert response and fingerprint(restored) == before
        assert result['policy']['optimizer_steps'] == (0 if variant == 'crpd_audit' else cfg['steps'])
        results[variant] = {'status':'passed','optimizer_steps':result['policy']['optimizer_steps'],
                            'pairs':result['counts']['pairs'],'standalone_generated_tokens':response}
    final = {'synthetic':True,'status':'passed','variants':results,'summary':'SMOKE PASSED: 7 variants, all stages, export/reload and standalone generation'}
    write_json(output/'STATUS.json',final)
    return final


def smoke_narval(output: Path) -> dict[str,Any]:
    """narval-tiny: tiny Qwen3-style models through every new stage; synthetic, not a result (§12)."""
    import json
    from .config import validate
    from .data import prepare_data
    from .diagnostics import diagnose
    from .evaluation import NARVAL_OPEN_LLM, report_narval
    from .tiny import build_tiny_qwen3, synthetic_ultrafeedback
    torch.set_num_threads(1)
    paths = build_tiny_qwen3(output/'models')
    synthetic_ultrafeedback(output/'data'/'uf_train.jsonl')
    prepare_data(output/'data'/'uf_train.jsonl',str(paths['student_ref']),'local',output/'data'/'train.jsonl',0,
                 template_kwargs={'enable_thinking':False},head_val_fraction=.2,select_fraction=.1)
    cfg = read_json(Path('config/crpd/narval/qwen3_8b_to_1p7b.json'))
    cfg.update(synthetic=True,device='cpu',horizon=8,chunk_length=2,batch_size=4,steps=2,head_epochs=2,
               head_batch_size=2,generation_batch_size=4,states_per_prompt=2,selection_interval=1,
               prompts=str(output/'data'/'train.jsonl'),
               models={role:{'path':str(path),'revision':'local'} for role,path in paths.items()})
    validate(cfg)
    diagnostic = diagnose(cfg,2,4,output/'diagnostics')
    results = {}
    for variant in cfg['narval']['variants']:
        run = {**cfg,'variant':variant,'output':str(output/variant)}
        validate(run)
        result = pipeline(run,'evaluate')
        _,fresh,_,info = load_models(run)
        fresh.load_state_dict(torch.load(output/variant/'student.pt',map_location='cpu',weights_only=True)['state_dict'])
        response = generate(fresh,[1,5,8,2,1,6,3,4],run['horizon'],info['eos'],stream(5,'narval-standalone'))
        assert result['policy']['optimizer_steps'] == run['steps'] and response and result['route'] == 'reference_mean'
        results[variant] = {'status':'passed','optimizer_steps':result['policy']['optimizer_steps'],
                            'pairs':result['counts']['pairs'],'standalone_generated_tokens':response}
    reported = [{'source':'NARVAL Table 1','reported':True,'synthetic':True,'method':'narval (fake)',
                 'pair':cfg['narval']['pair'],'metrics':{m:0. for m,_ in NARVAL_OPEN_LLM}}]
    (output/'report').mkdir(parents=True,exist_ok=True)
    (output/'report'/'reported.jsonl').write_text(json.dumps(reported[0])+'\n',encoding='utf-8')
    (output/'report'/'narval_tables.md').write_text(report_narval([],reported,allow_incomplete=True),encoding='utf-8')
    final = {'synthetic':True,'status':'passed','protocol':'narval-tiny','diagnostic':diagnostic,'variants':results,
             'summary':'NARVAL-TINY SMOKE PASSED: prepare-data, diagnose-displacement, 3 variants, report --layout narval'}
    write_json(output/'STATUS.json',final)
    return final
