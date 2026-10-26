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
