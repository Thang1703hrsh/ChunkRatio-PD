"""Qwen3 pilot helpers: outcome-independent subset and configs that validate in pilot mode (no torch)."""
import json
from pathlib import Path

import pytest

from chunkratio.config import validate
from chunkratio.pilot import PILOT_ASSUMPTIONS, pilot_config, pilot_subset, preparation_config


def rows() -> list[dict]:
    """Canonical rows over three splits plus test."""
    return [{'id':f'r{i}','group_id':f'g{i//2}','split':split,'prompt_ids':[1,i]}
            for i,split in enumerate(['train']*40+['head_val']*10+['select']*10+['test']*10)]


def test_subset_is_pre_registered() -> None:
    """Deterministic by group id and seed, sizes respected, duplicate groups kept together, test excluded."""
    sizes = {'train':5,'head_val':2,'select':1}
    first = pilot_subset(rows(),0,sizes)
    assert first == pilot_subset(list(reversed(rows())),0,sizes)
    counts = {s:len({r['group_id'] for r in first if r['split'] == s}) for s in sizes}
    assert counts == sizes and all(r['split'] != 'test' for r in first)
    assert all(sum(r['group_id'] == g for r in first) == 2 for g in {r['group_id'] for r in first})
    assert pilot_subset(rows(),1,sizes) != first


def test_pilot_configs_validate_and_record_assumptions() -> None:
    """Training inputs are complete; evaluation fields may stay null; unverified DPO reference fails."""
    protocol = json.loads(Path('config/crpd/narval/qwen3_1p7b_to_0p6b.json').read_text(encoding='utf-8'))
    checkpoints = {role:{'path':f'output/pilot/{role}','revision':'fp'+role} for role in ('teacher_plus','teacher_base','student_ref')}
    revisions = {k:'sha' for k in ('reward_model','teacher_base_model','student_base_model','sft_dataset','preference_dataset')}
    for variant in ('crpd_main','crpd_raw','crpd_anchor'):
        cfg = pilot_config(protocol,variant,checkpoints,revisions,'splitsha',True)
        validate(cfg)
        assert cfg['pilot_assumptions'] == PILOT_ASSUMPTIONS and cfg['narval']['evaluation']['mt_bench']['judge'] is None
        assert not Path(cfg['output']).is_absolute() and cfg['cache_root'].endswith('pilot/cache')
    with pytest.raises(ValueError,match='verified'):
        validate(pilot_config(protocol,'crpd_main',checkpoints,revisions,'splitsha',False))
    with pytest.raises(ValueError,match='assumed'):
        validate({**pilot_config(protocol,'crpd_main',checkpoints,revisions,'splitsha',True),'pilot_assumptions':{}})
    dpo = preparation_config('train-teacher-dpo','m','r','d','o')
    assert dpo['beta'] == .05 and dpo['eos_token'] == '<|im_end|>' and dpo['allow_tf32']
