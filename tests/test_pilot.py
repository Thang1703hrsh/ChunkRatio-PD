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
    assert dpo['autocast_dtype'] == 'bfloat16'
    tuned = pilot_config(protocol,'crpd_main',checkpoints,revisions,'splitsha',True)
    assert (tuned['lr_schedule'],tuned['warmup_ratio'],tuned['policy_autocast_dtype']) == ('cosine',.1,'bfloat16')
    assert tuned['alpha'] == protocol['alpha'] and tuned['policy_microbatch_size'] == 8


@pytest.mark.parametrize('pair,eos',[('qwen3_8b_to_1p7b','<|im_end|>'),('llama31_8b_to_llama32_1b','<|eot_id|>')])
def test_8b_pair_configs_validate_with_their_own_tokenizer_rules(pair: str, eos: str) -> None:
    """New pairs resolve to valid pilot configs; Llama gets <|eot_id|> and a pinned template date."""
    from chunkratio.pilot import PAIRS, pilot_assumptions
    protocol = json.loads(Path(f'config/crpd/narval/{pair}.json').read_text(encoding='utf-8'))
    checkpoints = {role:{'path':f'output/pilot/{role}','revision':'fp'+role} for role in ('teacher_plus','teacher_base','student_ref')}
    revisions = {k:'sha' for k in ('reward_model','teacher_base_model','student_base_model','sft_dataset','preference_dataset')}
    for variant in ('crpd_main','crpd_raw','crpd_anchor'):
        cfg = pilot_config(protocol,variant,checkpoints,revisions,'splitsha',True,pair)
        validate(cfg)
        assert cfg['eos_token'] == eos and cfg['output'].startswith(f'output/narval/{pair}/pilot/')
        assert cfg['pilot_assumptions'] == pilot_assumptions(pair) != PILOT_ASSUMPTIONS
        assert cfg['narval']['base_checkpoints']['teacher'] == PAIRS[pair]['teacher']+'@sha'
    sft = preparation_config('train-sft','m','r','d','o',pair,PAIRS[pair]['tokenizer'],'tsha')
    assert (sft['eos_token'],sft['tokenizer'],sft['tokenizer_revision']) == (eos,PAIRS[pair]['tokenizer'],'tsha')
    assert 'tokenizer' not in preparation_config('train-teacher-dpo','m','r','d','o',pair)
    if pair.startswith('llama'):
        bad = pilot_config(protocol,'crpd_main',checkpoints,revisions,'splitsha',True,pair)
        bad['chat_template_kwargs'] = {}
        with pytest.raises(ValueError,match='date_string'):
            validate(bad)
        with pytest.raises(ValueError,match='eot_id'):
            validate({**bad,'eos_token':'<|end_of_text|>'})


def test_hub_pair_configs_record_published_teachers() -> None:
    """Hub teachers load from pinned commits; assumptions name them and the contamination risk."""
    from chunkratio.pilot import HUB_PAIR, HUB_TEACHERS, pilot_assumptions
    protocol = json.loads(Path('config/crpd/narval/qwen3_1p7b_to_0p6b.json').read_text(encoding='utf-8'))
    checkpoints = {role:{'path':HUB_TEACHERS[role]['repo'],'revision':HUB_TEACHERS[role]['revision']}
                   for role in ('teacher_base','teacher_plus')}
    checkpoints['student_ref'] = {'path':'output/pilot/student_ref','revision':'fp'}
    revisions = {k:'sha' for k in ('reward_model','teacher_base_model','student_base_model','sft_dataset','preference_dataset')}
    cfg = pilot_config(protocol,'crpd_main',checkpoints,revisions,'splitsha',True,HUB_PAIR)
    validate(cfg)
    assert cfg['output'].startswith('output/narval/qwen3_1p7b_to_0p6b_hub/pilot/') and cfg['eos_token'] == '<|im_end|>'
    assumptions = pilot_assumptions(HUB_PAIR)
    assert cfg['pilot_assumptions'] == assumptions and 'contamination' in assumptions and 'LoRA' in assumptions['teacher_pair']
    assert all(len(source['revision']) == 40 for source in HUB_TEACHERS.values())


def test_default_pair_is_unchanged() -> None:
    """Adding pairs leaves the Qwen3-1.7B -> 0.6B helpers exactly as before."""
    from chunkratio.pilot import PAIR, pilot_assumptions
    assert pilot_assumptions() is PILOT_ASSUMPTIONS and pilot_assumptions(PAIR) is PILOT_ASSUMPTIONS
    assert 'tokenizer' not in preparation_config('train-sft','m','r','d','o')


def test_hda_trio_configs_load_every_role_from_the_hub() -> None:
    """The NARVAL-matching trio: all roles from pinned commits, <|im_end|> EOS, reference check named."""
    from chunkratio.pilot import HDA_CHECKPOINTS, HDA_PAIR, pilot_assumptions
    protocol = json.loads(Path('config/crpd/narval/qwen3_1p7b_to_0p6b.json').read_text(encoding='utf-8'))
    checkpoints = {role:{'path':source['repo'],'revision':source['revision']} for role,source in HDA_CHECKPOINTS.items()}
    revisions = {k:'sha' for k in ('reward_model','teacher_base_model','student_base_model','sft_dataset','preference_dataset')}
    cfg = pilot_config(protocol,'crpd_raw',checkpoints,revisions,'splitsha',True,HDA_PAIR)
    validate(cfg)
    assert cfg['output'].startswith('output/narval/qwen3_1p7b_to_0p6b_hda/pilot/') and cfg['eos_token'] == '<|im_end|>'
    assert {role:entry['path'] for role,entry in cfg['models'].items()} == {r:s['repo'] for r,s in HDA_CHECKPOINTS.items()}
    assumptions = pilot_assumptions(HDA_PAIR)
    assert cfg['pilot_assumptions'] == assumptions and 'nearest candidate' in assumptions['teacher_pair']
    assert all(len(source['revision']) == 40 for source in HDA_CHECKPOINTS.values())
