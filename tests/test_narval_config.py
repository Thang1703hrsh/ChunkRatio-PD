"""NARVAL-comparable protocol configs fail closed and enforce their structural rules (no torch)."""
import copy

import pytest

import chunkratio.config as config
from chunkratio.config import load_config, open_inputs, validate

PAIRS = ('qwen3_8b_to_1p7b','qwen3_1p7b_to_0p6b','llama31_8b_to_llama32_1b')


def raw(pair: str) -> dict:
    """Read a protocol file without validation."""
    import json
    from pathlib import Path
    return json.loads(Path(f'config/crpd/narval/{pair}.json').read_text(encoding='utf-8'))


def filled(value: object) -> object:
    """Replace every null with a placeholder, as a user would after resolving open inputs."""
    if value is None:
        return 'x'
    if isinstance(value,dict):
        return {k:filled(v) for k,v in value.items()}
    return value


@pytest.mark.parametrize('pair',PAIRS)
def test_unverified_values_fail_closed(pair: str) -> None:
    """Every null (hyperparameters, revisions, judges, few-shot counts) blocks validate-config."""
    with pytest.raises(ValueError,match='NARVAL open inputs') as error:
        load_config(f'config/crpd/narval/{pair}.json')
    for name in ('narval.teacher_dpo.beta','narval.teacher_dpo.max_length','narval.sft.epochs',
                 'narval.evaluation.open_llm.lm_eval_version','narval.evaluation.open_llm.tasks.gsm8k.num_fewshot',
                 'narval.evaluation.mt_bench.judge','narval.evaluation.hh_rlhf.reference_outputs'):
        assert name in str(error.value)
    assert ('eos_token' in open_inputs({'eos_token':raw(pair)['eos_token']})) == pair.startswith('llama')


def test_filled_protocol_validates_and_structural_rules() -> None:
    """Resolved configs pass; variant, replicates, EOS and thinking rules hold regardless."""
    cfg = filled(raw('qwen3_8b_to_1p7b'))
    cfg['teacher_dpo_reference_verified'] = True
    validate(cfg)
    for variant in ('crpd_raw','crpd_anchor'):
        validate({**cfg,'variant':variant})
    for change,message in (({'variant':'crpd_token'},'only crpd_main'),({'correction_replicates':4},'replicates = 1'),
                           ({'eos_token':'<|endoftext|>'},'im_end'),
                           ({'chat_template_kwargs':{'enable_thinking':True}},'enable_thinking')):
        with pytest.raises(ValueError,match=message):
            validate({**cfg,**change})


def test_provenance_link_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen3-1.7B -> 0.6B: T0/T+ must be the SFT/DPO 1.7B student rows of the 8B -> 1.7B protocol."""
    cfg = filled(raw('qwen3_1p7b_to_0p6b'))
    cfg['teacher_dpo_reference_verified'] = True
    linked = filled(raw('qwen3_8b_to_1p7b'))
    monkeypatch.setattr(config,'linked_protocol',lambda path: linked)
    validate(cfg)
    other = copy.deepcopy(linked)
    other['narval']['reference_rows']['student_dpo']['revision'] = 'other'
    monkeypatch.setattr(config,'linked_protocol',lambda path: other)
    with pytest.raises(ValueError,match='provenance mismatch: teacher_plus'):
        validate(cfg)
    assert cfg['narval']['provenance']['teacher_base']['row'] == 'student_sft'
