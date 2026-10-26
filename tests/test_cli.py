"""CLI remains inspectable without optional training dependencies (§10)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chunkratio.config import VARIANTS, load_config, validate
from chunkratio.data import helpsteer_pairs


def test_config_variants_and_strict_protocol() -> None:
    """Every variant is selectable, unresolved real model identities fail closed (§9)."""
    cfg = load_config('config/crpd/smoke.json')
    for variant in VARIANTS:
        validate({**cfg,'variant':variant})
    with pytest.raises(ValueError,match='path and revision'):
        load_config('config/crpd/protocol.json')
    with pytest.raises(ValueError,match='raw categorical'):
        validate({**cfg,'collection_decoding':{'temperature':.7}})


def test_module_and_additive_training_cli() -> None:
    """CRPD opt-in dispatch works without importing legacy Unix-only modules (§3)."""
    for command in ([sys.executable,'-m','chunkratio','--help'],[sys.executable,'train.py','--crpd','--help']):
        result = subprocess.run(command,capture_output=True,text=True)
        assert result.returncode == 0 and 'collect-correction' in result.stdout


def test_helpsteer_ties_removed() -> None:
    """Helpfulness pairs prepared once; main student then consumes prompts only (§11)."""
    rows = [{'prompt':'p','response':str(i),'helpfulness':score} for i,score in enumerate([2,2,4])]
    pairs = helpsteer_pairs(rows)
    assert len(pairs) == 2 and all(p['chosen'] == '2' for p in pairs)


class TemplateTokenizer:
    """Chat template ending assistant turns with token 99, then a trailing token 7 (§5.1)."""

    def __init__(self, eos: int) -> None:
        """Declared tokenizer EOS, possibly not the template's end-of-turn token."""
        self.eos_token_id = eos

    def apply_chat_template(self, messages: list, tokenize: bool, add_generation_prompt: bool) -> list:
        """User turn 10+len, assistant answer as character codes, end-of-turn 99."""
        ids = [10+len(messages[0]['content'])]
        if len(messages) > 1:
            ids += [ord(c) for c in messages[1]['content']]+[99,7]
        return ids


@pytest.mark.parametrize('eos,ok',[(99,True),(98,False)])
def test_prepare_data_requires_template_eos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eos: int, ok: bool) -> None:
    """Responses end at tokenizer EOS (counted once, trailer dropped) or preparation fails closed (§5.1)."""
    import json
    import sys
    from types import SimpleNamespace
    from chunkratio.artifacts import read_jsonl
    from chunkratio.data import prepare_data
    fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:TemplateTokenizer(eos)))
    monkeypatch.setitem(sys.modules,'transformers',fake)
    source = tmp_path/'export.jsonl'
    source.write_text(json.dumps({'messages':[{'role':'user','content':'hi'}],'chosen':'ab','rejected':'d',
                                  'split':'train'})+'\n',encoding='utf-8')
    if not ok:
        with pytest.raises(ValueError,match='end-of-turn'):
            prepare_data(source,'tok','rev',tmp_path/'out.jsonl',0)
        return
    prepare_data(source,'tok','rev',tmp_path/'out.jsonl',0)
    row, = read_jsonl(tmp_path/'out.jsonl')
    assert row['prompt_ids'] == [12] and row['chosen_ids'] == [97,98,99] and row['rejected_ids'] == [100,99]
