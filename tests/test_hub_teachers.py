"""Published teacher pairs: T+ must be T0 plus its published DPO LoRA (weights, not names)."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from safetensors.torch import save_file

from chunkratio.hub_teachers import lora_deltas, verify_lora_merge


def write_pair(tmp_path: Path, tamper: str | None = None) -> tuple[Path,Path,Path]:
    """Tiny bf16 T0, LoRA (r=2, alpha=4) and T+ = T0 + (alpha/r) B A merged in bf16."""
    gen = torch.Generator().manual_seed(0)
    t0 = {'model.layers.0.self_attn.q_proj.weight':torch.randn(6,5,generator=gen).bfloat16(),
          'model.layers.0.input_layernorm.weight':torch.ones(5).bfloat16()}
    a,b = torch.randn(2,5,generator=gen)*.1,torch.randn(6,2,generator=gen)*.1
    tplus = dict(t0)
    tplus['model.layers.0.self_attn.q_proj.weight'] = (t0['model.layers.0.self_attn.q_proj.weight'].float()+2*(b@a)).bfloat16()
    if tamper == 'untouched':
        tplus['model.layers.0.input_layernorm.weight'] = torch.full((5,),1.5).bfloat16()
    if tamper == 'unrelated':
        tplus['model.layers.0.self_attn.q_proj.weight'] = (t0['model.layers.0.self_attn.q_proj.weight'].float()
                                                           +torch.randn(6,5,generator=gen)*.1).bfloat16()
    folders = [tmp_path/name for name in ('t0','tplus','lora')]
    for folder in folders:
        folder.mkdir()
    save_file(t0,str(folders[0]/'model.safetensors'))
    save_file(tplus,str(folders[1]/'model.safetensors'))
    save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight':a,
               'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight':b},
              str(folders[2]/'adapter_model.safetensors'))
    (folders[2]/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':2,'lora_alpha':4,
                                                              'use_dora':False,'use_rslora':False,
                                                              'rank_pattern':{},'alpha_pattern':{}}),encoding='utf-8')
    return folders[0],folders[1],folders[2]


def test_published_merge_verifies(tmp_path: Path) -> None:
    """bf16 merge rounding passes; the untouched norm must be bit-identical."""
    report = verify_lora_merge(*write_pair(tmp_path))
    assert report['verified'] and report['targeted_tensors'] == 1 and report['untouched_mismatches'] == []
    assert report['max_ulp_distance'] <= 1 and report['fraction_exact_after_rounding'] == 1.


def test_bf16_merge_with_cancellation_verifies(tmp_path: Path) -> None:
    """A merge done in bf16 arithmetic (W + bf16(d)) passes even where W and d nearly cancel."""
    t0,tplus,lora = write_pair(tmp_path)
    from safetensors.torch import load_file
    base = load_file(str(t0/'model.safetensors'))
    key = 'model.layers.0.self_attn.q_proj.weight'
    delta = lora_deltas(lora)[key]
    base[key] = (-delta).bfloat16()+torch.tensor(1e-3).bfloat16()*(torch.arange(30).reshape(6,5)%2)
    save_file(base,str(t0/'model.safetensors'))
    merged = load_file(str(tplus/'model.safetensors'))
    merged[key] = base[key]+delta.bfloat16()
    save_file(merged,str(tplus/'model.safetensors'))
    report = verify_lora_merge(t0,tplus,lora)
    assert report['verified'] and report['max_ulp_distance'] <= 2


@pytest.mark.parametrize('tamper',['untouched','unrelated'])
def test_tampered_or_unrelated_teacher_fails(tmp_path: Path, tamper: str) -> None:
    """A changed untouched tensor or an update the LoRA does not explain is not verified."""
    report = verify_lora_merge(*write_pair(tmp_path,tamper))
    assert not report['verified']


def test_unsupported_adapter_rejected(tmp_path: Path) -> None:
    """DoRA/rsLoRA scale differently; refuse instead of mis-verifying."""
    _,_,lora = write_pair(tmp_path)
    config = json.loads((lora/'adapter_config.json').read_text(encoding='utf-8'))
    (lora/'adapter_config.json').write_text(json.dumps({**config,'use_dora':True}),encoding='utf-8')
    with pytest.raises(ValueError,match='plain LoRA'):
        lora_deltas(lora)
