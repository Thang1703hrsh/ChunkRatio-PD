"""Published teacher pairs: check that T+ is T0 plus a published DPO LoRA before using them.

A pilot may take T0 (an SFT checkpoint) and T+ (its DPO model) from the Hugging Face Hub instead
of training them, but CRPD needs T0 to be T+'s actual DPO reference. When the DPO LoRA is
published, rebuilding T0 + (alpha/r) B A and comparing it with T+ checks that relation from the
weights: tensors the LoRA does not touch must be bit-identical, and every merged element must
equal the rebuilt one up to the rounding of the stored dtype. Model names and cards are not evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# Rounding is relative to the weight, not to the (much smaller) DPO update, so the tolerance is in
# units in the last place of the stored dtype: 2 ulp allows a 16-bit merge plus one rounding. The
# ulp is that of the largest operand (T0, the LoRA update) or result: a 16-bit sum W + d is rounded
# relative to its operands, so where W and d nearly cancel the error is tiny in absolute terms but
# many ulp of the near-zero result (the published merge showed 256 such ulp at 99.85% exact matches).
MAX_ULP = 2.
MANTISSA_BITS = {torch.bfloat16:7,torch.float16:10,torch.float32:23}


def _tensors(folder: Path) -> dict[str,torch.Tensor]:
    """All safetensors shards of a folder, by key."""
    from safetensors.torch import load_file
    tensors: dict[str,torch.Tensor] = {}
    for shard in sorted(folder.glob('*.safetensors')):
        tensors.update(load_file(str(shard)))
    if not tensors:
        raise FileNotFoundError(f'no safetensors in {folder}')
    return tensors


def lora_deltas(adapter: Path) -> dict[str,torch.Tensor]:
    """(alpha/r) B A per targeted weight key, in FP32, from a PEFT LoRA folder (no peft import)."""
    config = json.loads((adapter/'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('peft_type') != 'LORA' or config.get('use_dora') or config.get('use_rslora'):
        raise ValueError('only plain LoRA adapters are supported')
    if config.get('rank_pattern') or config.get('alpha_pattern'):
        raise ValueError('per-module rank or alpha patterns are not supported')
    scale = config['lora_alpha']/config['r']
    weights = _tensors(adapter)
    deltas = {}
    for key,a in weights.items():
        if not key.endswith('.lora_A.weight'):
            continue
        b = weights[key.replace('.lora_A.weight','.lora_B.weight')]
        module = key[:-len('.lora_A.weight')].removeprefix('base_model.model.')
        deltas[module+'.weight'] = scale*(b.float()@a.float())
    if not deltas:
        raise ValueError('adapter has no LoRA weights')
    return deltas


def verify_lora_merge(base: Path, merged: Path, adapter: Path) -> dict[str,Any]:
    """Compare T+ with T0 + LoRA; `verified` only if untouched tensors are identical, T+ differs
    from T0, and every targeted element is within MAX_ULP of the rebuilt weight."""
    t0,tplus,deltas = _tensors(base),_tensors(merged),lora_deltas(adapter)
    missing = sorted(set(deltas)-set(t0))
    if missing:
        raise ValueError(f'LoRA targets missing from T0: {missing[:3]}')
    if set(t0) != set(tplus):
        raise ValueError('T0 and T+ have different tensor names')
    untouched_mismatch = [k for k in t0 if k not in deltas and not torch.equal(t0[k],tplus[k])]
    worst_ulp = changed = exact = total = 0.
    worst_key = None
    for key,delta in deltas.items():
        target = tplus[key]
        bits = MANTISSA_BITS.get(target.dtype)
        if bits is None:
            raise ValueError(f'unsupported dtype {target.dtype} for {key}')
        rebuilt = t0[key].double()+delta.double()
        magnitude = torch.stack([t0[key].double().abs(),delta.double().abs(),rebuilt.abs(),target.double().abs()]).amax(0)
        magnitude = magnitude.clamp(min=torch.finfo(target.dtype).tiny)
        ulp = torch.exp2(torch.floor(torch.log2(magnitude))-bits)
        distance = float(((target.double()-rebuilt).abs()/ulp).max())
        if distance > worst_ulp:
            worst_ulp,worst_key = distance,key
        changed += float((target != t0[key]).sum())
        exact += float((rebuilt.to(target.dtype) == target).sum())
        total += target.numel()
    return {'targeted_tensors':len(deltas),'untouched_tensors':len(t0)-len(deltas),
            'untouched_mismatches':untouched_mismatch[:20],'max_ulp_distance':worst_ulp,'worst_tensor':worst_key,
            'fraction_changed_by_dpo':changed/total,'fraction_exact_after_rounding':exact/total,
            'dtype':str(next(iter(tplus.values())).dtype),'max_ulp_allowed':MAX_ULP,
            'verified':not untouched_mismatch and changed > 0 and worst_ulp <= MAX_ULP}


def nearest_reference(plus: Path, reference: Path, alternatives: dict[str,Path]) -> dict[str,Any]:
    """When no LoRA is published: relative L2 distance from T+ to its claimed reference and to other
    candidates (for example the Base checkpoint T0 was trained from).

    `nearest` holds only if the claimed reference is strictly closer to T+ than every alternative and
    T+ differs from it. One epoch of DPO moves weights far less than SFT does, so a DPO of T0 should
    pass; this is evidence from the weights, not a proof that T0 was the DPO reference model.
    """
    tplus = _tensors(plus)
    def distance(folder: Path) -> float:
        """||T+ - other|| / ||T+|| over the tensors both checkpoints store."""
        other = _tensors(folder)
        # A tied output embedding may be stored in one checkpoint only; any other difference is an error.
        if (set(other) ^ set(tplus))-{'lm_head.weight'}:
            raise ValueError(f'{folder} and T+ hold different tensors')
        keys = sorted(set(other) & set(tplus))
        diff = sum(float((tplus[k].double()-other[k].double()).square().sum()) for k in keys)
        norm = sum(float(tplus[k].double().square().sum()) for k in keys)
        return (diff/norm)**.5
    distances = {'reference':distance(reference),**{name:distance(folder) for name,folder in alternatives.items()}}
    nearest = distances['reference'] > 0 and all(distances['reference'] < d for k,d in distances.items() if k != 'reference')
    return {'relative_l2_distance':distances,'nearest':nearest,
            'rule':'claimed reference strictly nearest to T+ in relative L2 over shared tensors, and not identical'}
