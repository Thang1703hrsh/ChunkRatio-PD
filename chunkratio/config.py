"""Explicit configuration and protocol validation (§§5,9,11). No tensor imports."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

VARIANTS = ('crpd_main','crpd_raw','crpd_mu_teacher','crpd_token','crpd_suffix','crpd_anchor','crpd_audit')
STAGES = ('validate-config','collect-prefixes','collect-candidates','collect-correction','fit-head','build-targets','train-policy','evaluate')


def collection_seed(cfg: dict[str,Any]) -> int:
    """Seed for splits, synthetic models, collection, endpoint selection and the head (§10).

    Training seeds share it; they vary only policy sampling and minibatch order.
    """
    return cfg.get('collection_seed',0)


def load_config(path: str | Path) -> dict[str,Any]:
    """Read explicit JSON/YAML config without silently inheriting TBPO values (§8)."""
    p = Path(path)
    if p.suffix.lower() == '.json':
        value = json.loads(p.read_text(encoding='utf-8-sig'))
    else:
        import yaml
        value = yaml.safe_load(p.read_text(encoding='utf-8-sig'))
    validate(value)
    return value


def validate(cfg: dict[str,Any]) -> None:
    """Fail closed for algorithmic settings, splits, decoding or protocol drift (§§2,11)."""
    required = ('variant','horizon','chunk_length','candidates','alpha','learning_rate',
                'optimizer','batch_size','steps','seed','head_epochs','head_learning_rate',
                'endpoint_fraction','correction_replicates','output','models','prompts','device')
    for key in required:
        if key not in cfg:
            raise ValueError(f'missing explicit setting: {key}')
    if cfg['variant'] not in VARIANTS:
        raise ValueError('unknown CRPD variant')
    for key in ('horizon','chunk_length','batch_size','steps','head_epochs','correction_replicates'):
        if not isinstance(cfg[key],int) or cfg[key] <= 0:
            raise ValueError(f'{key} must be positive integer')
    for key in ('alpha','learning_rate','head_learning_rate'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not isinstance(cfg.get('head_batch_size',1),int) or cfg.get('head_batch_size',1) <= 0:
        raise ValueError('head_batch_size must be positive integer')
    for key in ('states_per_prompt','generation_batch_size','collection_seed'):
        value = cfg.get(key)
        if value is not None and (not isinstance(value,int) or value < (0 if key == 'collection_seed' else 1)):
            raise ValueError(f'{key} must be a positive integer or null')
    if cfg.get('cache_root') is not None and (not isinstance(cfg['cache_root'],str) or Path(cfg['cache_root']).is_absolute()):
        raise ValueError('cache_root must be a relative path')
    if cfg['candidates'] != 4 or not 0 <= cfg['endpoint_fraction'] <= 1:
        raise ValueError('initial protocol requires K=4 and valid endpoint fraction')
    if cfg.get('weight_decay',0) != 0 or cfg['optimizer'] not in ('AdamW','Adam','RMSprop'):
        raise ValueError('unsupported optimizer or nonzero primary weight decay')
    if cfg.get('collection_decoding',{}) != {'temperature':1,'top_k':0,'top_p':1}:
        raise ValueError('collection must use raw categorical decoding')
    if set(cfg['models']) != {'teacher_plus','teacher_base','student_ref'}:
        raise ValueError('three explicit checkpoint roles required')
    if cfg['models']['student_ref'].get('device',cfg['device']) != cfg['device']:
        raise ValueError('student_ref, student and head share the top-level device')
    if cfg['variant'] == 'crpd_anchor' and (cfg.get('anchor_lambda',0) <= 0 or cfg.get('anchor_beta',0) <= 0):
        raise ValueError('anchor requires fixed lambda and beta')
    if cfg['variant'] == 'crpd_audit' and cfg.get('audit_replicates',0) < 2:
        raise ValueError('exponential audit needs replicated independent estimates')
    if not isinstance(cfg.get('selection_interval',1),int) or cfg.get('selection_interval',1) <= 0:
        raise ValueError('selection_interval must be positive integer')
    if cfg.get('lr_schedule','constant') not in ('constant','cosine'):
        raise ValueError('lr_schedule must be constant or cosine')
    if not 0 <= cfg.get('warmup_ratio',0) < 1:
        raise ValueError('warmup_ratio must be in [0,1)')
    if cfg.get('policy_autocast_dtype') not in (None,'bfloat16'):
        raise ValueError('policy_autocast_dtype must be null or bfloat16 (student forwards only)')
    if cfg.get('protocol_name') == 'narval':
        validate_narval(cfg)
    if not cfg.get('synthetic',False):
        if 'selection_interval' not in cfg:
            raise ValueError('real runs need an explicit selection_interval (RM selection per checkpoint)')
        for role,model in cfg['models'].items():
            if not model.get('path') or not model.get('revision'):
                raise ValueError(f'exact path and revision required for {role}')
        if not cfg.get('teacher_dpo_reference_verified',False):
            raise ValueError('actual teacher DPO reference must be verified')
        if not cfg.get('selection',{}).get('reward_model_revision'):
            raise ValueError('pin selection reward model revision')


NARVAL_VARIANTS = ('crpd_main','crpd_raw','crpd_anchor')


def open_inputs(value: Any, prefix: str = '') -> list[str]:
    """Dotted paths of every null leaf: the inputs a user must still fill (NARVAL protocol)."""
    if value is None:
        return [prefix]
    if isinstance(value,dict):
        return [p for k,v in value.items() for p in open_inputs(v,f'{prefix}.{k}' if prefix else k)]
    return []


def linked_protocol(path: str) -> dict[str,Any]:
    """Read a linked protocol relative to the repository root (no absolute paths)."""
    return json.loads((Path(__file__).resolve().parents[1]/path).read_text(encoding='utf-8-sig'))


def validate_narval(cfg: dict[str,Any]) -> None:
    """NARVAL-comparable protocol: structural rules always; every null fails closed for real runs."""
    narval = cfg.get('narval')
    if not isinstance(narval,dict):
        raise ValueError('narval protocol block required')
    if cfg['variant'] not in NARVAL_VARIANTS or cfg['variant'] not in narval.get('variants',[]):
        raise ValueError('NARVAL protocol runs only crpd_main, crpd_raw and crpd_anchor')
    if cfg['correction_replicates'] != 1:
        raise ValueError('NARVAL protocol pre-registers correction_replicates = 1')
    if str(narval.get('pair','')).startswith('qwen3'):
        if cfg.get('eos_token') != '<|im_end|>':
            raise ValueError('Qwen3 pairs use <|im_end|> as the EOS of every role')
        if cfg.get('chat_template_kwargs') != {'enable_thinking':False}:
            raise ValueError('Qwen3 chat template must be fixed with enable_thinking=false')
    if cfg.get('synthetic',False):
        return
    if cfg.get('pilot'):
        # Pilot runs need every training input; evaluation fields and reference rows may stay open.
        if not cfg.get('pilot_assumptions'):
            raise ValueError('pilot configs must list their assumed (unverified) settings')
        scope = {k:v for k,v in narval.items() if k not in ('evaluation','reference_rows')}
    else:
        scope = narval
    missing = open_inputs({'eos_token':cfg.get('eos_token'),'models':cfg['models'],
                           'selection':cfg.get('selection',{}),'narval':scope})
    if missing:
        raise ValueError('NARVAL open inputs (fill from the NARVAL repository or verified checkpoints): '+', '.join(missing))
    for role,link in (narval.get('provenance') or {}).items():
        if role == 'note':
            continue
        row = linked_protocol(link['protocol'])['narval']['reference_rows'][link['row']]
        mine = cfg['models'][role]
        if not row.get('path') or (row.get('path'),row.get('revision')) != (mine['path'],mine['revision']):
            raise ValueError(f"provenance mismatch: {role} must be {link['row']} of {link['protocol']}")
