"""Explicit configuration and protocol validation (§§5,9,11). No tensor imports."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

VARIANTS = ('crpd_main','crpd_raw','crpd_mu_teacher','crpd_token','crpd_suffix','crpd_anchor','crpd_audit')
STAGES = ('validate-config','collect-prefixes','collect-candidates','collect-correction','fit-head','build-targets','train-policy','evaluate')


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
