"""Synthetic or revision-pinned checkpoint loading with canonical token preflight (§3)."""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .config import collection_seed
from .scoring import freeze, preflight
from .tiny import TinyLM, TinyTokenizer


def load_models(cfg: dict[str,Any]) -> tuple[dict[str,nn.Module],nn.Module,list[Any],dict[str,Any]]:
    """Four model roles with identical tokenization and frozen initialization (§2)."""
    if cfg.get('synthetic',False):
        torch.manual_seed(collection_seed(cfg)+101)
        base = TinyLM()
        plus = copy.deepcopy(base)
        gen = torch.Generator().manual_seed(collection_seed(cfg)+103)
        with torch.no_grad():
            for p in plus.parameters():
                p.add_(torch.randn(p.shape,generator=gen)*.07)
        torch.manual_seed(collection_seed(cfg)+107)
        rho = TinyLM()
        tokenizers = [TinyTokenizer() for _ in range(4)]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        loaded,toks = [],[]
        for role in ('teacher_plus','teacher_base','student_ref'):
            entry = cfg['models'][role]
            dtype = entry.get('dtype',cfg.get('dtype','float32'))
            loaded.append(AutoModelForCausalLM.from_pretrained(entry['path'],revision=entry['revision'],
                                                             torch_dtype=getattr(torch,dtype)))
            tok = AutoTokenizer.from_pretrained(entry.get('tokenizer',entry['path']),
                                                revision=entry.get('tokenizer_revision',entry['revision']))
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            toks.append(tok)
        plus,base,rho = loaded
        tokenizers = toks+[copy.deepcopy(toks[2])]
    # Teachers may sit on another device; rho, the student and the head share cfg['device'].
    def device(role: str) -> str:
        """Per-role placement, defaulting to the shared device (§10)."""
        return cfg['models'][role].get('device',cfg['device'])
    models = {'teacher_plus':freeze(plus.to(device('teacher_plus'))),'teacher_base':freeze(base.to(device('teacher_base'))),
              'student_ref':freeze(rho.to(cfg['device']))}
    student = copy.deepcopy(rho).requires_grad_(True).eval()
    info = preflight(tokenizers,[plus,base,rho,student])
    return models,student,tokenizers,info
