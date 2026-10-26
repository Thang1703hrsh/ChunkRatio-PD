"""Synthetic or revision-pinned checkpoint loading with canonical token preflight (§3)."""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .scoring import freeze, preflight
from .tiny import TinyLM, TinyTokenizer


def load_models(cfg: dict[str,Any]) -> tuple[dict[str,nn.Module],nn.Module,list[Any],dict[str,Any]]:
    """Four model roles with identical tokenization and frozen initialization (§2)."""
    if cfg.get('synthetic',False):
        torch.manual_seed(cfg['seed']+101)
        base = TinyLM()
        plus = copy.deepcopy(base)
        gen = torch.Generator().manual_seed(cfg['seed']+103)
        with torch.no_grad():
            for p in plus.parameters():
                p.add_(torch.randn(p.shape,generator=gen)*.07)
        torch.manual_seed(cfg['seed']+107)
        rho = TinyLM()
        tokenizers = [TinyTokenizer() for _ in range(4)]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        loaded,toks = [],[]
        for role in ('teacher_plus','teacher_base','student_ref'):
            entry = cfg['models'][role]
            loaded.append(AutoModelForCausalLM.from_pretrained(entry['path'],revision=entry['revision'],
                                                             torch_dtype=getattr(torch,cfg.get('dtype','float32'))))
            tok = AutoTokenizer.from_pretrained(entry.get('tokenizer',entry['path']),
                                                revision=entry.get('tokenizer_revision',entry['revision']))
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            toks.append(tok)
        plus,base,rho = loaded
        tokenizers = toks+[copy.deepcopy(toks[2])]
    models = {'teacher_plus':freeze(plus.to(cfg['device'])),'teacher_base':freeze(base.to(cfg['device'])),
              'student_ref':freeze(rho.to(cfg['device']))}
    student = copy.deepcopy(rho).requires_grad_(True).eval()
    info = preflight(tokenizers,[plus,base,rho,student])
    return models,student,tokenizers,info
