"""Explicit full-finetuning teacher/student SFT and teacher-DPO recipes (§11.1).

Research-scale launcher code, never invoked by smoke tests or this implementation task.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .artifacts import digest, read_jsonl, write_json
from .sampling import stream
from .scoring import fingerprint, freeze, score_batch


def prepare_checkpoint(cfg: dict[str,Any], stage: str) -> None:
    """SFT 3 epochs/2e-5/128; DPO 2 epochs/5e-7/128/beta .05 (§11.1)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dpo = stage == 'train-teacher-dpo'
    expected = {'epochs':2 if dpo else 3,'learning_rate':5e-7 if dpo else 2e-5,'batch_size':128}
    for key,value in expected.items():
        if cfg[key] != value:
            raise ValueError(f'primary {stage} requires {key}={value}')
    if dpo and cfg['beta'] != .05:
        raise ValueError('primary teacher DPO beta must be .05')
    if not cfg.get('revision'):
        raise ValueError('pin base revision')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],torch_dtype=torch.float32).to(cfg['device'])
    ref = freeze(copy.deepcopy(model)) if dpo else None
    reference_identity = fingerprint(ref) if ref is not None else None
    rows = [r for r in read_jsonl(Path(cfg['data'])) if r['split'] == 'train']
    if not rows:
        raise ValueError('no train data')
    context = model.config.max_position_embeddings
    # Exclude over-budget examples before sampling, never silently truncate responses.
    rows = [r for r in rows if len(r['prompt_ids'])+max(len(r['chosen_ids']),len(r.get('rejected_ids',[]))) <= context]
    if not rows or any(not row['chosen_ids'] or (dpo and not row.get('rejected_ids')) for row in rows):
        raise ValueError('no eligible nonempty supervised responses after context preflight')
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    rng = stream(cfg['seed'],'preparation-shuffle')
    torch.manual_seed(cfg['seed'])
    history = []
    for epoch in range(cfg['epochs']):
        permutation = torch.randperm(len(rows),generator=rng).tolist()
        for start in range(0,len(rows),cfg['batch_size']):
            batch = [rows[i] for i in permutation[start:start+cfg['batch_size']]]
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for row in batch:
                prefix = [row['prompt_ids']]
                chosen = [row['chosen_ids']]
                chosen_score = score_batch(model,prefix,chosen,tokenizer.eos_token_id,False)
                if dpo:
                    rejected = [row['rejected_ids']]
                    margin = chosen_score-score_batch(model,prefix,rejected,tokenizer.eos_token_id,False)
                    reference = score_batch(ref,prefix,chosen,tokenizer.eos_token_id)-score_batch(ref,prefix,rejected,tokenizer.eos_token_id)
                    loss = -F.logsigmoid(cfg['beta']*(margin-reference)).mean()/len(batch)
                else:
                    loss = -chosen_score.mean()/len(row['chosen_ids'])/len(batch)
                assert torch.isfinite(loss)
                loss.backward()
                total += loss.item()
            optimizer.step()
            history.append({'epoch':epoch+1,'batch':start//cfg['batch_size'],'loss':total})
    output = Path(cfg['output'])
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_json(output/'preparation_manifest.json',{'stage':stage,'config':cfg,'data_hash':digest(rows),
                'teacher_dpo_reference_fingerprint':reference_identity,'model_fingerprint':fingerprint(model),
                'history':history,'note':'single-process full fine tuning; microbatch=1, effective batch=128 except final partial batch'})
