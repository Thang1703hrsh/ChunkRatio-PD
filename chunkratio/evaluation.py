"""Measured CRPD diagnostics, reward selection and research tables (§§8,11)."""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any

METRICS = ('rm','mt_bench','ae_vs_adpa','ae2_lc','ae2_wr','arena_hard',
           'arc','hellaswag','truthfulqa','mmlu','winogrande','gsm8k')
REFERENCE_ROWS = ('teacher_dpo','teacher_sft','student_sft','student_dpo')
BASELINES = ('simpo','dpkd','dckd','adpa','tvkd','tbpo_q','tbpo_a')


def paired_interval(left: dict[str,float], right: dict[str,float], seed: int = 0,
                    replicates: int = 2000) -> dict[str,float]:
    """Paired bootstrap resamples prompts, not correlated derived prefixes (§11.4)."""
    if set(left) != set(right) or not left:
        raise ValueError('paired comparisons require identical nonempty prompt sets')
    differences = [left[k]-right[k] for k in sorted(left)]
    if not all(math.isfinite(x) for x in differences):
        raise ValueError('nonfinite evaluation outcomes')
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(replicates))
    return {'mean_difference':statistics.mean(differences),'low':samples[int(.025*replicates)],
            'high':samples[min(replicates-1,int(.975*replicates))],'n_prompts':len(differences)}


def report_tables(rows: list[dict[str,Any]], allow_incomplete: bool = False) -> str:
    """Render measured common-protocol rows; reject synthetic/selection leakage (§11.5)."""
    if not rows:
        raise ValueError('no measured benchmark rows provided')
    normalized = []
    protocols: dict[tuple[str,str],str] = {}
    for row in rows:
        if row.get('synthetic',True):
            raise ValueError('synthetic diagnostics cannot populate research tables')
        if row.get('split') != 'test':
            raise ValueError('main table requires held-out test outcomes')
        if row['dataset'] not in ('DPO-MIX-7K','HelpSteer2'):
            raise ValueError('unexpected preference track')
        for metric,value in row['metrics'].items():
            if metric not in METRICS or not math.isfinite(value):
                raise ValueError('unknown/nonfinite measured metric')
            identity = row['protocols'][metric]
            key = (row['dataset'],metric)
            if key in protocols and protocols[key] != identity:
                raise ValueError('incomparable judge/version/decoding protocols')
            protocols[key] = identity
        if not allow_incomplete and set(row['metrics']) != set(METRICS):
            raise ValueError('all specified metrics required; use --allow-incomplete for a labeled partial report')
        normalized.append(row)
    methods = sorted({r['method'] for r in normalized})
    lines = ['# Measured CRPD results','',
             'Training seeds remain separate from prompt-bootstrap uncertainty. Missing values are not estimates.','',
             '| Method | DPO-MIX-7K: RM / MT / AE-vs-ADPA / AE2-LC / OLL | HelpSteer2: RM / MT / AE-vs-ADPA / AE2-LC / OLL | GPU-h |',
             '|---|---|---|---|']
    secondary = ['','| Method / track | AE2 raw | Arena-Hard | ARC | HellaSwag | TruthfulQA | MMLU | Winogrande | GSM8K |',
                 '|---|---|---|---|---|---|---|---|---|']
    for method in methods:
        cells, costs = [],[]
        for dataset in ('DPO-MIX-7K','HelpSteer2'):
            subset = [r for r in normalized if r['method'] == method and r['dataset'] == dataset]
            if not subset:
                cells.append('missing')
                continue
            seeds = {r['seed'] for r in subset}
            if len(seeds) != len(subset):
                raise ValueError('duplicate method/track/seed result')
            if not allow_incomplete and method in ('crpd_main','crpd_raw') and len(seeds) < 3:
                raise ValueError('at least three training seeds required')
            means = {m:statistics.mean(r['metrics'][m] for r in subset if m in r['metrics'])
                     for m in METRICS if any(m in r['metrics'] for r in subset)}
            tasks = ('arc','hellaswag','truthfulqa','mmlu','winogrande','gsm8k')
            if all(m in means for m in tasks):
                means['oll'] = statistics.mean(means[m] for m in tasks)
            cells.append(' / '.join(f'{means[m]:.4f}' if m in means else 'missing' for m in ('rm','mt_bench','ae_vs_adpa','ae2_lc','oll')))
            costs.extend(r['cost']['gpu_hours'] for r in subset)
            secondary.append('| '+method+' / '+dataset+' | '+' | '.join(f'{means[m]:.4f}' if m in means else 'missing' for m in ('ae2_wr','arena_hard',*tasks))+' |')
        lines.append('| '+method+' | '+' | '.join(cells)+' | '+(f'{statistics.mean(costs):.3f}' if costs else 'missing')+' |')
    return '\n'.join(lines+secondary)+'\n'


def reward_selector(cfg: dict[str,Any], prompts: list[dict[str,Any]], tokenizer: Any):
    """Pinned FsfairX RM, select prompts only, same generation rule for all methods (§11.1)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    selection = cfg['selection']
    rows = [p for p in prompts if p['split'] == 'select']
    if not rows:
        raise ValueError('select split required for reward checkpoint selection')
    name = 'sfairXC/FsfairX-LLaMA3-RM-v0.1'
    rm = AutoModelForSequenceClassification.from_pretrained(name,revision=selection['reward_model_revision'],torch_dtype=torch.float32)
    rm = rm.to(cfg['device']).eval().requires_grad_(False)
    rm_tok = AutoTokenizer.from_pretrained(name,revision=selection['reward_model_revision'])
    decode = selection['decoding']

    @torch.no_grad()
    def select(student):
        """Mean independent reward of generated responses on fixed select prompts (§11.1)."""
        scores = []
        for row in rows:
            ids = torch.tensor([row['prompt_ids']],device=cfg['device'])
            # Reset stream per prompt/checkpoint so selection noise is controlled.
            with torch.random.fork_rng(devices=[ids.device] if ids.is_cuda else []):
                torch.manual_seed(cfg['seed']+911)
                response = student.generate(ids,attention_mask=torch.ones_like(ids),
                                            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,**decode)
            if 'messages' not in row:
                raise ValueError('reward selection requires original prompt messages')
            messages = row['messages']+[{'role':'assistant','content':tokenizer.decode(response[0,ids.shape[1]:],skip_special_tokens=True)}]
            text = rm_tok.apply_chat_template(messages,tokenize=False)
            inputs = rm_tok(text,return_tensors='pt',add_special_tokens=False).to(cfg['device'])
            logits = rm(**inputs).logits
            if logits.numel() != 1:
                raise ValueError('reward model must return one scalar')
            scores.append(float(logits.item()))
        return statistics.mean(scores)
    return select
