"""Measured CRPD diagnostics, reward selection and research tables (§§8,11)."""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable
from collections import defaultdict
from typing import Any
from pathlib import Path

from .artifacts import atomic_bytes, read_jsonl, write_json

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
    required_methods = set(REFERENCE_ROWS+BASELINES+('crpd_main','crpd_raw'))
    missing_methods = sorted(required_methods-set(methods))
    if missing_methods and not allow_incomplete:
        raise ValueError('required reference/baseline rows missing: '+', '.join(missing_methods))
    lines = ['# '+('Partial measured CRPD results' if allow_incomplete else 'Measured CRPD results'),'',
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
    seed_variation = ['','| Method / track | Metric | Training seeds | Mean | Between-seed sample SD |',
                      '|---|---|---|---|---|']
    for method in methods:
        for dataset in ('DPO-MIX-7K','HelpSteer2'):
            subset = [r for r in normalized if r['method'] == method and r['dataset'] == dataset]
            for metric in METRICS:
                values = [r['metrics'][metric] for r in subset if metric in r['metrics']]
                if values:
                    std = f'{statistics.stdev(values):.4f}' if len(values)>1 else 'unavailable'
                    seed_variation.append(f'| {method} / {dataset} | {metric} | {len(values)} | {statistics.mean(values):.4f} | {std} |')
    if missing_methods:
        seed_variation += ['','Missing reference/baseline rows: '+', '.join(missing_methods)+'.']
    return '\n'.join(lines+secondary+seed_variation)+'\n'


def reward_selector(cfg: dict[str,Any], prompts: list[dict[str,Any]], tokenizer: Any) -> tuple[Callable[[Any],float],dict[str,int]]:
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
    cost = {'generated_tokens':0,'rm_scored_input_tokens':0,'selection_calls':0}

    @torch.no_grad()
    def select(student: Any) -> float:
        """Mean independent reward of generated responses on fixed select prompts (§11.1)."""
        scores = []
        cost['selection_calls'] += 1
        for row in rows:
            ids = torch.tensor([row['prompt_ids']],device=cfg['device'])
            # Reset stream per prompt/checkpoint so selection noise is controlled.
            with torch.random.fork_rng(devices=[ids.device] if ids.is_cuda else []):
                torch.manual_seed(cfg['seed']+911)
                response = student.generate(ids,attention_mask=torch.ones_like(ids),
                                            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,**decode)
            cost['generated_tokens'] += response.shape[1]-ids.shape[1]
            if 'messages' not in row:
                raise ValueError('reward selection requires original prompt messages')
            messages = row['messages']+[{'role':'assistant','content':tokenizer.decode(response[0,ids.shape[1]:],skip_special_tokens=True)}]
            text = rm_tok.apply_chat_template(messages,tokenize=False)
            inputs = rm_tok(text,return_tensors='pt',add_special_tokens=False).to(cfg['device'])
            cost['rm_scored_input_tokens'] += int(inputs['attention_mask'].sum())
            logits = rm(**inputs).logits
            if logits.numel() != 1:
                raise ValueError('reward model must return one scalar')
            scores.append(float(logits.item()))
        return statistics.mean(scores)
    return select,cost


def evaluate_reward_model(cfg: dict[str,Any], model_path: str, output: Path) -> dict[str,Any]:
    """Generate held-out answers and score pinned FsfairX RM; never use select prompts (§11.4)."""
    import json
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    decode = cfg.get('evaluation_decoding')
    if decode is None:
        raise ValueError('explicit evaluation_decoding required, separate from collection horizon')
    prompts = [row for row in read_jsonl(Path(cfg['prompts'])) if row['split'] == 'test']
    if not prompts:
        raise ValueError('held-out test prompts required for RM reporting')
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float32).to(cfg['device']).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    name = 'sfairXC/FsfairX-LLaMA3-RM-v0.1'
    revision = cfg['selection']['reward_model_revision']
    rm = AutoModelForSequenceClassification.from_pretrained(name,revision=revision,torch_dtype=torch.float32).to(cfg['device']).eval().requires_grad_(False)
    rm_tok = AutoTokenizer.from_pretrained(name,revision=revision)
    results = []
    start = time.perf_counter()
    with torch.no_grad():
        for row in prompts:
            if 'messages' not in row:
                raise ValueError('RM evaluation requires original prompt messages')
            ids = torch.tensor([row['prompt_ids']],device=cfg['device'])
            torch.manual_seed(cfg['seed']+911)
            response = model.generate(ids,attention_mask=torch.ones_like(ids),pad_token_id=tokenizer.eos_token_id,
                                      eos_token_id=tokenizer.eos_token_id,**decode)[0,len(row['prompt_ids']):]
            text = tokenizer.decode(response,skip_special_tokens=True)
            messages = row['messages']+[{'role':'assistant','content':text}]
            rm_text = rm_tok.apply_chat_template(messages,tokenize=False)
            inputs = rm_tok(rm_text,add_special_tokens=False,return_tensors='pt').to(cfg['device'])
            score = rm(**inputs).logits
            if score.numel() != 1 or not torch.isfinite(score).all():
                raise ValueError('nonfinite/non-scalar reward')
            results.append({'schema':1,'split':'test','synthetic':False,'prompt_id':row.get('id',str(len(results))),
                            'messages':row['messages'],'response':text,'generated_token_ids':response.tolist(),
                            'rm_score':score.item(),'reward_model':name,'reward_model_revision':revision,
                            'evaluation_decoding':decode})
    atomic_bytes(output,(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in results)).encode())
    summary = {'mean_rm_score':statistics.mean(r['rm_score'] for r in results),'n_prompts':len(results),
               'generated_tokens':sum(len(r['generated_token_ids']) for r in results),
               'wall_seconds':time.perf_counter()-start,'reward_model':name,'revision':revision,'split':'test'}
    write_json(output.with_suffix('.summary.json'),summary)
    return summary


NARVAL_OPEN_LLM = (('hellaswag','HellaSwag'),('arc','ARC'),('mmlu','MMLU'),('truthfulqa','TruthfulQA'),
                   ('winogrande','Winogrande'),('gsm8k','GSM8K'))
MT_BENCH_CATEGORIES = ('writing','roleplay','reasoning','math','coding','extraction','stem','humanities')
NARVAL_TABLE2 = tuple(('mt_bench_'+c,c.title()) for c in MT_BENCH_CATEGORIES)+(
    ('mt_bench_avg','MT avg'),('hh_lc_win_rate','HH LC WR'),('hh_win_rate','HH WR'))


def report_narval(measured: list[dict[str,Any]], reported: list[dict[str,Any]], allow_incomplete: bool = False) -> str:
    """NARVAL Table 1/2 column layout; reported rows are rendered apart and never enter any test.

    Measured rows: method, pair, seed, metrics, synthetic=false, split=test. Reported rows:
    source 'NARVAL Table 1' or 'NARVAL Table 2', reported=true, method, pair, metrics.
    """
    known = {m for m,_ in NARVAL_OPEN_LLM}|{m for m,_ in NARVAL_TABLE2}
    for row in measured:
        if row.get('reported') or row.get('synthetic',True) or row.get('split') != 'test':
            raise ValueError('measured NARVAL rows must be non-synthetic held-out test outcomes')
    for row in reported:
        if row.get('reported') is not True or row.get('source') not in ('NARVAL Table 1','NARVAL Table 2'):
            raise ValueError('reported rows need reported=true and source NARVAL Table 1/2')
    for row in measured+reported:
        if set(row['metrics'])-known or not all(math.isfinite(v) for v in row['metrics'].values()):
            raise ValueError('unknown or nonfinite NARVAL metric')
    if not measured and not allow_incomplete:
        raise ValueError('no measured rows; use --allow-incomplete for a reported-only layout check')
    lines = ['# NARVAL-comparable results'+('' if measured and not allow_incomplete else ' (incomplete)'),'',
             'Disclosure: crpd_main uses prompts only (labels come from the teacher pair); label-using baselines '
             'see human pairs, and crpd_anchor is the like-for-like comparison.',
             'Reported rows are copied from the NARVAL paper, not measured here, and are excluded from every '
             'significance test. Missing values are not estimates.']
    for title,columns in (('Table 1 layout: Open LLM',NARVAL_OPEN_LLM+(('open_llm_avg','Avg'),)),
                          ('Table 2 layout: MT-Bench and HH-RLHF',NARVAL_TABLE2)):
        for pair in sorted({r['pair'] for r in measured+reported}):
            lines += ['',f'## {title}: {pair}']
            for label,rows in (('Measured here (mean over training seeds)',measured),('Reported by NARVAL (not measured)',reported)):
                subset = [r for r in rows if r['pair'] == pair]
                if not subset:
                    continue
                lines += ['',f'### {label}','','| Method | '+' | '.join(c for _,c in columns)+' |',
                          '|---|'+'---|'*len(columns)]
                for method in sorted({r['method'] for r in subset}):
                    group = [r for r in subset if r['method'] == method]
                    means = {m:statistics.mean(r['metrics'][m] for r in group if m in r['metrics'])
                             for m,_ in NARVAL_OPEN_LLM+NARVAL_TABLE2 if any(m in r['metrics'] for r in group)}
                    if all(m in means for m,_ in NARVAL_OPEN_LLM):
                        means['open_llm_avg'] = statistics.mean(means[m] for m,_ in NARVAL_OPEN_LLM)
                    tag = ' (synthetic placeholder)' if any(r.get('synthetic') for r in group) else ''
                    lines.append(f'| {method}{tag} | '+' | '.join(f'{means[m]:.2f}' if m in means else 'missing'
                                                                    for m,_ in columns)+' |')
    return '\n'.join(lines)+'\n'
