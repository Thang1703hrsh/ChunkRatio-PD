"""Modal eval of published checkpoints on Winogrande and TruthfulQA-MC2, for comparison with NARVAL.

Settings follow script/eval/general (wino.sh: winogrande, 5-shot; qa.sh: truthfulqa_mc2, 0-shot) through
lm-eval 0.4.9.2's hf backend, as script/modal/winogrande_eval.py does. Every model runs in bfloat16 so the
table has one precision (some checkpoints are stored in FP32, which dtype=auto would keep). NARVAL's own
lm-eval version and few-shot settings are not verified (null in config/crpd/narval), so this table is a
same-harness comparison between checkpoints, not a reproduction of NARVAL's numbers.

Each model loads once and runs both tasks; models up to 2B use an L4, 8B models an L40S, all in parallel.
Repositories are pinned to the commits below and downloaded to container disk. Results are written to the
`crpd-benchmark-eval` volume; a model whose results exist is skipped, so relaunching resumes. A remote
orchestrator collects the table, so the local machine can be switched off once the launch has printed.

Usage (repository root):

  modal deploy script/modal/benchmark_eval.py
  modal run script/modal/benchmark_eval.py::launch
  modal volume get crpd-benchmark-eval summary.md .     # later: the finished table
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

APP_NAME = 'crpd-benchmark-eval'
RESULTS = Path('/results')
TASKS = {'winogrande':5,'truthfulqa_mc2':0}
# name: (repository, commit resolved on 2 November 2026, size class).
MODELS = {
    'qwen3-1.7b-ultrachat-sft':('ayushshah/Qwen3-1.7B-UltraChat-SFT','698d09a459656c6909e2b79c6827b66c7a10d70b','small'),
    'qwen3-1.7b-chat':('ayushshah/Qwen3-1.7B-Chat','2e46c30d9fba4ed1aaa8f21003e0de58fa7cf944','small'),
    'qwen3-1.7b-chat-lora':('ayushshah/Qwen3-1.7B-Chat-LoRA','c1a6e68d4e8adb95ef8c535d705985da6c7b6f15','small'),
    'qwen3-8b-deita-sft':('hoangducanh1865/qwen3-8b-deita-sft-teacher','7acddcfb521973f123e3b258252ae5e5123661af','large'),
    'qwen3-8b-uf-dpo':('hoangducanh1865/qwen3-8b-ultrafeedback-dpo-teacher','c53ee14c7488980af49831cfdc14bda57a0de5ff','large'),
    'qwen3-1.7b-deita-sft':('hoangducanh1865/qwen3-1.7b-deita-sft-student','52f7c14ea4893128252ada7a365c957813820070','small'),
    'qwen3-1.7b-uf-dpo':('hoangducanh1865/qwen3-1.7b-ultrafeedback-dpo-teacher','97a6de251a5e3f78494c6892fb4fbdf9b53c9f04','small'),
    'qwen3-0.6b-deita-sft':('hoangducanh1865/qwen3-0.6b-deita-sft-student','fc0f225cb63e5db6e84a7e5705c997891ec85d65','small'),
    'llama-3.2-1b-deita-sft':('hoangducanh1865/llama-3.2-1b-deita-sft-student','10eb4106f315aa1206e5c5350ef43d3548727edf','small'),
    'llama-3.2-1b-rlhflow-sft':('RLHFlow/LLaMA3.2-1B-SFT','bc1c0eafef4582ec4937ec6dc7988a36f7e3cc22','small'),
    'llama-3.2-1b-fusechat-sft':('FuseAI/FuseChat-Llama-3.2-1B-SFT','2982491cb86a5638bcc675ace781d9c20d1d029e','small'),
    'redsage-qwen3-8b-ins':('RISys-Lab/RedSage-Qwen3-8B-Ins','7c98b61983b146e70f4113eb0d7cfd9e5aaec709','large'),
    'redsage-qwen3-8b-dpo':('RISys-Lab/RedSage-Qwen3-8B-DPO','04ec0b7738ede6717d86fa67993ade30c02dc79e','large'),
    'llama-3.1-8b-rolemrc-sft':('jiazhengli/Llama-3.1-8B-RoleMRC-sft','6aa3b65a7ea2547b7e8db70e605fce44ad9877fd','large'),
    'llama-3.1-8b-rolemrc-dpo':('jiazhengli/Llama-3.1-8B-RoleMRC-dpo','38d07b2e56dd6ba87e0a6615be6f36bcc0b2d9b1','large'),
}
# The adapter's adapter_config.json names ayushshah/Qwen3-1.7B-UltraChat-SFT as its base.
ADAPTER_BASE = {'qwen3-1.7b-chat-lora':'qwen3-1.7b-ultrachat-sft'}
# SFT -> DPO pairs, plus the adapter against its merged release (expected equal up to bf16 rounding).
PAIRS = (('qwen3-1.7b-chat','qwen3-1.7b-ultrachat-sft'),('qwen3-1.7b-chat-lora','qwen3-1.7b-chat'),
         ('qwen3-8b-uf-dpo','qwen3-8b-deita-sft'),('qwen3-1.7b-uf-dpo','qwen3-1.7b-deita-sft'),
         ('redsage-qwen3-8b-dpo','redsage-qwen3-8b-ins'),('llama-3.1-8b-rolemrc-dpo','llama-3.1-8b-rolemrc-sft'))

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME,create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         # Same pins as winogrande_eval.py; peft for the adapter, hf_transfer for parallel downloads.
         .pip_install('torch==2.6.0','transformers==4.56.2','datasets==3.5.0','lm-eval==0.4.9.2','sentencepiece',
                      'peft==0.17.1','hf_transfer==0.1.9')
         .env({'HF_HOME':'/root/hf','HF_HUB_ENABLE_HF_TRANSFER':'1','HF_DATASETS_TRUST_REMOTE_CODE':'1',
               'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}))


def snapshot(name: str) -> str:
    """Download one pinned repository to container disk and return its local path."""
    from huggingface_hub import snapshot_download
    repo,revision,_ = MODELS[name]
    return snapshot_download(repo,revision=revision,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])


def evaluate(name: str) -> dict:
    """Both tasks for one model, loaded once; an existing result is returned without a GPU pass."""
    folder = RESULTS/'models'/name
    done = folder/'result.json'
    if done.exists():
        return json.loads(done.read_text(encoding='utf-8'))
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    start = time.time()
    try:
        # lm-eval passes one revision to both base and adapter, so both are given as pinned local paths.
        base = ADAPTER_BASE.get(name)
        lm = HFLM(pretrained=snapshot(base or name),peft=snapshot(name) if base else None,dtype='bfloat16',
                  batch_size='auto',max_batch_size=64,device='cuda')
        result = {'name':name,'repository':MODELS[name][0],'revision':MODELS[name][1],'dtype':'bfloat16',
                  'lm_eval':'0.4.9.2','tasks':{}}
        folder.mkdir(parents=True,exist_ok=True)
        for task,shots in TASKS.items():
            out = simple_evaluate(model=lm,tasks=[task],num_fewshot=shots,log_samples=True)
            metrics = out['results'][task]
            scores = {int(s['doc_id']):float(s['acc']) for s in out['samples'][task]}
            (folder/f'{task}.scores.json').write_text(json.dumps(scores),encoding='utf-8')
            (folder/f'{task}.results.json').write_text(json.dumps({k:v for k,v in out.items() if k != 'samples'},
                                                                  indent=2,default=str),encoding='utf-8')
            result['tasks'][task] = {'num_fewshot':shots,'acc':metrics['acc,none'],
                                     'stderr':metrics.get('acc_stderr,none'),'n':len(scores)}
        result['wall_s'] = round(time.time()-start,1)
        result['gpu_mem_max_gb'] = round(torch.cuda.max_memory_allocated()/2**30,2)
        tmp = done.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2),encoding='utf-8')
        tmp.replace(done)
    except Exception as error:  # one failing checkpoint must not stop the others
        result = {'name':name,'error':f'{type(error).__name__}: {error}','wall_s':round(time.time()-start,1)}
        (RESULTS/'errors').mkdir(parents=True,exist_ok=True)
        (RESULTS/'errors'/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    volume.commit()
    return result


common = {'image':image,'volumes':{str(RESULTS):volume},'retries':1}


# GPU lists are fallbacks in order of preference: on 2 November 2026 L4 jobs waited over 10 minutes for capacity.
@app.function(gpu=['L4','A10G','L40S'],cpu=4.0,memory=12288,timeout=2*3600,**common)
def evaluate_small(name: str) -> dict:
    """Models up to 2B: bf16 weights need under 4 GB of a 24 GB GPU."""
    return evaluate(name)


@app.function(gpu=['L40S','A100-40GB','A100-80GB'],cpu=4.0,memory=32768,timeout=3*3600,**common)
def evaluate_large(name: str) -> dict:
    """8B models: about 16 GB of bf16 weights plus batches on a 40-80 GB GPU."""
    return evaluate(name)


def paired(left: dict, right: dict) -> dict:
    """Paired difference on shared documents: mean, normal 95% interval and exact sign test."""
    docs = sorted(set(left)&set(right))
    diffs = [left[d]-right[d] for d in docs]
    mean = sum(diffs)/len(diffs)
    sd = math.sqrt(sum((x-mean)**2 for x in diffs)/(len(diffs)-1)) if len(diffs) > 1 else 0.
    wins,losses = sum(x > 0 for x in diffs),sum(x < 0 for x in diffs)
    total = wins+losses
    tail = sum(math.comb(total,k) for k in range(min(wins,losses)+1))/2**total if total else 1.
    half = 1.96*sd/math.sqrt(len(diffs))
    return {'mean_difference':mean,'low':mean-half,'high':mean+half,'n':len(docs),'wins':wins,'losses':losses,
            'p_sign_test':min(1.,2*tail)}


def summary_markdown(results: dict, comparisons: list) -> str:
    """Score table and paired comparisons as Markdown."""
    def cell(r: dict, task: str) -> str:
        t = r.get('tasks',{}).get(task)
        return f"{100*t['acc']:.2f} ± {100*(t['stderr'] or 0):.2f}" if t else 'failed'
    lines = ['| Model | Repository @ commit | Winogrande 5-shot acc (%) | TruthfulQA-MC2 0-shot (%) |','|---|---|---|---|']
    for name,(repo,revision,_) in MODELS.items():
        r = results.get(name,{})
        lines.append(f"| {name} | {repo} @ {revision[:10]} | {cell(r,'winogrande')} | {cell(r,'truthfulqa_mc2')} |")
    lines += ['','| Comparison | Task | Mean diff (pts) | 95% CI | wins/losses | sign-test p |','|---|---|---|---|---|---|']
    for c in comparisons:
        lines.append(f"| {c['left']} − {c['right']} | {c['task']} | {100*c['mean_difference']:+.2f} | "
                     f"[{100*c['low']:+.2f}, {100*c['high']:+.2f}] | {c['wins']}/{c['losses']} | {c['p_sign_test']:.3g} |")
    failed = [f"- {n}: {r['error']}" for n,r in results.items() if 'error' in r]
    return '\n'.join(lines+(['','Failed:']+failed if failed else []))+'\n'


@app.function(image=image,cpu=0.25,memory=1024,timeout=6*3600,volumes={str(RESULTS):volume})
def orchestrate() -> dict:
    """Evaluate every model in parallel and write summary.json and summary.md to the volume."""
    calls = {n:(evaluate_large if size == 'large' else evaluate_small).spawn(n) for n,(_,_,size) in MODELS.items()}
    results = {}
    for name,call in calls.items():
        try:
            results[name] = call.get()
        except Exception as error:
            results[name] = {'name':name,'error':f'{type(error).__name__}: {error}'}
    volume.reload()
    comparisons = []
    for left,right in PAIRS:
        if 'error' in results[left] or 'error' in results[right]:
            continue
        for task in TASKS:
            scores = [json.loads((RESULTS/'models'/n/f'{task}.scores.json').read_text(encoding='utf-8')) for n in (left,right)]
            comparisons.append({'left':left,'right':right,'task':task,**paired(*scores)})
    summary = {'tasks':TASKS,'dtype':'bfloat16','lm_eval':'0.4.9.2','results':results,'comparisons':comparisons}
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (RESULTS/'summary.md').write_text(summary_markdown(results,comparisons),encoding='utf-8')
    volume.commit()
    print(summary_markdown(results,comparisons))
    return summary


@app.local_entrypoint()
def launch() -> None:
    """Start the deployed orchestrator and return at once; the run continues without this machine."""
    call = modal.Function.from_name(APP_NAME,'orchestrate').spawn()
    print(f'started orchestrate call {call.object_id}; results: modal volume get {APP_NAME} summary.md .')
