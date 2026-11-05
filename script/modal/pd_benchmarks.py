"""Modal: the six general-capability tasks of the NARVAL paper's Table 1 for the hda trio and CRPD students.

Open LLM Leaderboard settings, as in script/eval/general: ARC-Challenge 25-shot (acc_norm), HellaSwag 10-shot
(acc_norm), MMLU 5-shot (acc), TruthfulQA mc2 0-shot, Winogrande 5-shot (acc), GSM8K 5-shot (exact match,
strict and flexible). lm-eval 0.4.9.2, hf backend, bfloat16 for every model (one precision per table; FP32
ran MMLU at about 2 h per model and GSM8K out of memory at batch 32), no chat template. One L40S container per model and
task, so the tasks of one model run side by side. Results go to the crpd-pd-benchmarks volume; finished
model/task pairs are reused, never recomputed.

Measurement only: one seed, no tuning on these tasks. Our numbers are compared with the paper through each
method's gain over its own SFT model (different harness settings shift absolute scores); the teachers are
evaluated to calibrate our setup against the paper's teacher rows.

  CRPD_BENCH_MODELS=rho,teacher_base,teacher_plus modal run script/modal/pd_benchmarks.py
  CRPD_BENCH_MODELS=rho,raw_gamma1 CRPD_BENCH_PAIRS=raw_gamma1:rho modal run script/modal/pd_benchmarks.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = 'crpd-pd-benchmarks'
ROOT = '/source/output/narval/qwen3_1p7b_to_0p6b_hda/pilot'
STUDENT = ROOT+'/{}/seed0/student'
MODELS = {
    'rho':'hoangducanh1865/qwen3-0.6b-deita-sft-student,revision=fc0f225cb63e5db6e84a7e5705c997891ec85d65',
    'teacher_base':'hoangducanh1865/qwen3-1.7b-deita-sft-student,revision=52f7c14ea4893128252ada7a365c957813820070',
    'teacher_plus':'hoangducanh1865/qwen3-1.7b-ultrafeedback-dpo-teacher,revision=97a6de251a5e3f78494c6892fb4fbdf9b53c9f04',
    'raw_turbo':STUDENT.format('crpd_raw-alpha0p03-turbo'),
    'raw_noend':STUDENT.format('crpd_raw-alpha0p03-noend'),
    'raw_gamma1':STUDENT.format('crpd_raw-alpha0p03-gamma'),
    'raw_gamma0p3':STUDENT.format('crpd_raw-alpha0p03-gamma0p3-gamma'),
    'raw_gamma0p1':STUDENT.format('crpd_raw-alpha0p03-gamma0p1-gamma'),
}
# task: (few-shot, metric key in lm-eval results, per-sample score key)
DTYPE = 'bfloat16'
TASKS = {'arc_challenge':(25,'acc_norm,none','acc_norm'),'hellaswag':(10,'acc_norm,none','acc_norm'),
         'mmlu':(5,'acc,none',None),'truthfulqa_mc2':(0,'acc,none','acc'),'winogrande':(5,'acc,none','acc'),
         'gsm8k':(5,'exact_match,strict-match','exact_match')}
NAMES = [n for n in os.environ.get('CRPD_BENCH_MODELS','rho,teacher_base,teacher_plus').split(',') if n]
PAIRS = [tuple(p.split(':')) for p in os.environ.get('CRPD_BENCH_PAIRS','').split(',') if p]
if not set(NAMES) <= set(MODELS) or not all(len(p) == 2 and set(p) <= set(NAMES) for p in PAIRS):
    raise SystemExit(f'CRPD_BENCH_MODELS must be names from {sorted(MODELS)}; pairs a:b among them')

app = modal.App(APP_NAME)
results = modal.Volume.from_name(APP_NAME,create_if_missing=True)
source = modal.Volume.from_name('crpd-qwen3-pilot-hda-x10').read_only()
image = (modal.Image.debian_slim(python_version='3.12')
         # Same packages as script/modal/winogrande_eval.py (its earlier Winogrande/TruthfulQA numbers).
         .pip_install('torch==2.6.0','transformers==4.56.2','datasets==3.5.0','lm-eval==0.4.9.2','sentencepiece')
         .env({'HF_HOME':'/results/hf','HF_DATASETS_TRUST_REMOTE_CODE':'1','TOKENIZERS_PARALLELISM':'false',
               'PYTHONUNBUFFERED':'1','CRPD_BENCH_MODELS':','.join(NAMES),
               'CRPD_BENCH_PAIRS':','.join(':'.join(p) for p in PAIRS)}))


def report(name: str, task: str) -> dict | None:
    """Mean metric and per-document scores of one finished model/task, or None."""
    folder = Path('/results')/DTYPE/name/task
    found = sorted(folder.rglob('results_*.json'))
    if not found:
        return None
    _,metric,sample_key = TASKS[task]
    data = json.loads(found[-1].read_text(encoding='utf-8'))['results']
    row = {'name':name,'task':task,'metric':metric,'score':data[task][metric]}
    if task == 'gsm8k':
        row['flexible'] = data[task].get('exact_match,flexible-extract')
    scores = {}
    if sample_key:
        for path in sorted(folder.rglob(f'samples_{task}_*.jsonl'))[-1:]:
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    sample = json.loads(line)
                    if task != 'gsm8k' or sample.get('filter') == 'strict-match':
                        scores[sample['doc_id']] = float(sample[sample_key])
    row['per_doc'] = scores
    return row


@app.function(image=image,gpu='L40S',cpu=4.0,memory=32768,timeout=3*3600,volumes={'/results':results,'/source':source})
def evaluate(name: str, task: str) -> dict:
    """One task for one model, reused when it already finished."""
    results.reload()
    done = report(name,task)
    if done is not None:
        print('reuse',name,task,flush=True)
        return done
    fewshot = TASKS[task][0]
    output = Path('/results')/DTYPE/name/task
    subprocess.run([sys.executable,'-m','lm_eval','--model','hf','--model_args',f'pretrained={MODELS[name]},dtype={DTYPE}',
                    '--tasks',task,'--num_fewshot',str(fewshot),'--batch_size','auto:4' if task != 'gsm8k' else '8',
                    '--device','cuda','--log_samples','--output_path',str(output)],check=True)
    results.commit()
    return report(name,task)


def paired(left: dict, right: dict, draws: int = 2000) -> dict:
    """Mean per-document difference left - right with a paired bootstrap 95% interval."""
    docs = sorted(set(left) & set(right))
    diffs = [left[d]-right[d] for d in docs]
    rng = random.Random(0)
    means = sorted(sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(draws))
    return {'n':len(docs),'mean_difference':sum(diffs)/len(diffs),'ci95':[means[int(.025*draws)],means[int(.975*draws)-1]]}


@app.local_entrypoint()
def main() -> None:
    """Every requested model on every task, then the table and the paired comparisons."""
    jobs = [(n,t) for n in NAMES for t in TASKS]
    # One failed task must not stop the others (an exception would stop the whole app).
    outcomes = list(evaluate.starmap(jobs,return_exceptions=True))
    failed = [(job,repr(o)[:300]) for job,o in zip(jobs,outcomes) if not isinstance(o,dict)]
    for job,error in failed:
        print('FAILED',job,error,flush=True)
    rows = {(r['name'],r['task']):r for r in outcomes if isinstance(r,dict)}
    NAMES[:] = [n for n in NAMES if all((n,t) in rows for t in TASKS)]
    PAIRS[:] = [p for p in PAIRS if set(p) <= set(NAMES)]
    table = {n:{t:rows[(n,t)]['score'] for t in TASKS} for n in NAMES}
    for n in NAMES:
        table[n]['avg'] = sum(table[n][t] for t in TASKS)/len(TASKS)
        table[n]['gsm8k_flexible'] = rows[(n,'gsm8k')].get('flexible')
    comparisons = {f'{a}_vs_{b}':{t:paired(rows[(a,t)]['per_doc'],rows[(b,t)]['per_doc'])
                                  for t in TASKS if rows[(a,t)]['per_doc'] and rows[(b,t)]['per_doc']} for a,b in PAIRS}
    summary = {'lm_eval':'0.4.9.2','dtype':DTYPE,'failed':[list(j) for j,_ in failed],'fewshot':{t:v[0] for t,v in TASKS.items()},
               'metric':{t:v[1] for t,v in TASKS.items()},'table':table,'paired':comparisons}
    out = Path('output')/'benchmarks'
    out.mkdir(parents=True,exist_ok=True)
    (out/f"pd_benchmarks_{'-'.join(NAMES)}.json").write_text(json.dumps(summary,indent=1),encoding='utf-8')
    print('BENCH SUMMARY '+json.dumps({'table':table,'paired':comparisons}))
