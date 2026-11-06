"""Winogrande (5-shot) and TruthfulQA-MC2 (0-shot), hf fp32, lm-eval 0.4.9.2, for the 2,000-prompt pilot student,
the x10 gamma 0.05 student, rho (SFT) and NARVAL riskkd, with per-question paired tests against rho and NARVAL.
Winogrande per-question results of rho and riskkd are reused from crpd-quick-eval/winogrande-vukien (same settings).
Summary: crpd-quick-eval/pilot-benchmarks/summary.md

  modal run --detach script/modal/eval_vukien/pilot_benchmarks.py
"""
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import modal

PILOT = 'output/narval/qwen3_1p7b_to_0p6b_hda/pilot'
MODELS = {  # name: (volume mount, path or HF repo,revision)
    'pilot2k_gamma0p05': ('/x30', f'/x30/{PILOT}/crpd_raw-alpha0p03-gamma0p05-p2k/seed0/student'),
    'x10_gamma0p05': ('/x10', f'/x10/{PILOT}/crpd_raw-alpha0p03-gamma0p05-g01/seed0/student'),
    'rho_sft': (None, 'vukien2301/qwen3-0.6b-deita-sft-student,revision=df3f07e928671519377a58f2cd399643c5fe65f5'),
    'narval_riskkd': (None, 'vukien2301/qwen3-0.6b-riskkd,revision=e3936e47a86bf7d088bc6a70ea7bcc327e93e0a6'),
}
STORED_WINO = {'rho_sft': 'winogrande-vukien/sft_student_0p6b', 'narval_riskkd': 'winogrande-vukien/narval_riskkd'}
TASKS = {'winogrande': 5, 'truthfulqa_mc2': 0}
OUT = Path('/results/pilot-benchmarks')
app = modal.App('crpd-pilot-benchmarks')
results = modal.Volume.from_name('crpd-quick-eval', create_if_missing=True)
x10 = modal.Volume.from_name('crpd-qwen3-pilot-hda-x10').read_only()
x30_rw = modal.Volume.from_name('crpd-qwen3-pilot-hda-x30', create_if_missing=True)
x30 = x30_rw.read_only()
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.56.2', 'datasets==3.5.0', 'lm-eval==0.4.9.2', 'sentencepiece')
         .env({'HF_HOME': '/results/hf', 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'PYTHONUNBUFFERED': '1'}))


def read(folder: Path, task: str) -> dict:
    """Per-question score: 0/1 accuracy for Winogrande, the normalized true-answer probability for TQA-MC2."""
    samples = sorted(folder.rglob(f'samples_{task}_*.jsonl'))[-1]
    scores = {}
    for line in samples.read_text(encoding='utf-8').split('\n'):
        if line.strip():
            row = json.loads(line)
            scores[row['doc_id']] = float(row['acc'])
    return scores


@app.function(image=image, gpu='L40S', cpu=2.0, memory=16384, timeout=3600,
              volumes={'/results': results, '/x10': x10, '/x30': x30})
def evaluate(name: str, task: str) -> tuple:
    results.reload()
    if task == 'winogrande' and name in STORED_WINO:
        return name, task, read(Path('/results') / STORED_WINO[name], task)
    out = OUT / task / name
    if not list(out.rglob(f'samples_{task}_*.jsonl')):
        subprocess.run([sys.executable, '-m', 'lm_eval', '--model', 'hf', '--model_args',
                        f'pretrained={MODELS[name][1]},dtype=float32', '--tasks', task, '--num_fewshot',
                        str(TASKS[task]), '--batch_size', '32', '--device', 'cuda', '--log_samples',
                        '--output_path', str(out)], check=True)
        results.commit()
    return name, task, read(out, task)


def paired(a: dict, b: dict, task: str) -> str:
    """Winogrande: exact sign test on differing questions. TQA: mean difference with a paired bootstrap 95% CI."""
    if task == 'winogrande':
        w = sum(a[d] > b[d] for d in a)
        l = sum(a[d] < b[d] for d in a)
        t = w + l
        tail = sum(math.comb(t, k) for k in range(min(w, l) + 1)) / 2 ** t if t else 1.
        return f'{w}/{l} p={min(1., 2 * tail):.3f}'
    keys = sorted(a)
    diffs = [a[k] - b[k] for k in keys]
    rng = random.Random(0)
    boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(2000))
    return f'{100 * sum(diffs) / len(diffs):+.2f} [{100 * boots[50]:+.2f}, {100 * boots[1949]:+.2f}]'


@app.function(image=image, cpu=0.5, memory=4096, timeout=3600, volumes={'/results': results})
def summarize(scores: dict) -> str:
    lines = []
    for task in TASKS:
        s = {n: v for (n, t), v in scores.items() if t == task}
        lines += [f'## {task} ({TASKS[task]}-shot, hf fp32, lm-eval 0.4.9.2)', '',
                  '| Model | Score | vs rho (paired) | vs NARVAL (paired) |', '|---|---:|---|---|']
        for n, v in s.items():
            score = 100 * sum(v.values()) / len(v)
            lines.append(f"| {n} | {score:.2f} | {paired(v, s['rho_sft'], task) if n != 'rho_sft' else '—'} | "
                         f"{paired(v, s['narval_riskkd'], task) if n != 'narval_riskkd' else '—'} |")
        lines.append('')
    text = '\n'.join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.md').write_text(text)
    results.commit()
    return text


def run_benchmarks() -> str:
    """Every (model, task) job, then the paired summary."""
    jobs = [(n, t) for t in TASKS for n in MODELS]
    scores = {}
    for r in evaluate.starmap(jobs, return_exceptions=True):
        if isinstance(r, tuple):
            scores[(r[0], r[1])] = r[2]
        else:
            print('EVAL ERROR', repr(r)[:400], flush=True)
    return summarize.remote(scores)


@app.function(image=image, cpu=0.25, memory=1024, timeout=24 * 3600, volumes={'/x30': x30_rw})
def after_pilot() -> str:
    """Server-side chain (survives the local machine sleeping): wait for the pilot's diagnostics.json, then start
    the deployed held-out pd_diagnostics (crpd-pd-diagnostics, deployed for this run) and the benchmarks."""
    import time
    done = Path(f'/x30/{PILOT}/crpd_raw-alpha0p03-gamma0p05-p2k/seed0/diagnostics.json')
    while True:
        x30_rw.reload()
        if done.exists():
            break
        time.sleep(120)
    print('pilot finished; starting held-out diagnostics and benchmarks', flush=True)
    modal.Function.from_name('crpd-pd-diagnostics', 'diagnose').spawn()
    text = run_benchmarks()
    print(text, flush=True)
    return text


@app.local_entrypoint()
def main():
    print(run_benchmarks())
