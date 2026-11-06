"""Winogrande 5-shot, hf fp32, lm-eval 0.4.9.2 (the default harness of these comparisons) for the gamma sweep
students on the nuocb0739 hda x10 volume, compared per question with the stored NARVAL riskkd, rho (SFT),
pfw-tail and the first gamma 0.1 student (crpd-quick-eval volume). Summary: crpd-quick-eval/winogrande-gamma-sweep/.

  modal run script/modal/eval_vukien/winogrande_gamma_sweep.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import modal

PILOT = '/train/output/narval/qwen3_1p7b_to_0p6b_hda/pilot'
STUDENTS = {'gamma0p1_rerun': 'crpd_raw-alpha0p03-gamma0p1-g01', 'gamma0p05': 'crpd_raw-alpha0p03-gamma0p05-g01',
            'gamma0p03': 'crpd_raw-alpha0p03-gamma0p03-g01'}
OUT = Path('/results/winogrande-gamma-sweep')
app = modal.App('crpd-wino-gamma-sweep')
results = modal.Volume.from_name('crpd-quick-eval', create_if_missing=True)
train = modal.Volume.from_name('crpd-qwen3-pilot-hda-x10').read_only()
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.56.2', 'datasets==3.5.0', 'lm-eval==0.4.9.2', 'sentencepiece')
         .env({'HF_HOME': '/results/hf', 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'PYTHONUNBUFFERED': '1'}))


def read(folder: str) -> dict:
    """Per-question 0/1 scores from the latest lm-eval samples file."""
    samples = sorted(Path(folder).rglob('samples_winogrande_*.jsonl'))[-1]
    scores = {}
    for line in samples.read_text(encoding='utf-8').split('\n'):
        if line.strip():
            row = json.loads(line)
            scores[row['doc_id']] = float(row['acc'])
    return scores


@app.function(image=image, gpu='L40S', cpu=2.0, memory=16384, timeout=1800,
              volumes={'/results': results, '/train': train})
def evaluate(name: str) -> tuple:
    out = OUT / name
    if not list(out.rglob('samples_winogrande_*.jsonl')):
        subprocess.run([sys.executable, '-m', 'lm_eval', '--model', 'hf', '--model_args',
                        f'pretrained={PILOT}/{STUDENTS[name]}/seed0/student,dtype=float32', '--tasks', 'winogrande',
                        '--num_fewshot', '5', '--batch_size', '32', '--device', 'cuda', '--log_samples',
                        '--output_path', str(out)], check=True)
        results.commit()
    return name, read(str(out))


def sign(a: dict, b: dict) -> str:
    """Exact two-sided sign test on the questions where two models differ."""
    w = sum(a[d] > b[d] for d in a)
    l = sum(a[d] < b[d] for d in a)
    t = w + l
    tail = sum(math.comb(t, k) for k in range(min(w, l) + 1)) / 2 ** t if t else 1.
    return f'{w}/{l} p={min(1., 2 * tail):.3f}'


@app.function(image=image, cpu=0.5, memory=2048, timeout=3600, volumes={'/results': results})
def summarize(new: dict) -> str:
    results.reload()
    ref = {'narval_riskkd': read('/results/winogrande-vukien/narval_riskkd'),
           'rho_sft': read('/results/winogrande-vukien/sft_student_0p6b'),
           'pfw_tail': read('/results/winogrande-vukien/pfw_tail'),
           'gamma0p1_first': read('/results/winogrande-crpd-latest/gamma0p1')}
    acc = lambda s: 100 * sum(s.values()) / len(s)
    lines = ['| Model | Winogrande | vs NARVAL | sign vs NARVAL | sign vs rho |', '|---|---:|---:|---|---|']
    for n, s in {**ref, **new}.items():
        lines.append(f"| {n} | {acc(s):.2f} | {acc(s) - acc(ref['narval_riskkd']):+.2f} | "
                     f"{sign(s, ref['narval_riskkd'])} | {sign(s, ref['rho_sft'])} |")
    text = '\n'.join(lines) + '\n'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.md').write_text(text)
    results.commit()
    return text


@app.local_entrypoint()
def main():
    new = {}
    for r in evaluate.map(list(STUDENTS), return_exceptions=True):
        if isinstance(r, tuple):
            new[r[0]] = r[1]
        else:
            print('EVAL ERROR', repr(r)[:400])
    print(summarize.remote(new))
