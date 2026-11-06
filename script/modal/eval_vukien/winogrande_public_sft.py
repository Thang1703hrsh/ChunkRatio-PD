"""Winogrande 5-shot, hf fp32, lm-eval 0.4.9.2 (the default harness of these comparisons) for public Qwen3-0.6B SFT
checkpoints and Qwen3-0.6B-Base, compared per question with the vukien SFT (rho) and NARVAL riskkd stored in
crpd-quick-eval/winogrande-vukien. The LoRA adapter is evaluated on its declared base through lm-eval's peft argument.
Summary: crpd-quick-eval/winogrande-public-sft/summary.md

  modal run script/modal/eval_vukien/winogrande_public_sft.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import modal

BASE = 'Qwen/Qwen3-0.6B-Base,revision=da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
MODELS = {
    # Its tokenizer_config lists extra_special_tokens as a list, which transformers 4.56.2 cannot read; the
    # Qwen3-0.6B-Base tokenizer (same BPE vocabulary and merges) is used instead.
    'lixiaojian_smoltalk2_sft': ('Lixiaojian123/qwen3-0.6b-sft,revision=907e118dcc94938a76a2167f0ee2a41c4d531390,'
                                 'tokenizer=Qwen/Qwen3-0.6B-Base'),
    'wassname_ultrachat_sft': 'wassname/Qwen3-0.6B-sft,revision=27620bc22fc9acaec16fac2dc3709e9a9e3431ed',
    # lm-eval passes `revision` to the adapter as well, so the base is named without it; Qwen3-0.6B-Base main was
    # da87bfb6 when this ran (checked through the Hub API), and the adapter is read at its main (b9939c77).
    'ermia_ultrachat_lora': 'Qwen/Qwen3-0.6B-Base,peft=ermiaazarkhalili/Qwen3-0.6B-SFT-UltraChat',
    'qwen3_0p6b_base': BASE,
}
OUT = Path('/results/winogrande-public-sft')
app = modal.App('crpd-wino-public-sft')
results = modal.Volume.from_name('crpd-quick-eval', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.56.2', 'datasets==3.5.0', 'lm-eval==0.4.9.2', 'sentencepiece',
                      'peft==0.17.1')
         .env({'HF_HOME': '/results/hf', 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'PYTHONUNBUFFERED': '1'}))


def read(folder: Path) -> dict:
    """Per-question 0/1 accuracy from the latest lm-eval samples file."""
    samples = sorted(folder.rglob('samples_winogrande_*.jsonl'))[-1]
    scores = {}
    for line in samples.read_text(encoding='utf-8').split('\n'):
        if line.strip():
            row = json.loads(line)
            scores[row['doc_id']] = float(row['acc'])
    return scores


@app.function(image=image, gpu='L40S', cpu=2.0, memory=16384, timeout=1800, volumes={'/results': results})
def evaluate(name: str) -> tuple:
    out = OUT / name
    if not list(out.rglob('samples_winogrande_*.jsonl')):
        p = subprocess.run([sys.executable, '-m', 'lm_eval', '--model', 'hf', '--model_args',
                        f'pretrained={MODELS[name]},dtype=float32', '--tasks', 'winogrande', '--num_fewshot', '5',
                        '--batch_size', '32', '--device', 'cuda', '--log_samples', '--output_path', str(out)],
                       capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f'{name} failed: ' + p.stderr[-3000:])
        results.commit()
    return name, read(out)


def sign(a: dict, b: dict) -> str:
    """Exact two-sided sign test on the questions where two models differ."""
    w = sum(a[d] > b[d] for d in a)
    l = sum(a[d] < b[d] for d in a)
    t = w + l
    tail = sum(math.comb(t, k) for k in range(min(w, l) + 1)) / 2 ** t if t else 1.
    return f'{w}/{l} p={min(1., 2 * tail):.3f}'


@app.function(image=image, cpu=0.5, memory=2048, timeout=600, volumes={'/results': results})
def summarize(new: dict) -> str:
    results.reload()
    ref = {'vukien_sft_rho': read(Path('/results/winogrande-vukien/sft_student_0p6b')),
           'narval_riskkd': read(Path('/results/winogrande-vukien/narval_riskkd'))}
    acc = lambda s: 100 * sum(s.values()) / len(s)
    lines = ['| Model | Winogrande | vs vukien SFT (paper 59.51) | sign vs vukien SFT |', '|---|---:|---:|---|']
    for n, s in {**ref, **new}.items():
        lines.append(f"| {n} | {acc(s):.2f} | {acc(s) - acc(ref['vukien_sft_rho']):+.2f} | "
                     f"{sign(s, ref['vukien_sft_rho'])} |")
    text = '\n'.join(lines) + '\n'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.md').write_text(text)
    results.commit()
    return text


@app.local_entrypoint()
def main():
    new = {}
    for r in evaluate.map(list(MODELS), return_exceptions=True):
        if isinstance(r, tuple):
            new[r[0]] = r[1]
        else:
            print('EVAL ERROR', repr(r)[:600])
    print(summarize.remote(new))
