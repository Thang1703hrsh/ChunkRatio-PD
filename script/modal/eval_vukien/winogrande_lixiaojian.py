"""Winogrande 5-shot, hf fp32 for Lixiaojian123/qwen3-0.6b-sft, saved with transformers 5 (rope_parameters, list-form
extra_special_tokens), which transformers 4.56.2 cannot read. Same task and settings with a transformers 5 image and
the newest lm-eval; versions did not change Winogrande scores elsewhere (DECISIONS: harness matrix). Compared per
question with the vukien SFT stored in crpd-quick-eval/winogrande-vukien.

  modal run script/modal/eval_vukien/winogrande_lixiaojian.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import modal

MODEL = 'Lixiaojian123/qwen3-0.6b-sft,revision=907e118dcc94938a76a2167f0ee2a41c4d531390'
OUT = Path('/results/winogrande-public-sft/lixiaojian_smoltalk2_sft_tf5')
app = modal.App('crpd-wino-lixiaojian')
results = modal.Volume.from_name('crpd-quick-eval', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.7.1', 'transformers>=5.0,<6', 'datasets==3.5.0', 'lm-eval', 'sentencepiece', 'accelerate')
         .env({'HF_HOME': '/results/hf', 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'PYTHONUNBUFFERED': '1'}))


def read(folder: Path) -> dict:
    samples = sorted(folder.rglob('samples_winogrande_*.jsonl'))[-1]
    return {json.loads(l)['doc_id']: float(json.loads(l)['acc'])
            for l in samples.read_text(encoding='utf-8').split('\n') if l.strip()}


@app.function(image=image, gpu='L40S', cpu=2.0, memory=16384, timeout=1800, volumes={'/results': results})
def evaluate() -> str:
    import importlib.metadata as md
    if not list(OUT.rglob('samples_winogrande_*.jsonl')):
        p = subprocess.run([sys.executable, '-m', 'lm_eval', '--model', 'hf', '--model_args',
                            f'pretrained={MODEL},dtype=float32', '--tasks', 'winogrande', '--num_fewshot', '5',
                            '--batch_size', '32', '--device', 'cuda', '--log_samples', '--output_path', str(OUT)],
                           capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError('failed: ' + p.stderr[-2500:])
        results.commit()
    results.reload()
    a, b = read(OUT), read(Path('/results/winogrande-vukien/sft_student_0p6b'))
    w, l = sum(a[d] > b[d] for d in a), sum(a[d] < b[d] for d in a)
    tail = sum(math.comb(w + l, k) for k in range(min(w, l) + 1)) / 2 ** (w + l)
    return (f"lixiaojian_smoltalk2_sft {100 * sum(a.values()) / len(a):.2f} | vs vukien SFT "
            f"{100 * (sum(a.values()) - sum(b.values())) / len(a):+.2f} | sign {w}/{l} p={min(1., 2 * tail):.3f} | "
            f"transformers {md.version('transformers')} lm-eval {md.version('lm_eval')}")


@app.local_entrypoint()
def main():
    print('RESULT', evaluate.remote())
