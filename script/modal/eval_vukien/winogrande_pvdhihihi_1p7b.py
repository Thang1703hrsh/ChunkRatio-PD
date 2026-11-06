"""Winogrande 5-shot for pvdhihihi/qwen-1.7b-sft (byte-identical copy of vukien2301/qwen3-1.7b-deita-sft-student, the
SFT teacher T0), in the default harness (hf fp32, lm-eval 0.4.9.2) and in bf16, next to NARVAL Table 1 (SFT teacher
64.40). Results: crpd-quick-eval/winogrande-pvdhihihi-1p7b/.

  modal run script/modal/eval_vukien/winogrande_pvdhihihi_1p7b.py
"""
import json
import subprocess
import sys
from pathlib import Path

import modal

MODEL = 'pvdhihihi/qwen-1.7b-sft,revision=0c28858886c981527ee2de4ab5130d3086c3a4e9'
OUT = Path('/results/winogrande-pvdhihihi-1p7b')
app = modal.App('crpd-wino-pvdhihihi-1p7b')
results = modal.Volume.from_name('crpd-quick-eval', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.56.2', 'datasets==3.5.0', 'lm-eval==0.4.9.2', 'sentencepiece')
         .env({'HF_HOME': '/results/hf', 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'PYTHONUNBUFFERED': '1'}))


@app.function(image=image, gpu='L40S', cpu=2.0, memory=24576, timeout=1800, volumes={'/results': results})
def evaluate(dtype: str) -> str:
    out = OUT / dtype
    if not list(out.rglob('results_*.json')):
        p = subprocess.run([sys.executable, '-m', 'lm_eval', '--model', 'hf', '--model_args',
                            f'pretrained={MODEL},dtype={dtype}', '--tasks', 'winogrande', '--num_fewshot', '5',
                            '--batch_size', '32', '--device', 'cuda', '--log_samples', '--output_path', str(out)],
                           capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError('failed: ' + p.stderr[-2500:])
        results.commit()
    r = json.loads(sorted(out.rglob('results_*.json'))[-1].read_text())['results']['winogrande']
    return f"{dtype}: {100 * r['acc,none']:.2f} (stderr {100 * r['acc_stderr,none']:.2f})"


@app.local_entrypoint()
def main():
    for line in evaluate.map(['float32', 'bfloat16']):
        print('RESULT', line)
