"""Winogrande 5-shot for vukien2301/llama-3.2-1b-deita-sft-student (the Llama-3.2-1B SFT student of NARVAL Table 1,
Winogrande 62.90) in fp16 (user request), fp32 (default harness) and bf16, hf backend, lm-eval 0.4.9.2.
Results: crpd-quick-eval/winogrande-vukien-llama1b/.

  modal run script/modal/eval_vukien/winogrande_vukien_llama1b.py
"""
import json
import subprocess
import sys
from pathlib import Path

import modal

MODEL = 'vukien2301/llama-3.2-1b-deita-sft-student,revision=5900e0a4fa809ab32a534ad8a5003c9a2bb28e32'
OUT = Path('/results/winogrande-vukien-llama1b')
app = modal.App('crpd-wino-vukien-llama1b')
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
    for line in evaluate.map(['float16', 'float32', 'bfloat16']):
        print('RESULT', line)
