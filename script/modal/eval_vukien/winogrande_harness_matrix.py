"""Winogrande 5-shot under several harness settings, to find which one reproduces NARVAL Table 1.

Hypothesis 1 (backend/dtype): hf bf16, hf fp16, vLLM dtype=auto (as script/eval/general/wino.sh), vLLM bf16.
Hypothesis 2 (lm-eval version): hf bf16 with lm-eval 0.4.8, 0.4.5, 0.4.2.
The hf fp32 / lm-eval 0.4.9.2 numbers were measured earlier (winogrande-vukien).
"""
import json, subprocess, sys
from pathlib import Path
import modal

APP = 'crpd-wino-matrix'
U = 'vukien2301/'
MODELS = {
    'sft': (U + 'qwen3-0.6b-deita-sft-student', 'df3f07e928671519377a58f2cd399643c5fe65f5'),
    'narval_riskkd': (U + 'qwen3-0.6b-riskkd', 'e3936e47a86bf7d088bc6a70ea7bcc327e93e0a6'),
    'dpo_teacher': (U + 'qwen3-1.7b-ultrafeedback-dpo-teacher', '3ebd9cf9da9b21043cb70529b618cff8d28846ab'),
    'sft_teacher': (U + 'qwen3-1.7b-deita-sft-student', '68735fe3c227a7c62c0f34fe6664de8b3714b039'),
}
PAPER = {'sft': 59.51, 'narval_riskkd': 60.14, 'dpo_teacher': 64.01, 'sft_teacher': 64.40}
ENV = {'HF_HOME': '/root/hf', 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
       'PYTHONUNBUFFERED': '1'}
base = modal.Image.debian_slim(python_version='3.12')
img_new = base.pip_install('torch==2.6.0', 'transformers==4.56.2', 'datasets==3.5.0', 'lm-eval==0.4.9.2',
                           'sentencepiece').env(ENV)
img_vllm = base.pip_install('vllm==0.10.1.1', 'transformers==4.56.2', 'lm-eval==0.4.9.2', 'datasets==3.5.0').env(ENV)


def img_old(v):
    return base.pip_install('torch==2.6.0', 'transformers==4.52.4', 'datasets==3.5.0', f'lm-eval=={v}',
                            'sentencepiece', 'accelerate').env(ENV)


app = modal.App(APP)
vol = modal.Volume.from_name('crpd-quick-eval', create_if_missing=True)
OUT = Path('/results/wino-matrix')


def run(config, name, backend, dtype):
    repo, rev = MODELS[name]
    out = OUT / config / name
    vol.reload()
    done = sorted(out.rglob('results*.json'))
    if done:
        r = json.loads(done[-1].read_text())['results']['winogrande']
        return {'config': config, 'name': name, 'acc': 100 * r.get('acc,none', r.get('acc')), 'reused': True}
    if backend == 'vllm':
        args = f'pretrained={repo},revision={rev},dtype={dtype},gpu_memory_utilization=0.8,max_model_len=4096'
        bs = 'auto'
    else:
        args = f'pretrained={repo},revision={rev},dtype={dtype}'
        bs = '32'
    p = subprocess.run([sys.executable, '-m', 'lm_eval', '--model', backend, '--model_args', args,
                        '--tasks', 'winogrande', '--num_fewshot', '5', '--batch_size', bs,
                        '--output_path', str(out)], capture_output=True, text=True)
    log = p.stdout[-3000:] + p.stderr[-6000:]
    files = sorted(out.rglob('results*.json'))
    if p.returncode or not files:
        return {'config': config, 'name': name, 'error': log[-2500:]}
    r = json.loads(files[-1].read_text())['results']['winogrande']
    acc = r.get('acc,none', r.get('acc'))
    import importlib.metadata as md
    used = [l for l in log.splitlines() if 'dtype' in l.lower()][-3:]
    vol.commit()
    return {'config': config, 'name': name, 'acc': 100 * acc, 'lm_eval': md.version('lm_eval'), 'dtype_log': used}


kw = dict(gpu=['L4', 'A10G', 'L40S'], cpu=4.0, memory=24576, timeout=3600, volumes={'/results': vol})


@app.function(image=img_new, **kw)
def hf_new(config, name, dtype):
    return run(config, name, 'hf', dtype)


@app.function(image=img_vllm, **kw)
def vllm_run(config, name, dtype):
    return run(config, name, 'vllm', dtype)


@app.function(image=img_old('0.4.8'), **kw)
def hf_048(config, name, dtype):
    return run(config, name, 'hf', dtype)


@app.function(image=img_old('0.4.5'), **kw)
def hf_045(config, name, dtype):
    return run(config, name, 'hf', dtype)


@app.function(image=img_old('0.4.2'), **kw)
def hf_042(config, name, dtype):
    return run(config, name, 'hf', dtype)


CONFIGS = {  # config: (function, dtype)
    'hf_bf16_0.4.9.2': (hf_new, 'bfloat16'),
    'hf_fp16_0.4.9.2': (hf_new, 'float16'),
    'vllm_auto_0.4.9.2': (vllm_run, 'auto'),
    'vllm_bf16_0.4.9.2': (vllm_run, 'bfloat16'),
    'hf_bf16_0.4.8': (hf_048, 'bfloat16'),
    'hf_bf16_0.4.5': (hf_045, 'bfloat16'),
    'hf_bf16_0.4.2': (hf_042, 'bfloat16'),
}


@app.local_entrypoint()
def main():
    calls = [(c, n, f.spawn(c, n, d)) for c, (f, d) in CONFIGS.items() for n in MODELS]
    rows = {}
    for c, n, call in calls:
        try:
            r = call.get()
        except Exception as e:
            r = {'config': c, 'name': n, 'error': repr(e)[:500]}
        rows.setdefault(c, {})[n] = r
        print('RESULT ' + json.dumps(r))
    earlier = {'sft': 57.85, 'narval_riskkd': 58.72, 'dpo_teacher': 64.40, 'sft_teacher': 64.17}
    print('\n| Config | ' + ' | '.join(MODELS) + ' | mean |Δ| vs paper |')
    print('|' + '---|' * (len(MODELS) + 2))
    allrows = {'hf_fp32_0.4.9.2 (earlier)': {n: {'acc': v} for n, v in earlier.items()}, **rows}
    for c, rs in allrows.items():
        cells, d = [], []
        for n in MODELS:
            a = rs.get(n, {}).get('acc')
            if a is None:
                cells.append('ERR')
            else:
                cells.append(f'{a:.2f} ({a - PAPER[n]:+.2f})')
                d.append(abs(a - PAPER[n]))
        print(f'| {c} | ' + ' | '.join(cells) + f' | {sum(d) / len(d):.2f} |' if d else ' | — |')
    print('PAPER ' + json.dumps(PAPER))
