"""Server-side (nuocb0739): fetch the four latest CRPD hda x10 students from trinhson3392's volume (read-only API
use with the `trinhson-token` secret; trinhson3392 runs no compute), store them in crpd-quick-eval/crpd-students,
then Winogrande 5-shot, hf fp32, lm-eval 0.4.9.2 (the default harness), compared per question with NARVAL riskkd,
rho and pfw-tail (stored per-sample results in winogrande-vukien). Summary: crpd-quick-eval/winogrande-crpd-latest/summary.md

  modal deploy wino_crpd.py
  python -c "import modal; print(modal.Function.from_name('crpd-wino-crpd','orchestrate').spawn().object_id)"
  modal volume get crpd-quick-eval winogrande-crpd-latest/summary.md .      (later)
"""
import json, math, os, shutil, subprocess, sys
from pathlib import Path
import modal

APP = 'crpd-wino-crpd'
PILOT = 'output/narval/qwen3_1p7b_to_0p6b_hda/pilot'
STUDENTS = {'gamma0p1': 'gamma0p1-gamma', 'gamma0p3': 'gamma0p3-gamma', 'gamma1': 'gamma', 'noend': 'noend'}
OUT = Path('/results/winogrande-crpd-latest')
app = modal.App(APP)
results = modal.Volume.from_name('crpd-quick-eval', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.56.2', 'datasets==3.5.0', 'lm-eval==0.4.9.2', 'sentencepiece',
                      'modal==1.6.3')
         .env({'HF_HOME': '/results/hf', 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'PYTHONUNBUFFERED': '1'}))


@app.function(image=image, cpu=2.0, memory=4096, timeout=2 * 3600, volumes={'/results': results},
              secrets=[modal.Secret.from_name('trinhson-token')])
def fetch(name):
    dest = Path('/results/crpd-students') / STUDENTS[name]
    if (dest / 'model.safetensors').exists():
        return name
    env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL_')}
    env.update(MODAL_TOKEN_ID=os.environ['SRC_TOKEN_ID'], MODAL_TOKEN_SECRET=os.environ['SRC_TOKEN_SECRET'])
    tmp = Path('/tmp/dl') / name
    tmp.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(['modal', 'volume', 'get', 'crpd-qwen3-pilot-hda-x10',
                        f'{PILOT}/crpd_raw-alpha0p03-{STUDENTS[name]}/seed0/student/', '.', '--force'],
                       env=env, cwd=tmp, capture_output=True, text=True)
    print('GET', name, p.returncode, (p.stdout + p.stderr)[-1500:], flush=True)
    listing = [(str(f), f.stat().st_size) for root in (tmp, Path('/root'), Path('/tmp'))
               for f in root.rglob('*') if f.is_file() and f.suffix in ('.safetensors', '.json')][:40]
    print('FILES', name, listing, flush=True)
    hits = [f for root in (tmp, Path('/root')) for f in root.rglob('*.safetensors')]
    if not hits:
        raise RuntimeError(f'no safetensors after download for {name}')
    found = hits[0].parent
    shutil.copytree(found, dest, dirs_exist_ok=True)
    results.commit()
    return name


def read(folder):
    samples = sorted(Path(folder).rglob('samples_winogrande_*.jsonl'))[-1]
    s = {}
    for line in samples.read_text(encoding='utf-8').split('\n'):
        if line.strip():
            row = json.loads(line)
            s[row['doc_id']] = float(row['acc'])
    return s


@app.function(image=image, gpu='L40S', cpu=2.0, memory=16384, timeout=30 * 60, volumes={'/results': results})
def evaluate(name):
    results.reload()
    out = OUT / name
    if not list(out.rglob('samples_winogrande_*.jsonl')):
        subprocess.run([sys.executable, '-m', 'lm_eval', '--model', 'hf', '--model_args',
                        f'pretrained=/results/crpd-students/{STUDENTS[name]},dtype=float32', '--tasks', 'winogrande',
                        '--num_fewshot', '5', '--batch_size', '32', '--device', 'cuda', '--log_samples',
                        '--output_path', str(out)], check=True)
        results.commit()
    return name, read(out)


def sign(a, b):
    w = sum(a[d] > b[d] for d in a); l = sum(a[d] < b[d] for d in a); t = w + l
    tail = sum(math.comb(t, k) for k in range(min(w, l) + 1)) / 2 ** t if t else 1.
    return f'{w}/{l} p={min(1., 2 * tail):.3f}'


@app.function(image=image, cpu=0.5, memory=2048, timeout=4 * 3600, volumes={'/results': results})
def orchestrate():
    fetched = [r for r in fetch.map(list(STUDENTS), return_exceptions=True) if isinstance(r, str)]
    reps = {}
    for r in evaluate.map(fetched, return_exceptions=True):
        if isinstance(r, tuple):
            reps[r[0]] = r[1]
        else:
            print('EVAL ERROR', repr(r)[:500])
    results.reload()
    v = '/results/winogrande-vukien'
    ref = {'narval_riskkd': read(f'{v}/narval_riskkd'), 'rho_sft': read(f'{v}/sft_student_0p6b'),
           'pfw_tail': read(f'{v}/pfw_tail')}
    acc = lambda s: 100 * sum(s.values()) / len(s)
    lines = ['| Model | Winogrande | vs NARVAL | sign vs NARVAL | sign vs rho |', '|---|---:|---:|---|---|']
    for n, s in {**ref, **reps}.items():
        lines.append(f"| {n} | {acc(s):.2f} | {acc(s) - acc(ref['narval_riskkd']):+.2f} | "
                     f"{sign(s, ref['narval_riskkd'])} | {sign(s, ref['rho_sft'])} |")
    text = '\n'.join(lines) + f"\n\nfetched: {fetched}\nmissing: {[n for n in STUDENTS if n not in reps]}\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.md').write_text(text)
    results.commit()
    print(text)
    return text
