"""Winogrande (5-shot) and TruthfulQA-MC2 (0-shot), hf backend, float32: CRPD students against rho and the riskKD
(NARVAL) student of the same pair, with per-question paired tests. The harness of every comparison in this repo
(lm-eval 0.4.9.2, transformers 4.56.2): in it the authors' Llama-3.2-1B SFT, Qwen3-1.7B SFT and DPO checkpoints
reproduce NARVAL Table 1 Winogrande within 0.4 points (DECISIONS 91-93). Run it with the Python of a separate
environment holding requirements-eval.txt (the CRPD environment pins transformers 4.52.4).

  <eval-python> script/server/benchmarks.py --pair qwen3_1p7b_to_0p6b --runs <run> [<run> ...] --gpus 0,1 \
      --output output/server/<pair>/<scale>/benchmarks
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
TASKS = {'winogrande':5,'truthfulqa_mc2':0}


def read(folder: Path, task: str) -> dict:
    """Per-question score: 0/1 for Winogrande, the normalized true-answer probability for TruthfulQA-MC2."""
    samples = sorted(folder.rglob(f'samples_{task}_*.jsonl'))[-1]
    return {json.loads(l)['doc_id']:float(json.loads(l)['acc'])
            for l in samples.read_text(encoding='utf-8').split('\n') if l.strip()}


def paired(a: dict, b: dict, task: str) -> str:
    """Winogrande: exact sign test on differing questions; TruthfulQA: mean difference with a paired bootstrap."""
    if task == 'winogrande':
        w,l = sum(a[d] > b[d] for d in a),sum(a[d] < b[d] for d in a)
        tail = sum(math.comb(w+l,k) for k in range(min(w,l)+1))/2**(w+l) if w+l else 1.
        return f'{w}/{l} p={min(1.,2*tail):.3f}'
    keys = sorted(a)
    diffs = [a[k]-b[k] for k in keys]
    rng = random.Random(0)
    boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs) for _ in range(2000))
    return f'{100*sum(diffs)/len(diffs):+.2f} [{100*boots[50]:+.2f}, {100*boots[1949]:+.2f}]'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair',required=True)
    parser.add_argument('--runs',nargs='*',default=[],help='Finished CRPD run folders (each holds student/)')
    parser.add_argument('--gpus',default='0')
    parser.add_argument('--output',required=True)
    parser.add_argument('--from-hub',action='store_true',
                        help='Evaluate each run as pushed (repo and commit from <run>/hub.json) instead of its local student/')
    parser.add_argument('--models',nargs='*',default=[],
                        help='Extra Hub models as name=repo or name=repo@revision (e.g. a pushed student elsewhere)')
    args = parser.parse_args()
    from chunkratio.riskkd import RISKKD
    spec = RISKKD[args.pair]
    rho,narval = spec['checkpoints']['student_ref'],spec['narval']
    models = {'rho_sft':f"{rho['repo']},revision={rho['revision']}",'narval':f"{narval['repo']},revision={narval['revision']}"}
    for run in args.runs:
        name = Path(run).parent.name if Path(run).name == 'seed0' else Path(run).name
        if args.from_hub:
            hub = Path(run)/'hub.json'
            if not hub.exists():
                raise SystemExit(f'{run}: no hub.json; push it first (run_pair.py --stages push)')
            record = json.loads(hub.read_text(encoding='utf-8'))
            if not record.get('verified'):
                raise SystemExit(f"{record['repo']}: the pushed weights were not verified against the local run")
            models[name] = f"{record['repo']},revision={record['revision']}"
        else:
            models[name] = str(Path(run)/'student')
    for item in args.models:
        name,_,target = item.partition('=')
        repo,_,revision = target.partition('@')
        if not name or not repo or name in models:
            raise SystemExit(f'--models needs distinct name=repo[@revision], got {item!r}')
        models[name] = f'{repo},revision={revision}' if revision else repo
    out = Path(args.output)
    gpus = [g for g in args.gpus.split(',') if g]
    jobs = [(name,task) for task in TASKS for name in models]
    def run_job(index_job: tuple[int,tuple[str,str]]) -> None:
        index,(name,task) = index_job
        folder = out/task/name
        if list(folder.rglob(f'samples_{task}_*.jsonl')):
            return
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':gpus[index % len(gpus)],'HF_DATASETS_TRUST_REMOTE_CODE':'1'}
        subprocess.run([sys.executable,'-m','lm_eval','--model','hf','--model_args',f'pretrained={models[name]},dtype=float32',
                        '--tasks',task,'--num_fewshot',str(TASKS[task]),'--batch_size','32','--device','cuda',
                        '--log_samples','--output_path',str(folder)],env=env,check=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(run_job,enumerate(jobs)))
    lines,summary = [],{}
    for task in TASKS:
        scores = {name:read(out/task/name,task) for name in models}
        lines += [f'## {task} ({TASKS[task]}-shot, hf fp32)','','| Model | Score | vs rho | vs NARVAL |','|---|---:|---|---|']
        for name,s in scores.items():
            value = 100*sum(s.values())/len(s)
            summary.setdefault(task,{})[name] = value
            lines.append(f"| {name} | {value:.2f} | {paired(s,scores['rho_sft'],task) if name != 'rho_sft' else '—'} | "
                         f"{paired(s,scores['narval'],task) if name != 'narval' else '—'} |")
        lines.append('')
    (out/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({'models':models,'scores':summary},indent=1),encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
