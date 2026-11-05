"""Winogrande 5-shot (lm-eval 0.4.9.2, fp32) for the vukien2301 NARVAL checkpoints; same settings as
script/modal/winogrande_eval.py in the ChunkRatio-PD repo."""
import json, math, subprocess, sys
from pathlib import Path
import modal

APP = 'crpd-quick-eval'
MODELS = {
    'narval_riskkd': 'vukien2301/qwen3-0.6b-riskkd,revision=e3936e47a86bf7d088bc6a70ea7bcc327e93e0a6',
    'dpo_teacher_1p7b': 'vukien2301/qwen3-1.7b-ultrafeedback-dpo-teacher,revision=3ebd9cf9da9b21043cb70529b618cff8d28846ab',
    'sft_student_0p6b': 'vukien2301/qwen3-0.6b-deita-sft-student,revision=df3f07e928671519377a58f2cd399643c5fe65f5',
    'pfw_tail': 'vukien2301/qwen3-0.6b-pfw-tail,revision=a196a905a3c8c92ee0b0e933b6ea4379919f62fa',
}
PAPER = {'narval_riskkd': 60.14, 'dpo_teacher_1p7b': 64.01, 'sft_student_0p6b': 59.51}
OUT = 'winogrande-vukien'

app = modal.App(APP)
results = modal.Volume.from_name(APP, create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.56.2', 'datasets==3.5.0', 'lm-eval==0.4.9.2', 'sentencepiece')
         .env({'HF_HOME': '/results/hf', 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'PYTHONUNBUFFERED': '1'}))


def read_report(name):
    samples = sorted((Path('/results') / OUT / name).rglob('samples_winogrande_*.jsonl'))[-1]
    scores = {}
    for line in samples.read_text(encoding='utf-8').split('\n'):
        if line.strip():
            row = json.loads(line)
            scores[row['doc_id']] = float(row['acc'])
    return {'name': name, 'accuracy': sum(scores.values()) / len(scores), 'n': len(scores), 'correct': scores}


@app.function(image=image, gpu='L40S', cpu=2.0, memory=16384, timeout=30 * 60, volumes={'/results': results})
def evaluate(name):
    out = Path('/results') / OUT / name
    subprocess.run([sys.executable, '-m', 'lm_eval', '--model', 'hf', '--model_args',
                    f'pretrained={MODELS[name]},dtype=float32', '--tasks', 'winogrande', '--num_fewshot', '5',
                    '--batch_size', '32', '--device', 'cuda', '--log_samples', '--output_path', str(out)], check=True)
    results.commit()
    return read_report(name)


def sign_test(a, b):
    w = sum(a[d] > b[d] for d in a); l = sum(a[d] < b[d] for d in a); t = w + l
    tail = sum(math.comb(t, k) for k in range(min(w, l) + 1)) / 2 ** t if t else 1.
    return {'only_left_correct': w, 'only_right_correct': l, 'p_two_sided': min(1., 2 * tail)}


@app.function(image=image, cpu=0.25, memory=1024, timeout=5 * 60, volumes={'/results': results})
def stored(name):
    return read_report(name)


@app.local_entrypoint()
def main():
    new = ['pfw_tail']
    reps = {r['name']: r for r in evaluate.map(new)}
    reps.update({r['name']: r for r in stored.map([n for n in MODELS if n not in new])})
    summary = {'task': 'winogrande', 'num_fewshot': 5, 'dtype': 'float32', 'lm_eval': '0.4.9.2',
               'revisions': MODELS,
               'accuracy_pct': {n: round(100 * r['accuracy'], 2) for n, r in reps.items()},
               'paper_pct': PAPER, 'n': {n: r['n'] for n, r in reps.items()},
               'narval_vs_sft': sign_test(reps['narval_riskkd']['correct'], reps['sft_student_0p6b']['correct']),
               'pfw_tail_vs_sft': sign_test(reps['pfw_tail']['correct'], reps['sft_student_0p6b']['correct']),
               'pfw_tail_vs_narval': sign_test(reps['pfw_tail']['correct'], reps['narval_riskkd']['correct'])}
    print('EVAL SUMMARY ' + json.dumps(summary))
