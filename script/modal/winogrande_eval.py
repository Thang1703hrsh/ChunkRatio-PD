"""Modal quick check: one lm-eval task for CRPD students, compared with rho question by question.

A single benchmark on one seed: a first look, not a CRPD-versus-baseline result. Settings follow
script/eval/general (wino.sh: 5 few-shot examples; qa.sh: truthfulqa_mc2, none) through lm-eval's hf
backend. Checkpoints are read from their volume without writing to it; results go to crpd-quick-eval.

CRPD_EVAL_TASK picks winogrande (default) or truthfulqa_mc2. CRPD_EVAL_SET picks the models: `x10` or
`stability`. Winogrande on `stability` runs the two sweep students and reuses the stored results of its
rho and alpha 0.3 student; truthfulqa_mc2 on `stability` runs every student, rho and both teachers.

Usage (repository root; one L40S per model, hard 30-minute cap each):

  modal run script/modal/winogrande_eval.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = 'crpd-quick-eval'
CHECKPOINTS = '/source/output/narval/qwen3_1p7b_to_0p6b_hub/pilot'
RHO = f'{CHECKPOINTS}/checkpoints/student_ref'
STUDENT = CHECKPOINTS+'/{}/seed0/student'
TEACHERS = {'teacher_base':'ayushshah/Qwen3-1.7B-UltraChat-SFT,revision=698d09a459656c6909e2b79c6827b66c7a10d70b',
            'teacher_plus':'ayushshah/Qwen3-1.7B-Chat,revision=2e46c30d9fba4ed1aaa8f21003e0de58fa7cf944'}
# hda trio: rho, T0 and T+ are pinned published checkpoints; only the students live on the volume.
HDA_STUDENT = '/source/output/narval/qwen3_1p7b_to_0p6b_hda/pilot/{}/seed0/student'
HDA_SNAPSHOT = '/source/output/narval/qwen3_1p7b_to_0p6b_hda/pilot/{}-snapshots/seed0/snapshots/step{}'
HDA_MODELS = {'rho':'hoangducanh1865/qwen3-0.6b-deita-sft-student,revision=fc0f225cb63e5db6e84a7e5705c997891ec85d65',
              'teacher_base':'hoangducanh1865/qwen3-1.7b-deita-sft-student,revision=52f7c14ea4893128252ada7a365c957813820070',
              'teacher_plus':'hoangducanh1865/qwen3-1.7b-ultrafeedback-dpo-teacher,revision=97a6de251a5e3f78494c6892fb4fbdf9b53c9f04'}
TASK = os.environ.get('CRPD_EVAL_TASK','winogrande')
FEWSHOT = {'winogrande':5,'truthfulqa_mc2':0}
if TASK not in FEWSHOT:
    raise SystemExit('CRPD_EVAL_TASK must be winogrande or truthfulqa_mc2')
SETS = {
    'x10':{'volume':'crpd-qwen3-pilot-hub-x10','output':'winogrande-x10','stored':(),
           'models':{'rho':RHO,'alpha0p3':STUDENT.format('crpd_main'),'alpha0p1':STUDENT.format('crpd_main-alpha0p1')},
           'pairs':(('alpha0p3','rho'),('alpha0p1','rho'),('alpha0p1','alpha0p3'))},
    'stability':{'volume':'crpd-qwen3-pilot-hub-stability','output':'winogrande','stored':('rho','crpd_main'),
                 'models':{'alpha0p1':STUDENT.format('crpd_main-alpha0p1'),'alpha1p0':STUDENT.format('crpd_main-alpha1p0')},
                 'pairs':(('alpha0p1','rho'),('alpha1p0','rho'),('alpha0p1','crpd_main'),('alpha1p0','crpd_main'))},
    'hda-x10':{'volume':'crpd-qwen3-pilot-hda-x10','output':'winogrande-hda-x10','stored':(),
               'models':{**HDA_MODELS,'main_alpha0p03':HDA_STUDENT.format('crpd_main-alpha0p03-v2'),
                         'raw_alpha0p03':HDA_STUDENT.format('crpd_raw-alpha0p03-v2')},
               'pairs':(('main_alpha0p03','rho'),('raw_alpha0p03','rho'),('main_alpha0p03','raw_alpha0p03'),
                        ('teacher_plus','teacher_base'))},
    # Students saved every 50 steps by script/modal/policy_snapshots.py, each compared with rho.
    'hda-x10-steps':{'volume':'crpd-qwen3-pilot-hda-x10','output':'winogrande-hda-x10-steps','stored':(),
                     # main's snapshot run stopped after step 200; its step 300 is the x10 v2 export (best = step 300),
                     # and raw's exported student is evaluated beside its step-300 snapshot as a reproduction check.
                     'models':{'rho':HDA_MODELS['rho'],
                               **{f'raw_step{step}':HDA_SNAPSHOT.format('crpd_raw-alpha0p03-v2',step) for step in range(50,301,50)},
                               **{f'main_step{step}':HDA_SNAPSHOT.format('crpd_main-alpha0p03-v2',step) for step in range(50,201,50)},
                               'main_step300':HDA_STUDENT.format('crpd_main-alpha0p03-v2'),
                               'raw_export300':HDA_STUDENT.format('crpd_raw-alpha0p03-v2')},
                     'pairs':(*((f'raw_step{step}','rho') for step in range(50,301,50)),
                              *((f'main_step{step}','rho') for step in (50,100,150,200,300)),
                              ('raw_step300','raw_export300'))},
    # Turbo crpd_raw x10 (150 steps x 128 pairs) on the trinhson3392 volume, against rho.
    'hda-x10-turbo':{'volume':'crpd-qwen3-pilot-hda-x10','output':'winogrande-hda-x10-turbo','stored':(),
                     'models':{'rho':HDA_MODELS['rho'],'raw_turbo':HDA_STUDENT.format('crpd_raw-alpha0p03-turbo')},
                     'pairs':(('raw_turbo','rho'),)},
    # Latest hda smoke (10 steps, 8 train prompts): a plumbing check of the eval path, not a result.
    'hda-smoke':{'volume':'crpd-qwen3-pilot-hda-smoke','output':'winogrande-hda-smoke','stored':(),
                 'models':{'rho':HDA_MODELS['rho'],'main_alpha0p03':HDA_STUDENT.format('crpd_main-alpha0p03-v2'),
                           'raw_alpha0p03':HDA_STUDENT.format('crpd_raw-alpha0p03-v2')},
                 'pairs':(('main_alpha0p03','rho'),('raw_alpha0p03','rho'),('main_alpha0p03','raw_alpha0p03'))},
}
SET_NAME = os.environ.get('CRPD_EVAL_SET','x10')
if SET_NAME not in SETS:
    raise SystemExit(f'CRPD_EVAL_SET must be one of {", ".join(SETS)}')
SELECTED = SETS[SET_NAME]
if TASK != 'winogrande':
    SELECTED = {**SELECTED,'output':f'{TASK}-{SET_NAME}','stored':()}
    if SET_NAME == 'stability':
        SELECTED['models'] = {'rho':RHO,'crpd_main':STUDENT.format('crpd_main'),**SELECTED['models'],**TEACHERS}
        SELECTED['pairs'] = (('crpd_main','rho'),('alpha0p1','rho'),('alpha1p0','rho'),('teacher_plus','teacher_base'))

app = modal.App(APP_NAME)
results = modal.Volume.from_name(APP_NAME,create_if_missing=True)
source = modal.Volume.from_name(SELECTED['volume']).read_only()
image = (modal.Image.debian_slim(python_version='3.12')
         # lm-eval 0.4.9.2 always loads with dtype=, which transformers accepts from 4.56 (training pins 4.52.4).
         .pip_install('torch==2.6.0','transformers==4.56.2','datasets==3.5.0','lm-eval==0.4.9.2','sentencepiece')
         .env({'HF_HOME':'/results/hf','HF_DATASETS_TRUST_REMOTE_CODE':'1','TOKENIZERS_PARALLELISM':'false',
               'PYTHONUNBUFFERED':'1','CRPD_EVAL_SET':SET_NAME,'CRPD_EVAL_TASK':TASK}))


def read_report(name: str) -> dict:
    """Mean score and per-document scores from one model's latest lm-eval samples.

    Winogrande scores are 0 or 1; truthfulqa_mc2 scores are the normalized true-answer probability.
    """
    samples = sorted((Path('/results')/SELECTED['output']/name).rglob(f'samples_{TASK}_*.jsonl'))[-1]
    scores = {}
    for line in samples.read_text(encoding='utf-8').split('\n'):
        if line.strip():
            row = json.loads(line)
            scores[row['doc_id']] = float(row['acc'])
    return {'name':name,'accuracy':sum(scores.values())/len(scores),'n':len(scores),'correct':scores}


@app.function(image=image,gpu='L40S',cpu=2.0,memory=16384,timeout=30*60,volumes={'/results':results,'/source':source})
def evaluate(name: str) -> dict:
    """One lm-eval task for one model."""
    output = Path('/results')/SELECTED['output']/name
    subprocess.run([sys.executable,'-m','lm_eval','--model','hf','--model_args',f"pretrained={SELECTED['models'][name]},dtype=float32",
                    '--tasks',TASK,'--num_fewshot',str(FEWSHOT[TASK]),'--batch_size','32','--device','cuda','--log_samples',
                    '--output_path',str(output)],check=True)
    results.commit()
    return read_report(name)


@app.function(image=image,cpu=0.25,memory=1024,timeout=5*60,volumes={'/results':results})
def stored(name: str) -> dict:
    """Results an earlier run already wrote; needs no GPU."""
    return read_report(name)


def sign_test(left: dict, right: dict) -> dict:
    """Exact two-sided sign test on the documents where two models differ (McNemar for 0/1 scores)."""
    assert set(left) == set(right)
    wins = sum(left[d] > right[d] for d in left)
    losses = sum(left[d] < right[d] for d in left)
    total = wins+losses
    tail = sum(math.comb(total,k) for k in range(min(wins,losses)+1))/2**total if total else 1.
    return {'only_left_correct':wins,'only_right_correct':losses,'p_two_sided':min(1.,2*tail)}


@app.local_entrypoint()
def main() -> None:
    """Evaluate the selected models once and print mean scores with their paired tests."""
    # stored.map over an empty list never returned (it hung the local run after the GPU work had finished).
    reused = list(stored.map(list(SELECTED['stored']))) if SELECTED['stored'] else []
    reports = {r['name']:r for r in list(evaluate.map(list(SELECTED['models'])))+reused}
    summary = {'set':SET_NAME,'task':TASK,'num_fewshot':FEWSHOT[TASK],'lm_eval':'0.4.9.2',
               'accuracy':{n:r['accuracy'] for n,r in reports.items()},'n':{n:r['n'] for n,r in reports.items()},
               **{f'{a}_vs_{b}':sign_test(reports[a]['correct'],reports[b]['correct']) for a,b in SELECTED['pairs']}}
    print('EVAL SUMMARY '+json.dumps(summary))
