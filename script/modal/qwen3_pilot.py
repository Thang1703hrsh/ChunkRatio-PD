"""Modal launcher: ChunkRatio-PD pilot, Qwen3-1.7B teacher pair -> Qwen3-0.6B student.

Stages (idempotent; outputs and Hugging Face cache live on the Modal Volume `crpd-qwen3-pilot`):

  phase 1 (run_pilot): prepare data (CPU) -> SFT 1.7B (A100-80GB) || SFT 0.6B (L40S) -> DPO 1.7B (A100-80GB)
  phase 2 (finish):    displacement diagnostic, 100 prompts (L40S) -> crpd_main, crpd_raw, crpd_anchor (A100-80GB)

Each phase is a separate function call so each stays inside Modal's 24 h limit. GPUs are attached
only to the stages that need them; orchestrators use a quarter CPU. A finished stage writes a
marker and is skipped on re-run; CRPD stages resume from their own shards.

Usage (run from the repository root with the Python that has `modal` installed):

  modal deploy script/modal/qwen3_pilot.py
  python script/modal/qwen3_pilot.py start      # spawn phase 1 (+2) on Modal and return
  python script/modal/qwen3_pilot.py finish     # spawn phase 2 only (after phase 1 finished)

Unverified settings are listed in chunkratio/pilot.py::PILOT_ASSUMPTIONS and in every config.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = 'crpd-qwen3-pilot'
REPO = Path(__file__).resolve().parents[2]
REMOTE_REPO = '/root/repo'
VOLUME_PATH = '/vol'
PAIR = 'qwen3_1p7b_to_0p6b'
DATA = f'data/narval/{PAIR}'
CHECKPOINTS = f'output/narval/{PAIR}/pilot/checkpoints'
VARIANTS = ('crpd_main','crpd_raw','crpd_anchor')

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME,create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0','transformers==4.52.4','datasets==3.5.0','numpy','pyyaml','safetensors','tokenizers')
         .env({'HF_HOME':f'{VOLUME_PATH}/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'})
         .add_local_dir(REPO,REMOTE_REPO,ignore=['output','data','.git','mtbench','winrate_eval','diversity_metrics',
                                                 'promt','prompts','env','reports','**/__pycache__','.pytest_cache']))
common = {'image':image,'volumes':{VOLUME_PATH:volume},'timeout':24*3600}


def workdir() -> Path:
    """Read-only code plus volume-backed output/ and data/, so configs keep relative paths."""
    work = Path('/root/work')
    work.mkdir(exist_ok=True)
    for name in ('chunkratio','config','reference','script'):
        if not (work/name).exists():
            (work/name).symlink_to(Path(REMOTE_REPO)/name)
    for name in ('output','data'):
        (Path(VOLUME_PATH)/name).mkdir(parents=True,exist_ok=True)
        if not (work/name).exists():
            (work/name).symlink_to(Path(VOLUME_PATH)/name)
    return work


def crpd(revision: str, *args: str) -> None:
    """Run one chunkratio CLI command in the work directory."""
    work = workdir()
    print('+ python -m chunkratio',' '.join(args),flush=True)
    subprocess.run([sys.executable,'-m','chunkratio',*args],cwd=work,check=True,
                   env={**os.environ,'CRPD_CODE_REVISION':revision})


def done(path: str) -> bool:
    """Stage marker on the volume (relative to the work directory)."""
    return (workdir()/path).exists()


def train_checkpoint(revision: str, stage: str, model: str, model_revision: str, data: str, out: str) -> None:
    """SFT or teacher DPO with the assumed recipe; skipped when the checkpoint manifest exists."""
    volume.reload()
    if done(f'{out}/preparation_manifest.json'):
        print('skip',out,flush=True)
        return
    sys.path.insert(0,str(workdir()))
    from chunkratio.pilot import preparation_config
    config = workdir()/f'{out}.config.json'
    config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps(preparation_config(stage,model,model_revision,data,out),indent=2),encoding='utf-8')
    crpd(revision,stage,'--config',str(config.relative_to(workdir())))
    volume.commit()


@app.function(cpu=4.0,memory=16384,**common)
def prepare(revision: str) -> None:
    """Pin revisions, download Deita and UltraFeedback, build canonical data and the pilot subset (CPU)."""
    volume.reload()
    work = workdir()
    sys.path.insert(0,str(work))
    from chunkratio.artifacts import read_jsonl, write_json
    from chunkratio.pilot import (PREFERENCE_DATASET, REWARD_MODEL, SFT_DATASET, STUDENT_BASE_MODEL,
                                  TEACHER_BASE_MODEL, pilot_subset)
    if done(f'{DATA}/prepared.json'):
        print('skip prepare',flush=True)
        return
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    revisions_path = work/DATA/'revisions.json'
    if revisions_path.exists():
        revisions = json.loads(revisions_path.read_text(encoding='utf-8'))
    else:
        revisions = {'teacher_base_model':api.model_info(TEACHER_BASE_MODEL).sha,
                     'student_base_model':api.model_info(STUDENT_BASE_MODEL).sha,
                     'reward_model':api.model_info(REWARD_MODEL).sha,
                     'sft_dataset':api.dataset_info(SFT_DATASET).sha,
                     'preference_dataset':api.dataset_info(PREFERENCE_DATASET).sha}
        write_json(revisions_path,revisions)
    raw = work/DATA/'raw'
    raw.mkdir(parents=True,exist_ok=True)
    for name,dataset,split,key in (('deita.jsonl',SFT_DATASET,'train','sft_dataset'),
                                   ('uf_train_prefs.jsonl',PREFERENCE_DATASET,'train_prefs','preference_dataset'),
                                   ('uf_test_prefs.jsonl',PREFERENCE_DATASET,'test_prefs','preference_dataset')):
        if not (raw/name).exists():
            load_dataset(dataset,split=split,revision=revisions[key]).to_json(str(raw/name),lines=True,force_ascii=False)
    template = json.dumps({'enable_thinking':False})
    tokenizer = ['--tokenizer',STUDENT_BASE_MODEL,'--revision',revisions['student_base_model'],
                 '--eos-token','<|im_end|>','--chat-template-kwargs',template]
    crpd(revision,'prepare-data','--input',f'{DATA}/raw/deita.jsonl','--output',f'{DATA}/deita_sft.jsonl',
         '--format','sharegpt','--assign-split','train',*tokenizer)
    crpd(revision,'prepare-data','--input',f'{DATA}/raw/uf_train_prefs.jsonl','--output',f'{DATA}/ultrafeedback_train.jsonl',
         '--format','ultrafeedback',*tokenizer)
    crpd(revision,'prepare-data','--input',f'{DATA}/raw/uf_test_prefs.jsonl','--output',f'{DATA}/ultrafeedback_test.jsonl',
         '--format','ultrafeedback','--assign-split','test',*tokenizer)
    subset = pilot_subset(read_jsonl(work/DATA/'ultrafeedback_train.jsonl'))
    (work/DATA/'ultrafeedback_train_pilot.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),
                                                             encoding='utf-8')
    write_json(work/DATA/'prepared.json',{'revisions':revisions,'pilot_rows':len(subset),'code_revision':revision})
    volume.commit()


def _revisions() -> dict:
    """Revisions pinned by prepare()."""
    return json.loads((workdir()/DATA/'revisions.json').read_text(encoding='utf-8'))


@app.function(gpu='A100-80GB',cpu=4.0,memory=32768,retries=1,**common)
def sft_teacher(revision: str) -> None:
    """T0: SFT of Qwen3-1.7B-Base on Deita (full fine-tuning, FP32 weights, TF32 matmuls)."""
    volume.reload()
    train_checkpoint(revision,'train-sft','Qwen/Qwen3-1.7B-Base',_revisions()['teacher_base_model'],
                     f'{DATA}/deita_sft.jsonl',f'{CHECKPOINTS}/teacher_base')


@app.function(gpu='L40S',cpu=4.0,memory=16384,retries=1,**common)
def sft_student(revision: str) -> None:
    """rho: SFT of Qwen3-0.6B-Base on the same Deita data (L40S is enough for 0.6B)."""
    volume.reload()
    train_checkpoint(revision,'train-sft','Qwen/Qwen3-0.6B-Base',_revisions()['student_base_model'],
                     f'{DATA}/deita_sft.jsonl',f'{CHECKPOINTS}/student_ref')


@app.function(gpu='A100-80GB',cpu=4.0,memory=32768,retries=1,**common)
def dpo_teacher(revision: str) -> None:
    """T+: DPO of T0 on UltraFeedback train_prefs; the frozen DPO reference is exactly T0."""
    volume.reload()
    train_checkpoint(revision,'train-teacher-dpo',f'{CHECKPOINTS}/teacher_base','local',
                     f'{DATA}/ultrafeedback_train.jsonl',f'{CHECKPOINTS}/teacher_plus')


def write_pilot_configs() -> dict[str,str]:
    """Resolved pilot configs from the NARVAL protocol file; T+'s DPO reference must equal T0."""
    work = workdir()
    sys.path.insert(0,str(work))
    from chunkratio.pilot import pilot_config
    manifests = {role:json.loads((work/CHECKPOINTS/role/'preparation_manifest.json').read_text(encoding='utf-8'))
                 for role in ('teacher_base','teacher_plus','student_ref')}
    verified = manifests['teacher_plus']['teacher_dpo_reference_fingerprint'] == manifests['teacher_base']['model_fingerprint']
    checkpoints = {role:{'path':f'{CHECKPOINTS}/{role}','revision':manifests[role]['model_fingerprint']} for role in manifests}
    split = json.loads((work/DATA/'ultrafeedback_train.splits.json').read_text(encoding='utf-8'))['assignment_sha256']
    protocol = json.loads((work/'config/crpd/narval'/f'{PAIR}.json').read_text(encoding='utf-8'))
    paths = {}
    for variant in VARIANTS:
        path = work/f'output/narval/{PAIR}/pilot/configs/{variant}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(pilot_config(protocol,variant,checkpoints,_revisions(),split,verified),indent=2),encoding='utf-8')
        paths[variant] = str(path.relative_to(work))
    return paths


@app.function(gpu='L40S',cpu=4.0,memory=24576,**common)
def diagnose(revision: str) -> None:
    """Displacement diagnostic on 100 pilot train prompts before any student training."""
    volume.reload()
    out = f'output/narval/{PAIR}/pilot/diagnostic'
    if done(f'{out}/displacement.json'):
        print('skip diagnose',flush=True)
        return
    configs = write_pilot_configs()
    crpd(revision,'diagnose-displacement','--config',configs['crpd_main'],'--prompts','100','--replicates','16','--output',out)
    volume.commit()


@app.function(gpu='A100-80GB',cpu=8.0,memory=65536,retries=1,**common)
def train_variants(revision: str) -> None:
    """crpd_main, crpd_raw, crpd_anchor (seed 0) sharing one cache_root; each resumes from its shards."""
    volume.reload()
    configs = write_pilot_configs()
    for variant in VARIANTS:
        config = json.loads((workdir()/configs[variant]).read_text(encoding='utf-8'))
        if done(f"{config['output']}/diagnostics.json"):
            print('skip',variant,flush=True)
            continue
        crpd(revision,'evaluate','--config',configs[variant])
        volume.commit()


@app.function(cpu=0.25,memory=1024,**common)
def finish(revision: str) -> None:
    """Phase 2 orchestrator: diagnostic, then the three variants."""
    diagnose.remote(revision)
    train_variants.remote(revision)


@app.function(cpu=0.25,memory=1024,**common)
def run_pilot(revision: str) -> None:
    """Phase 1 orchestrator; spawns phase 2 so each phase has its own 24 h budget."""
    prepare.remote(revision)
    calls = [sft_teacher.spawn(revision),sft_student.spawn(revision)]
    for call in calls:
        call.get()
    dpo_teacher.remote(revision)
    finish.spawn(revision)


def local_revision() -> str:
    """Git HEAD of the uploaded code, marked dirty when the work tree has changes."""
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    return head+('-dirty' if dirty else '')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'start'
    if command not in ('start','finish'):
        raise SystemExit('usage: python script/modal/qwen3_pilot.py [start|finish]')
    function = modal.Function.from_name(APP_NAME,'run_pilot' if command == 'start' else 'finish')
    call = function.spawn(local_revision())
    print(f'spawned {command} on Modal app {APP_NAME}: {call.object_id}')
    print(f'logs: modal app logs {APP_NAME}    files: modal volume ls {APP_NAME} output/narval/{PAIR}/pilot')
