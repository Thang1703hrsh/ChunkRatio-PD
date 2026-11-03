"""Modal launcher for the 8B NARVAL pairs: Qwen3-8B -> Qwen3-1.7B and Llama-3.1-8B -> Llama-3.2-1B.

Same stages as script/modal/qwen3_pilot.py (which stays the launcher for Qwen3-1.7B -> Qwen3-0.6B):

  phase 1 (run_pilot): prepare data (CPU) -> SFT teacher || SFT student -> DPO teacher
  phase 2 (finish):    displacement diagnostic, 100 prompts -> crpd_main, crpd_raw, crpd_anchor

Choose the pair with CRPD_PAIR for both deploy and start; each pair is its own app and volume.
Full FP32 AdamW fine-tuning of an 8B model needs about 128 GB (DPO adds a frozen 32 GB reference),
so 8B stages run on B200 (192 GB); CRPD holds two FP32 8B teachers plus the 8B RM on B200 too.
Llama repos are gated: create the Modal secret `huggingface` with HF_TOKEN first.

  $env:CRPD_PAIR = "qwen3_8b_to_1p7b"          # or llama31_8b_to_llama32_1b
  modal deploy script/modal/narval_pair.py
  python script/modal/narval_pair.py start      # spawn phase 1 (+2) and return
  python script/modal/narval_pair.py finish     # phase 2 only

Unverified settings are listed per pair by chunkratio/pilot.py::pilot_assumptions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REMOTE_REPO = '/root/repo'
# Inside a container this file is /root/narval_pair.py, so the repo root only exists locally.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_REPO)
sys.path.insert(0,str(REPO))
from chunkratio.pilot import MAX_LENGTH, PAIRS  # noqa: E402

# (gpu, cpu, memory MiB, timeout h) per stage; the default pair has its own launcher.
HARDWARE = {
    'qwen3_8b_to_1p7b':{'sft_teacher':('B200',8.,98304,10),'sft_student':('A100-80GB',4.,32768,4),
                        'dpo_teacher':('B200',8.,131072,22),'diagnose':('H200',4.,65536,4),
                        'variants':('B200',8.,131072,24)},
    'llama31_8b_to_llama32_1b':{'sft_teacher':('B200',8.,98304,10),'sft_student':('L40S',4.,32768,4),
                                'dpo_teacher':('B200',8.,131072,22),'diagnose':('H200',4.,65536,4),
                                'variants':('B200',8.,131072,24)},
}
PAIR = os.environ.get('CRPD_PAIR','')
if PAIR not in HARDWARE:
    raise SystemExit(f'set CRPD_PAIR to one of {sorted(HARDWARE)} (qwen3_1p7b_to_0p6b uses qwen3_pilot.py)')
SPEC = PAIRS[PAIR]
APP_NAME = 'crpd-'+PAIR.replace('_','-')
VOLUME_PATH = '/vol'
DATA = f'data/narval/{PAIR}'
CHECKPOINTS = f'output/narval/{PAIR}/pilot/checkpoints'
SFT_DATA = f'{DATA}/deita_sft_fit2048.jsonl'
VARIANTS = ('crpd_main','crpd_raw','crpd_anchor')

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME,create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0','transformers==4.52.4','datasets==3.5.0','numpy','pyyaml','safetensors','tokenizers')
         .env({'HF_HOME':f'{VOLUME_PATH}/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','CRPD_PAIR':PAIR})
         .add_local_dir(REPO,REMOTE_REPO,ignore=['output','data','.git','mtbench','winrate_eval','diversity_metrics',
                                                 'promt','prompts','env','reports','**/__pycache__','.pytest_cache']))
secrets = [modal.Secret.from_name('huggingface')] if SPEC['gated'] else []


def stage(name: str, retries: int = 0) -> dict:
    """Modal resources of one stage; preparation stages fail deterministically, so no retries."""
    gpu,cpu,memory,hours = HARDWARE[PAIR][name]
    return {'image':image,'volumes':{VOLUME_PATH:volume},'secrets':secrets,'gpu':gpu,'cpu':cpu,
            'memory':memory,'timeout':hours*3600,'retries':retries}


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


def finished(config: dict, marker: str) -> bool:
    """A CRPD run is finished only if its marker exists and it ran exactly this config; an output folder
    holding another config's results is an error, not a skip."""
    if not done(f"{config['output']}/{marker}"):
        return False
    resolved = workdir()/config['output']/'resolved_config.json'
    if not resolved.exists() or json.loads(resolved.read_text(encoding='utf-8')) != config:
        raise ValueError(f"{config['output']} holds results of a different config; use another output to rerun")
    return True


def _revisions() -> dict:
    """Revisions pinned by prepare()."""
    return json.loads((workdir()/DATA/'revisions.json').read_text(encoding='utf-8'))


def train_checkpoint(revision: str, kind: str, model: str, model_revision: str, data: str, out: str,
                     shared_tokenizer: bool) -> None:
    """SFT or teacher DPO with the assumed recipe; skipped when the checkpoint manifest exists."""
    volume.reload()
    if done(f'{out}/preparation_manifest.json'):
        print('skip',out,flush=True)
        return
    from chunkratio.pilot import preparation_config
    tokenizer = (SPEC['tokenizer'],_revisions()['tokenizer']) if shared_tokenizer else (None,None)
    config = workdir()/f'{out}.config.json'
    config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps(preparation_config(kind,model,model_revision,data,out,PAIR,*tokenizer),indent=2),
                      encoding='utf-8')
    crpd(revision,kind,'--config',str(config.relative_to(workdir())))
    volume.commit()


@app.function(image=image,volumes={VOLUME_PATH:volume},secrets=secrets,cpu=4.0,memory=16384,timeout=6*3600)
def prepare(revision: str) -> None:
    """Pin revisions, download Deita and UltraFeedback, build canonical data and the pilot subset (CPU)."""
    volume.reload()
    work = workdir()
    from chunkratio.artifacts import read_jsonl, write_json
    from chunkratio.pilot import PREFERENCE_DATASET, REWARD_MODEL, SFT_DATASET, pilot_subset
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
        revisions = {'teacher_base_model':api.model_info(SPEC['teacher']).sha,
                     'student_base_model':api.model_info(SPEC['student']).sha,
                     'tokenizer':api.model_info(SPEC['tokenizer']).sha,
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
    tokenizer = ['--tokenizer',SPEC['tokenizer'],'--revision',revisions['tokenizer'],'--eos-token',SPEC['eos'],
                 '--chat-template-kwargs',json.dumps(SPEC['chat_template_kwargs'])]
    # Each output is built once; a later code version can add a file without redoing the others.
    if not done(SFT_DATA):
        crpd(revision,'prepare-data','--input',f'{DATA}/raw/deita.jsonl','--output',SFT_DATA,
             '--format','sharegpt','--assign-split','train','--fit-max-length',str(MAX_LENGTH),*tokenizer)
    if not done(f'{DATA}/ultrafeedback_train.jsonl'):
        crpd(revision,'prepare-data','--input',f'{DATA}/raw/uf_train_prefs.jsonl','--output',f'{DATA}/ultrafeedback_train.jsonl',
             '--format','ultrafeedback',*tokenizer)
    if not done(f'{DATA}/ultrafeedback_test.jsonl'):
        crpd(revision,'prepare-data','--input',f'{DATA}/raw/uf_test_prefs.jsonl','--output',f'{DATA}/ultrafeedback_test.jsonl',
             '--format','ultrafeedback','--assign-split','test',*tokenizer)
    subset = pilot_subset(read_jsonl(work/DATA/'ultrafeedback_train.jsonl'))
    (work/DATA/'ultrafeedback_train_pilot.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),
                                                             encoding='utf-8')
    write_json(work/DATA/'prepared.json',{'pair':PAIR,'revisions':revisions,'pilot_rows':len(subset),
                                          'code_revision':revision,'sft_data':SFT_DATA})
    volume.commit()


@app.function(**stage('sft_teacher'))
def sft_teacher(revision: str) -> None:
    """T0: full SFT of the teacher Base model on turn-fitted Deita (FP32 weights, bf16 autocast)."""
    volume.reload()
    train_checkpoint(revision,'train-sft',SPEC['teacher'],_revisions()['teacher_base_model'],SFT_DATA,
                     f'{CHECKPOINTS}/teacher_base',True)


@app.function(**stage('sft_student'))
def sft_student(revision: str) -> None:
    """rho: SFT of the student Base model on the same data with the same recipe."""
    volume.reload()
    train_checkpoint(revision,'train-sft',SPEC['student'],_revisions()['student_base_model'],SFT_DATA,
                     f'{CHECKPOINTS}/student_ref',True)


@app.function(**stage('dpo_teacher'))
def dpo_teacher(revision: str) -> None:
    """T+: DPO of T0 on UltraFeedback train_prefs; the frozen DPO reference is exactly T0."""
    volume.reload()
    train_checkpoint(revision,'train-teacher-dpo',f'{CHECKPOINTS}/teacher_base','local',
                     f'{DATA}/ultrafeedback_train.jsonl',f'{CHECKPOINTS}/teacher_plus',False)


def write_pilot_configs() -> dict[str,str]:
    """Resolved pilot configs from the pair's NARVAL protocol file; T+'s DPO reference must equal T0."""
    work = workdir()
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
        path.write_text(json.dumps(pilot_config(protocol,variant,checkpoints,_revisions(),split,verified,PAIR),indent=2),
                        encoding='utf-8')
        paths[variant] = str(path.relative_to(work))
    return paths


@app.function(**stage('diagnose'))
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


@app.function(**stage('variants',retries=1))
def train_variants(revision: str) -> None:
    """crpd_main, crpd_raw, crpd_anchor (seed 0) sharing one cache_root; each resumes from its shards."""
    volume.reload()
    configs = write_pilot_configs()
    for variant in VARIANTS:
        config = json.loads((workdir()/configs[variant]).read_text(encoding='utf-8'))
        if finished(config,'diagnostics.json'):
            print('skip',variant,flush=True)
            continue
        crpd(revision,'evaluate','--config',configs[variant])
        volume.commit()


@app.function(image=image,volumes={VOLUME_PATH:volume},cpu=0.25,memory=1024,timeout=24*3600)
def finish(revision: str) -> None:
    """Phase 2 orchestrator: diagnostic, then the three variants."""
    diagnose.remote(revision)
    train_variants.remote(revision)


@app.function(image=image,volumes={VOLUME_PATH:volume},cpu=0.25,memory=1024,timeout=24*3600)
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
        raise SystemExit('usage: CRPD_PAIR=<pair> python script/modal/narval_pair.py [start|finish]')
    function = modal.Function.from_name(APP_NAME,'run_pilot' if command == 'start' else 'finish')
    call = function.spawn(local_revision())
    print(f'spawned {command} for {PAIR} on Modal app {APP_NAME}: {call.object_id}')
    print(f'logs: modal app logs {APP_NAME}    files: modal volume ls {APP_NAME} output/narval/{PAIR}/pilot')
