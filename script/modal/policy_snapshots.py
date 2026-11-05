"""Modal: retrain a finished hda x10 variant's policy and save the student every N steps.

For benchmark curves over training. The run reuses the variant's prefix, candidate, correction, head and
target caches: the chunkratio sources must be unchanged since that run, and CRPD_CODE_REVISION repeats
the commit it recorded, so only the policy stage trains. Snapshots are saved by wrapping the trainer's
selection call from here, outside chunkratio, so the source hash (and every cache key) stays the same.
Selection still runs at every snapshot, which gives the select-split RM score at each step too.

Policy sampling is seeded and selection restores the torch RNG, so the trajectory repeats the original
run's up to nondeterministic GPU kernels; its step-300 snapshot should match the exported student.

Usage (repository root):

  modal run script/modal/policy_snapshots.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import modal

APP_NAME = 'crpd-qwen3-pilot-hda-x10-snapshots'
ROOT = 'output/narval/qwen3_1p7b_to_0p6b_hda/pilot'
VARIANTS = tuple(v for v in os.environ.get('CRPD_SNAPSHOT_VARIANTS','crpd_raw-alpha0p03-v2,crpd_main-alpha0p03-v2').split(',') if v)
INTERVAL = int(os.environ.get('CRPD_SNAPSHOT_INTERVAL','50'))
REMOTE_REPO = '/root/repo'
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_REPO)
VOLUME_PATH = '/vol'
TOKENIZER_FILES = ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt',
                   'added_tokens.json','chat_template.jinja')

app = modal.App(APP_NAME)
volume = modal.Volume.from_name('crpd-qwen3-pilot-hda-x10')
# Same packages as script/modal/qwen3_pilot.py, so scores and cache keys match the original run.
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0','transformers==4.52.4','datasets==3.5.0','numpy','pyyaml','safetensors','tokenizers',
                      'huggingface_hub')
         .env({'HF_HOME':f'{VOLUME_PATH}/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
               'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','CRPD_SNAPSHOT_INTERVAL':str(INTERVAL)})
         .add_local_dir(REPO,REMOTE_REPO,ignore=['output','data','.git','mtbench','winrate_eval','diversity_metrics',
                                                 'promt','prompts','env','reports','**/__pycache__','.pytest_cache']))


def workdir() -> Path:
    """Read-only code plus volume-backed output/ and data/, as in the pilot launcher."""
    work = Path('/root/work')
    work.mkdir(exist_ok=True)
    for name in ('chunkratio','config','reference','script'):
        if not (work/name).exists():
            (work/name).symlink_to(Path(REMOTE_REPO)/name)
    for name in ('output','data'):
        if not (work/name).exists():
            (work/name).symlink_to(Path(VOLUME_PATH)/name)
    return work


@app.function(image=image,gpu='A100-80GB',cpu=4.0,memory=65536,timeout=4*3600,retries=1,volumes={VOLUME_PATH:volume})
def snapshots(variant: str, commit: str) -> dict:
    """Train `variant`'s policy again under <variant>-snapshots, saving the student at every selection."""
    volume.reload()
    work = workdir()
    os.chdir(work)
    sys.path.insert(0,str(work))
    os.environ['CRPD_CODE_REVISION'] = commit
    from chunkratio import training
    from chunkratio.artifacts import code_revision, write_json
    from chunkratio.pipeline import pipeline
    source = work/ROOT/variant/'seed0'
    recorded = json.loads((source/'manifest.json').read_text(encoding='utf-8'))['code_revision']
    if code_revision() != recorded:
        raise ValueError(f'code revision {code_revision()} differs from the run ({recorded}); caches would not be reused')
    cfg = json.loads((source/'resolved_config.json').read_text(encoding='utf-8'))
    output = f'{ROOT}/{variant}-snapshots/seed0'
    cfg.update(output=output,selection_interval=INTERVAL,selection_baseline=False)
    original = training.ChunkRatioTrainer.select

    def select(self: training.ChunkRatioTrainer) -> tuple:
        # Called once per interval, after that step's optimizer update (and at the final step); the
        # history holds one scored entry per earlier selection, also after a resume.
        step = min(INTERVAL*(sum(h['selection_score'] is not None for h in self.history)+1),self.cfg['steps'])
        folder = work/output/'snapshots'/f'step{step}'
        if not (folder/'model.safetensors').exists():
            self.student.save_pretrained(folder)
            for name in TOKENIZER_FILES:
                if (source/'student'/name).exists():
                    shutil.copy(source/'student'/name,folder/name)
            volume.commit()
        return original(self)

    training.ChunkRatioTrainer.select = select
    policy = pipeline(cfg,'train-policy')
    curve = [{'step':h['step'],'selection_score':h['selection_score']} for h in policy['history'] if h['selection_score'] is not None]
    write_json(work/output/'snapshot_curve.json',{'variant':variant,'interval':INTERVAL,'curve':curve})
    volume.commit()
    return {'variant':variant,'curve':curve}


def local_commit() -> str:
    """Git HEAD of the uploaded code, marked dirty as the pilot launcher marks it."""
    import subprocess
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    return head+('-dirty' if dirty else '')


@app.local_entrypoint()
def main() -> None:
    for result in snapshots.starmap([(v,local_commit()) for v in VARIANTS]):
        print('SNAPSHOT CURVE '+json.dumps(result),flush=True)


if __name__ == '__main__':
    # Server-side calls that outlive the local client (a dropped connection cancelled `modal run` inputs):
    #   modal deploy script/modal/policy_snapshots.py && python script/modal/policy_snapshots.py
    function = modal.Function.from_name(APP_NAME,'snapshots')
    for variant in VARIANTS:
        print('spawned',variant,function.spawn(variant,local_commit()).object_id)
