"""Modal launcher: ChunkRatio-PD pilot, Qwen3-1.7B teacher pair -> Qwen3-0.6B student.

Stages (idempotent; outputs and Hugging Face cache live on the Modal Volume named after the app):

  phase 1 (run_pilot): prepare data (CPU) -> SFT 1.7B (A100-80GB) || SFT 0.6B (L40S) -> DPO 1.7B (A100-80GB)
  phase 2 (finish):    displacement diagnostic, 100 prompts (L40S) -> crpd_main, crpd_raw, crpd_anchor (A100-80GB)

Each phase is a separate function call so each stays inside Modal's 24 h limit. GPUs are attached
only to the stages that need them; orchestrators use a quarter CPU. A finished stage writes a
marker and is skipped on re-run; CRPD stages resume from their own shards. Every chunkratio
command writes its wall time, CPU cores, peak RAM and GPU memory/utilization to resources/*.json.

Environment (read at deploy and at start; baked into the image):
  CRPD_SCALE=smoke  small data (256 Deita rows, 512 UltraFeedback pairs, 8/4/4 pilot prompts, crpd_main,
                    10 steps) on the same GPUs and H=512, in its own app/volume crpd-qwen3-pilot-smoke
  CRPD_HF_PUSH=1    push T0, rho, T+ and each CRPD student to public Hugging Face model repos as soon as
                    they exist; needs the Modal secret `huggingface` with a write HF_TOKEN

Usage (run from the repository root with the Python that has `modal` installed):

  modal deploy script/modal/qwen3_pilot.py
  python script/modal/qwen3_pilot.py start      # spawn phase 1 (+2) on Modal and return
  python script/modal/qwen3_pilot.py finish     # spawn phase 2 only (after phase 1 finished)
  python script/modal/qwen3_pilot.py push       # (re)push every finished artifact

Unverified settings are listed in chunkratio/pilot.py::PILOT_ASSUMPTIONS and in every config.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

SCALE = os.environ.get('CRPD_SCALE','full')
if SCALE not in ('full','smoke','stability','x10','x30'):
    raise SystemExit('CRPD_SCALE must be full, smoke, stability, x10 or x30')
# smoke: plumbing and per-stage timing; stability: enough steps and RM checkpoints to see whether
# policy training moves the right way. Both are small runs in their own app and volume.
SMOKE = SCALE == 'smoke'
SMALL = SCALE != 'full'
HF_PUSH = os.environ.get('CRPD_HF_PUSH','0') == '1'
# CRPD_TEACHERS=hub takes T0/T+ from pinned published checkpoints (chunkratio.pilot.HUB_TEACHERS),
# checks T+ = T0 + LoRA from the weights, and trains only rho (UltraChat SFT of Qwen3-0.6B-Base).
# CRPD_TEACHERS=hda takes T0, T+ and rho from the pinned NARVAL-matching published trio
# (chunkratio.pilot.HDA_CHECKPOINTS), checks that T0 is the nearest candidate to T+ in weight space,
# and trains nothing before CRPD.
TEACHERS = os.environ.get('CRPD_TEACHERS','train')
if TEACHERS not in ('train','hub','hda'):
    raise SystemExit('CRPD_TEACHERS must be train, hub or hda')
HUB = TEACHERS == 'hub'
HDA = TEACHERS == 'hda'
APP_NAME = 'crpd-qwen3-pilot'+('-hub' if HUB else '-hda' if HDA else '')+(f'-{SCALE}' if SMALL else '')
REMOTE_REPO = '/root/repo'
# Inside a container this file is /root/qwen3_pilot.py, so the repo root only exists locally.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_REPO)
VOLUME_PATH = '/vol'
PAIR = 'qwen3_1p7b_to_0p6b'
RUN_PAIR = PAIR+('_hub' if HUB else '_hda' if HDA else '')
DATA = f'data/narval/{RUN_PAIR}'
ROOT = f'output/narval/{RUN_PAIR}/pilot'
CHECKPOINTS = f'{ROOT}/checkpoints'
# Turn-fitted SFT data (Deita: 98.8% of conversations kept at 2048 tokens instead of 5.4%).
SFT_DATA = f"{DATA}/{'ultrachat' if HUB else 'deita'}_sft_fit2048.jsonl"
# The hda trio runs main against raw at every scale: whether the correction helps is its first question.
VARIANTS = (('crpd_main','crpd_raw') if HDA else ('crpd_main',) if SMALL else ('crpd_main','crpd_raw') if HUB
            else ('crpd_main','crpd_raw','crpd_anchor'))
# CRPD_VARIANTS=crpd_raw runs only the named variants, for example on an existing small-scale cache.
VARIANT_TEXT = os.environ.get('CRPD_VARIANTS','')
VARIANTS = tuple(v for v in VARIANT_TEXT.split(',') if v) or VARIANTS
# Smoke sizes keep the real horizon and chunking so per-prompt and per-pair rates extrapolate.
SMOKE_SIZES = {'deita_rows':256,'ultrachat_rows':256,'uf_train_rows':512,'uf_test_rows':64,
               'subset':{'train':8,'head_val':4,'select':4},'states_per_prompt':4,
               'steps':10,'batch_size':16,'selection_interval':10,'diagnostic_prompts':8,'diagnostic_replicates':4}
# Reduced hub pilot, fixed before any collection (DECISIONS): fewer prompts, 2 states per prompt.
# Microbatch 4: one long UltraFeedback prompt (1224 tokens against a median of 87) pads every row of its
# microbatch, and 8 rows x about 1600 positions ran an A100-80GB out of memory. The objective is unchanged.
HUB_SIZES = {'ultrachat_rows':10000,'subset':{'train':1000,'head_val':100,'select':64},'states_per_prompt':2,
             'steps':300,'batch_size':64,'selection_interval':150,'diagnostic_prompts':32,'diagnostic_replicates':8,
             'policy_microbatch_size':4}
# Stability check: 150 steps x 32 pairs over 64 prompts x 2 states (~768 pairs, ~6 passes), RM
# selection at steps 50/100/150; rho is SFT'd on 2,000 conversations so its rollouts are usable.
STABILITY_SIZES = {'deita_rows':2000,'ultrachat_rows':2000,'uf_train_rows':2048,'uf_test_rows':64,
                   'subset':{'train':64,'head_val':16,'select':16},'states_per_prompt':2,
                   'steps':150,'batch_size':32,'selection_interval':50,'diagnostic_prompts':8,'diagnostic_replicates':4,
                   'policy_microbatch_size':4,'selection_baseline':True}
# Ten times the stability collection (640 + 160 prompts). Select stays at 64 because each RM selection
# greedily generates up to 1024 tokens per prompt, one prompt at a time. Generation batch stays at 32:
# collect-candidates already peaked at 53 GB there, and the 2048-token cap bounds the padded width.
X10_SIZES = {'deita_rows':10000,'ultrachat_rows':10000,'uf_train_rows':20480,'uf_test_rows':64,
             'subset':{'train':640,'head_val':160,'select':64},'states_per_prompt':2,
             'steps':300,'batch_size':64,'selection_interval':150,'diagnostic_prompts':8,'diagnostic_replicates':4,
             'policy_microbatch_size':4,'selection_baseline':True,'max_prompt_tokens':2048}
# x30: the 2,000-prompt pilot (DECISIONS 90). Same 20,480 UltraFeedback rows as x10, so the hash-ranked splits
# agree: the 64 select prompts and the 64 test prompts are x10's, and the 2,000 train prompts contain x10's 640.
# 2 states per prompt as x10; 450 steps keep about x10's passes over the pairs (3.1x the prompts, 3x the steps),
# with RM selection every 150 steps.
X30_SIZES = {**X10_SIZES,'subset':{'train':2000,'head_val':200,'select':64},'steps':450,'selection_interval':150}
SIZES = {'smoke':SMOKE_SIZES,'stability':STABILITY_SIZES,'x10':X10_SIZES,'x30':X30_SIZES}.get(SCALE) or (HUB_SIZES if HUB else None)
# hda x10: speed-only knobs raised toward the A100-80GB. The hub x10 run peaked at 53 GB in collect-candidates
# with 32 rows (17.7 GB of it FP32 weights) and at 51 GB in train-policy with microbatch 4. Neither knob
# changes results: collection groups never affect samples, and microbatches accumulate with per-pair weights.
if HDA and SCALE in ('x10','x30'):
    SIZES = {**SIZES,'generation_batch_size':48,'policy_microbatch_size':6}
# hda: RM selection generates 16 select prompts per batch instead of one at a time (each selection took
# about 15 min of the hub x10 train-policy). Same prompts, greedy rule and 1024-token cap.
if HDA and SIZES:
    SIZES = {**SIZES,'selection_generation_batch_size':16}
# CRPD_CORRECTION_BATCH=96 gives collect-correction its own generation group size (it is generation-bound
# and used 34 GB at 48 rows in hda x10); grouping never changes a job's samples.
CORRECTION_BATCH = os.environ.get('CRPD_CORRECTION_BATCH','')
if CORRECTION_BATCH and SIZES:
    SIZES = {**SIZES,'correction_generation_batch_size':int(CORRECTION_BATCH)}
# CRPD_FAST_OPTIONS=1 turns on the chunkratio.fast execution options: candidate scoring across states
# within the token budget (FP32 rounding differences only) and a 3 GiB host cache of head features.
FAST_OPTIONS = os.environ.get('CRPD_FAST_OPTIONS','0') == '1'
if FAST_OPTIONS and SIZES:
    SIZES = {**SIZES,'batch_candidate_scoring':True,'head_feature_cache_mb':3072}
# CRPD_BIG_BATCH=1: larger grouping batches only (results change by FP32 rounding at most); the training
# batch of 64 pairs per step is unchanged. Prefix generation 96 rows (37 GB at 48 rows in hda x10, groups
# now length-sorted), policy microbatch 8 (55 GB at 6), selection generation 32 prompts, candidate
# scoring across states with a 16k-token budget.
BIG_BATCH = os.environ.get('CRPD_BIG_BATCH','0') == '1'
if BIG_BATCH and SIZES:
    SIZES = {**SIZES,'prefix_generation_batch_size':96,'policy_microbatch_size':8,'selection_generation_batch_size':32,
             'batch_candidate_scoring':True,'score_tokens_per_batch':16384}
# CRPD_STEPS=150 trains for that many steps and selects only at the end (plus the rho baseline).
STEPS = os.environ.get('CRPD_STEPS','')
if STEPS and SIZES:
    SIZES = {**SIZES,'steps':int(STEPS),'selection_interval':int(STEPS)}
# CRPD_BATCH_SIZE=128 changes the training batch (pairs per optimizer step): a training setting, not a
# speed option. GPU memory follows policy_microbatch_size, so 128 pairs run as 16 accumulated microbatches.
BATCH_SIZE = os.environ.get('CRPD_BATCH_SIZE','')
if BATCH_SIZE and SIZES:
    SIZES = {**SIZES,'batch_size':int(BATCH_SIZE)}
# CRPD_TURBO=1: every measured speed option at once. Training targets and pairs are unchanged up to FP32
# rounding; gpu_sampling draws the same categorical distributions with different random draws (exact
# sampling, tested), so a turbo run is a fresh collection, never mixed with a non-turbo cache.
#   generation groups by token budget (KV cache ~229 KB/token FP32 for both Qwen3 sizes: 160k tokens ~37 GB),
#   train-policy microbatches by token budget 8192 (4.1 s vs 7.6 s per 128-pair step at microbatch 8,
#   54.6 GB peak; 16k ran out of memory), cross-state candidate scoring, selection generation 32 prompts.
TURBO = os.environ.get('CRPD_TURBO','0') == '1'
# CRPD_ONE_PROCESS=1 (implied by CRPD_TURBO): small scales run the whole pipeline in one `evaluate` call
# instead of one process per stage, each of which reloads and fingerprints the three models.
ONE_PROCESS = os.environ.get('CRPD_ONE_PROCESS','0') == '1' or TURBO
# CRPD_NO_END_TOKEN=1: end-of-turn tokens are unscored in every chunk score (teacher delta_T and the student
# and rho margins), a user-requested deviation from 'include sampled EOS exactly once' (DECISIONS). The hda
# T+ ends turns with <|endoftext|> and writes about twice as long, so it scores the shared <|im_end|> about
# 20 nats below T0 and the turbo student learned never to end a response (0% EOS on held-out prompts).
NO_END_TOKEN = os.environ.get('CRPD_NO_END_TOKEN','0') == '1'
# CRPD_PAIR_SOURCES=within trains crpd_raw on same-source sibling pairs only (rho-rho, T+-T+), removing the
# proposal-source bias of rho-T+ labels; student_ref keeps rho-rho pairs only (DECISIONS 92). Unset: all pairs.
PAIR_SOURCES = os.environ.get('CRPD_PAIR_SOURCES','')
if PAIR_SOURCES and SIZES:
    SIZES = {**SIZES,'pair_sources':PAIR_SOURCES}
if NO_END_TOKEN and SIZES:
    SIZES = {**SIZES,'score_end_token':False}
if TURBO and SIZES:
    SIZES = {**SIZES,'gpu_sampling':True,'generation_tokens_per_batch':163840,'generation_max_rows':384,
             'policy_tokens_per_microbatch':8192,'batch_candidate_scoring':True,'score_tokens_per_batch':16384,
             'selection_generation_batch_size':32,
             # One prefix pass per state for all siblings and per pair for c and d (A100 benchmarks: T+ scoring
             # 24 s vs 58 s per 1,600 siblings at 33 GB with 48 states per forward; policy step 2.9 s vs 4.0 s).
             'shared_prefix_scoring':True,'score_max_states':48}
HF_PREFIX = 'crpd-qwen3-1p7b-to-0p6b'+('-hub' if HUB else '-hda' if HDA else '')+(f'-{SCALE}' if SMALL else '')
# GPU of the CRPD stage (collection, head, targets, policy). Another GPU writes its outputs and cache
# under a GPU tag so a comparison run collects afresh instead of reusing the default GPU's shards.
CRPD_GPU = os.environ.get('CRPD_GPU','A100-80GB')
GPU_TAG = '' if CRPD_GPU == 'A100-80GB' else '-'+CRPD_GPU.lower().replace('-','')
# CRPD_OUTPUT_TAG=v2 writes outputs and cache under a tag as well, so a rerun with new code sits beside
# the finished results of an earlier config instead of colliding with them.
OUTPUT_TAG = os.environ.get('CRPD_OUTPUT_TAG','')
# CRPD_CACHE_TAG=shared keeps the cache under its own tag while CRPD_OUTPUT_TAG names the outputs, so runs
# with new output names reuse one collection (the stage keys still check models, data, sampler, end-token
# rule and code; DECISIONS 87). Unset: the cache follows the output tag, as before.
CACHE_TAG = os.environ.get('CRPD_CACHE_TAG','')
CACHE_SUFFIX = GPU_TAG+('-'+CACHE_TAG if CACHE_TAG else '-'+OUTPUT_TAG if OUTPUT_TAG else '')
GPU_TAG += '-'+OUTPUT_TAG if OUTPUT_TAG else ''
# CRPD_COMPACT=1: the user's compact configuration (DECISIONS), tagged so it never shares outputs or cache
# with protocol runs. K=2 (one rho + one T+ sibling), correction rolled out from every nonterminal
# endpoint only, head fitted on chunk-boundary states only; selection every 50 steps, because one pair
# per state makes the pool about six times smaller.
COMPACT = os.environ.get('CRPD_COMPACT','0') == '1'
COMPACT_SETTINGS = {'candidates':2,'correction_seeds':'endpoints','endpoint_fraction':1.,'head_positions':'boundary',
                    'states_per_prompt':2,'selection_interval':50}
GPU_TAG += '-compact' if COMPACT else ''
CACHE_SUFFIX += '-compact' if COMPACT else ''
# CRPD_ALPHAS=0.03,0.1 trains each variant once per alpha instead of the protocol alpha. Collection,
# head and target keys do not contain alpha, so the sweep reuses them while the code revision is the same.
ALPHA_TEXT = os.environ.get('CRPD_ALPHAS','')
ALPHAS = tuple(float(a) for a in ALPHA_TEXT.split(',') if a)
# CRPD_SWEEP_LITE=1 trims a sweep to train-policy on the existing cache: rho is not rescored (it does not
# depend on alpha) and evaluate is skipped. Selection scores stay in the policy result and the resume file.
SWEEP_LITE = os.environ.get('CRPD_SWEEP_LITE','0') == '1'
if not all(0 < a < float('inf') for a in ALPHAS):
    raise SystemExit('CRPD_ALPHAS must be finite positive numbers')
# CRPD_GAMMAS=1,0.3,0.1 trains crpd_raw once per alpha and gamma, t = gamma * delta_T (DECISIONS). gamma
# only enters the target and policy keys, so all gammas of one launch share one collection; gamma = 1
# writes the same config (and name) as a run without the option.
GAMMA_TEXT = os.environ.get('CRPD_GAMMAS','')
GAMMAS = tuple(float(g) for g in GAMMA_TEXT.split(',') if g)
if not all(0 < g <= 1 for g in GAMMAS):
    raise SystemExit('CRPD_GAMMAS must be numbers in (0, 1]')
if GAMMAS and not ALPHAS:
    raise SystemExit('CRPD_GAMMAS needs CRPD_ALPHAS (the sweep names its runs by alpha and gamma)')
if any(g != 1 for g in GAMMAS) and set(VARIANTS) != {'crpd_raw'}:
    raise SystemExit('CRPD_GAMMAS below 1 is implemented for CRPD_VARIANTS=crpd_raw only')


def run_name(variant: str, alpha: float, gamma: float = 1.) -> str:
    """Config and output name of one sweep run, for example crpd_main-alpha0p03 or
    crpd_raw-alpha0p03-gamma0p3 (gamma = 1 keeps the name without the option)."""
    name = f"{variant}-alpha{str(alpha).replace('.','p')}"
    return name if gamma == 1 else f"{name}-gamma{str(gamma).replace('.','p')}"


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME,create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0','transformers==4.52.4','datasets==3.5.0','numpy','pyyaml','safetensors','tokenizers',
                      'huggingface_hub')
         .env({'HF_HOME':f'{VOLUME_PATH}/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
               'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
               'CRPD_SCALE':SCALE,'CRPD_HF_PUSH':'1' if HF_PUSH else '0','CRPD_TEACHERS':TEACHERS,
               'CRPD_GPU':CRPD_GPU,'CRPD_ALPHAS':ALPHA_TEXT,'CRPD_SWEEP_LITE':'1' if SWEEP_LITE else '0',
               'CRPD_VARIANTS':VARIANT_TEXT,'CRPD_OUTPUT_TAG':OUTPUT_TAG,'CRPD_CACHE_TAG':CACHE_TAG,'CRPD_COMPACT':'1' if COMPACT else '0',
               'CRPD_CORRECTION_BATCH':CORRECTION_BATCH,'CRPD_FAST_OPTIONS':'1' if FAST_OPTIONS else '0',
               'CRPD_BIG_BATCH':'1' if BIG_BATCH else '0','CRPD_STEPS':STEPS,'CRPD_BATCH_SIZE':BATCH_SIZE,
               'CRPD_TURBO':'1' if TURBO else '0','CRPD_ONE_PROCESS':'1' if ONE_PROCESS else '0',
               'CRPD_NO_END_TOKEN':'1' if NO_END_TOKEN else '0','CRPD_PAIR_SOURCES':PAIR_SOURCES,'CRPD_GAMMAS':GAMMA_TEXT})
         .add_local_dir(REPO,REMOTE_REPO,ignore=['output','data','.git','mtbench','winrate_eval','diversity_metrics',
                                                 'promt','prompts','env','reports','**/__pycache__','.pytest_cache']))
common = {'image':image,'volumes':{VOLUME_PATH:volume},'timeout':24*3600}
# Preparation stages fail deterministically, so no retries, and a stall cannot hold a GPU for 24 h.
sft_limits = {**common,'timeout':4*3600}
dpo_limits = {**common,'timeout':16*3600}
hf_secret = [modal.Secret.from_name('huggingface')] if HF_PUSH else []


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


def gpu_sampler(samples: list[tuple[float,float]], stop: threading.Event) -> None:
    """Every 2 s record (GPU memory used GiB, utilization %) of GPU 0 via nvidia-smi; silent without a GPU."""
    while not stop.wait(2.):
        try:
            line = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                                  capture_output=True,text=True,timeout=10).stdout.splitlines()[0]
        except (OSError,IndexError,subprocess.SubprocessError):
            return
        memory,utilization = (float(v) for v in line.split(','))
        samples.append((memory/1024,utilization))


def crpd(revision: str, *args: str) -> dict:
    """Run one chunkratio CLI command in the work directory and log its resource use."""
    import resource  # Unix only; the launcher itself is also imported on Windows
    work = workdir()
    print('+ python -m chunkratio',' '.join(args),flush=True)
    samples: list[tuple[float,float]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=gpu_sampler,args=(samples,stop),daemon=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    sampler.start()
    try:
        subprocess.run([sys.executable,'-m','chunkratio',*args],cwd=work,check=True,
                       env={**os.environ,'CRPD_CODE_REVISION':revision})
    finally:
        stop.set()
        sampler.join()
        wall = time.perf_counter()-start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        usage = {'command':args[0],'args':list(args[1:]),'scale':SCALE,'wall_s':round(wall,1),
                 'cpu_cores_mean':round((after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/max(wall,1e-9),2),
                 'max_rss_gb_so_far':round(after.ru_maxrss/2**20,2),
                 'gpu_mem_max_gb':round(max((m for m,_ in samples),default=0.),2),
                 'gpu_util_mean':round(sum(u for _,u in samples)/len(samples),1) if samples else None,
                 'gpu':os.environ.get('MODAL_GPU_TYPE') or None}
        print(json.dumps({'resource':usage}),flush=True)
        # One file per command: parallel containers committing one shared file overwrite each other.
        log = work/ROOT/'resources'/f'{time.time_ns()}-{os.getpid()}-{args[0]}.json'
        log.parent.mkdir(parents=True,exist_ok=True)
        log.write_text(json.dumps(usage),encoding='utf-8')
    return usage


def done(path: str) -> bool:
    """Stage marker on the volume (relative to the work directory)."""
    return (workdir()/path).exists()


def finished(config: dict, marker: str, revision: str) -> bool:
    """A CRPD run is finished only if its marker exists, it ran exactly this config, and its manifest
    records the code revision this code would record. Another config's results in the output folder
    are an error; results of other code are not finished, so the run repeats (each stage's cache key
    contains the revision, so nothing of the old code is reused)."""
    if not done(f"{config['output']}/{marker}"):
        return False
    resolved = workdir()/config['output']/'resolved_config.json'
    if not resolved.exists() or json.loads(resolved.read_text(encoding='utf-8')) != config:
        raise ValueError(f"{config['output']} holds results of a different config; use another output to rerun")
    sys.path.insert(0,str(workdir()))
    from chunkratio.artifacts import code_revision
    manifest = workdir()/config['output']/'manifest.json'
    return manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')).get('code_revision') == code_revision(revision)


def train_checkpoint(revision: str, stage: str, model: str, model_revision: str, data: str, out: str,
                     tokenizer: tuple[str,str] | None = None) -> None:
    """SFT or teacher DPO with the assumed recipe; skipped when the checkpoint manifest exists."""
    volume.reload()
    if done(f'{out}/preparation_manifest.json'):
        print('skip',out,flush=True)
        return
    sys.path.insert(0,str(workdir()))
    from chunkratio.pilot import preparation_config
    config = workdir()/f'{out}.config.json'
    config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps(preparation_config(stage,model,model_revision,data,out,PAIR,*(tokenizer or (None,None))),
                                 indent=2),encoding='utf-8')
    crpd(revision,stage,'--config',str(config.relative_to(workdir())))
    volume.commit()


def head_lines(source: Path, destination: Path, rows: int) -> None:
    """Smoke subset: the first `rows` JSONL records of a raw download."""
    with open(source,encoding='utf-8') as handle, open(destination,'w',encoding='utf-8') as out:
        for i,line in enumerate(handle):
            if i >= rows:
                break
            out.write(line)


# CPU and RAM below follow the smoke measurements (every stage used about 1 core; peak RSS in
# parentheses) with headroom; Modal bills the reserved amount.
@app.function(cpu=2.0,memory=12288,**common)  # (3.7 GB smoke)
def prepare(revision: str) -> None:
    """Pin revisions, download Deita and UltraFeedback, build canonical data and the pilot subset (CPU)."""
    volume.reload()
    work = workdir()
    sys.path.insert(0,str(work))
    from chunkratio.artifacts import read_jsonl, write_json
    from chunkratio.pilot import (HDA_CHECKPOINTS, HUB_TEACHERS, MAX_LENGTH, PREFERENCE_DATASET, REWARD_MODEL,
                                  SFT_DATASET, STUDENT_BASE_MODEL, TEACHER_BASE_MODEL, pilot_subset)
    # The hda trio needs no SFT data: every role is a published checkpoint.
    if done(f'{DATA}/prepared.json') and (HDA or done(SFT_DATA)):
        print('skip prepare',flush=True)
        return
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    # hda: the SFT data its model card names, pinned for provenance only (nothing is trained or downloaded).
    sft_dataset = 'HuggingFaceH4/ultrachat_200k' if HUB else 'HuggingFaceH4/deita-10k-v0-sft' if HDA else SFT_DATASET
    revisions_path = work/DATA/'revisions.json'
    if revisions_path.exists():
        revisions = json.loads(revisions_path.read_text(encoding='utf-8'))
    else:
        revisions = {'student_base_model':api.model_info(STUDENT_BASE_MODEL).sha,
                     'reward_model':api.model_info(REWARD_MODEL).sha,
                     'sft_dataset':api.dataset_info(sft_dataset).sha,
                     'preference_dataset':api.dataset_info(PREFERENCE_DATASET).sha}
        if HDA:
            # Pinned trio; the Hub must still serve exactly these commits. Base checkpoints are the
            # alternatives of the weight-space reference check.
            for source in HDA_CHECKPOINTS.values():
                if api.model_info(source['repo'],revision=source['revision']).sha != source['revision']:
                    raise ValueError(f"{source['repo']} does not serve revision {source['revision']}")
            revisions.update(teacher_base_model=HDA_CHECKPOINTS['teacher_base']['revision'],
                             teacher_plus_model=HDA_CHECKPOINTS['teacher_plus']['revision'],
                             student_base_model=HDA_CHECKPOINTS['student_ref']['revision'],
                             tokenizer=HDA_CHECKPOINTS['teacher_base']['revision'],
                             qwen3_1p7b_base=api.model_info(TEACHER_BASE_MODEL).sha,
                             qwen3_1p7b=api.model_info('Qwen/Qwen3-1.7B').sha)
        elif HUB:
            # Pinned teacher revisions; the Hub must still serve exactly these commits.
            for role,source in HUB_TEACHERS.items():
                if api.model_info(source['repo'],revision=source['revision']).sha != source['revision']:
                    raise ValueError(f"{source['repo']} does not serve revision {source['revision']}")
            revisions.update(teacher_base_model=HUB_TEACHERS['teacher_base']['revision'],
                             teacher_plus_model=HUB_TEACHERS['teacher_plus']['revision'],
                             dpo_lora=HUB_TEACHERS['dpo_lora']['revision'],
                             tokenizer=HUB_TEACHERS['teacher_base']['revision'])
        else:
            revisions['teacher_base_model'] = api.model_info(TEACHER_BASE_MODEL).sha
        write_json(revisions_path,revisions)
    raw = work/DATA/'raw'
    raw.mkdir(parents=True,exist_ok=True)
    sft_raw = 'ultrachat.jsonl' if HUB else 'deita.jsonl'
    for name,dataset,split,key in ((() if HDA else ((sft_raw,sft_dataset,'train_sft' if HUB else 'train','sft_dataset'),))+
                                   (('uf_train_prefs.jsonl',PREFERENCE_DATASET,'train_prefs','preference_dataset'),
                                    ('uf_test_prefs.jsonl',PREFERENCE_DATASET,'test_prefs','preference_dataset'))):
        if not (raw/name).exists():
            load_dataset(dataset,split=split,revision=revisions[key]).to_json(str(raw/name),lines=True,force_ascii=False)
    if HUB and not (raw/'ultrachat_sharegpt.jsonl').exists():
        # UltraChat stores role/content messages; prepare-data's ShareGPT reader takes from/value turns.
        roles = {'user':'human','assistant':'gpt','system':'system'}
        with open(raw/sft_raw,encoding='utf-8') as source, open(raw/'ultrachat_sharegpt.jsonl','w',encoding='utf-8') as out:
            for i,line in enumerate(source):
                if i >= SIZES['ultrachat_rows']:
                    break
                row = json.loads(line)
                out.write(json.dumps({'id':row['prompt_id'],'conversations':[{'from':roles[m['role']],'value':m['content']}
                                                                            for m in row['messages']]},ensure_ascii=False)+'\n')
    sources = {'deita':'ultrachat_sharegpt.jsonl' if HUB else 'deita.jsonl','uf_train':'uf_train_prefs.jsonl',
               'uf_test':'uf_test_prefs.jsonl'}
    if HDA:
        del sources['deita']
    if SMALL:
        for key,name in sources.items():
            head_lines(raw/name,raw/f'{SCALE}_{name}',SIZES[f'{key}_rows'])
        sources = {key:f'{SCALE}_{name}' for key,name in sources.items()}
    template = json.dumps({'enable_thinking':False})
    # Hub teachers: their own tokenizer and plain ChatML template format every role's data.
    tokenizer = ['--tokenizer',HUB_TEACHERS['teacher_base']['repo'] if HUB else HDA_CHECKPOINTS['teacher_base']['repo']
                 if HDA else STUDENT_BASE_MODEL,
                 '--revision',revisions['tokenizer'] if HUB or HDA else revisions['student_base_model'],
                 '--eos-token','<|im_end|>','--chat-template-kwargs',template]
    # Each output is built once; a later code version can add a file without redoing the others.
    if not HDA and not done(SFT_DATA):
        crpd(revision,'prepare-data','--input',f"{DATA}/raw/{sources['deita']}",'--output',SFT_DATA,
             '--format','sharegpt','--assign-split','train','--fit-max-length',str(MAX_LENGTH),*tokenizer)
    if not done(f'{DATA}/ultrafeedback_train.jsonl'):
        crpd(revision,'prepare-data','--input',f"{DATA}/raw/{sources['uf_train']}",'--output',f'{DATA}/ultrafeedback_train.jsonl',
             '--format','ultrafeedback',*tokenizer)
    if not done(f'{DATA}/ultrafeedback_test.jsonl'):
        crpd(revision,'prepare-data','--input',f"{DATA}/raw/{sources['uf_test']}",'--output',f'{DATA}/ultrafeedback_test.jsonl',
             '--format','ultrafeedback','--assign-split','test',*tokenizer)
    subset = pilot_subset(read_jsonl(work/DATA/'ultrafeedback_train.jsonl'),sizes=SIZES['subset'] if SIZES else None)
    (work/DATA/'ultrafeedback_train_pilot.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),
                                                             encoding='utf-8')
    write_json(work/DATA/'prepared.json',{'revisions':revisions,'pilot_rows':len(subset),'code_revision':revision,
                                          'sft_data':SFT_DATA,'scale':SCALE})
    volume.commit()


def _revisions() -> dict:
    """Revisions pinned by prepare()."""
    return json.loads((workdir()/DATA/'revisions.json').read_text(encoding='utf-8'))


@app.function(gpu='A100-80GB',cpu=2.0,memory=24576,retries=0,**sft_limits)  # (13.3 GB; GPU 44.6/80)
def sft_teacher(revision: str) -> None:
    """T0: SFT of Qwen3-1.7B-Base on Deita (full fine-tuning, FP32 weights, bf16 autocast)."""
    volume.reload()
    train_checkpoint(revision,'train-sft','Qwen/Qwen3-1.7B-Base',_revisions()['teacher_base_model'],
                     SFT_DATA,f'{CHECKPOINTS}/teacher_base')


@app.function(gpu='L40S',cpu=2.0,memory=12288,retries=0,**sft_limits)  # (7.4 GB; GPU 25.8/48)
def sft_student(revision: str) -> None:
    """rho: SFT of Qwen3-0.6B-Base on the same SFT data (L40S is enough for 0.6B); hub teachers
    lend their tokenizer and template so every role shares the same prompt tokens."""
    volume.reload()
    sys.path.insert(0,str(workdir()))
    from chunkratio.pilot import HUB_TEACHERS
    shared = (HUB_TEACHERS['teacher_base']['repo'],_revisions()['tokenizer']) if HUB else None
    train_checkpoint(revision,'train-sft','Qwen/Qwen3-0.6B-Base',_revisions()['student_base_model'],
                     SFT_DATA,f'{CHECKPOINTS}/student_ref',shared)


@app.function(cpu=4.0,memory=32768,timeout=2*3600,**{k:v for k,v in common.items() if k != 'timeout'})
def verify_teachers(revision: str) -> dict:
    """Hub teachers: download the pinned T0, T+ and DPO LoRA and check T+ = T0 + LoRA (CPU)."""
    volume.reload()
    out = workdir()/ROOT/'teacher_verification.json'
    # Only a passed check is a completion marker; a failed report is kept beside it for inspection.
    if out.exists() and json.loads(out.read_text(encoding='utf-8'))['verified']:
        print('skip verify_teachers',flush=True)
        return json.loads(out.read_text(encoding='utf-8'))
    sys.path.insert(0,str(workdir()))
    from huggingface_hub import snapshot_download
    from chunkratio.artifacts import write_json
    from chunkratio.hub_teachers import verify_lora_merge
    from chunkratio.pilot import HUB_TEACHERS
    folders = {role:Path(snapshot_download(source['repo'],revision=source['revision']))
               for role,source in HUB_TEACHERS.items()}
    report = {**verify_lora_merge(folders['teacher_base'],folders['teacher_plus'],folders['dpo_lora']),
              'sources':HUB_TEACHERS,'code_revision':revision}
    print(json.dumps({'teacher_verification':{k:v for k,v in report.items() if k != 'sources'}}),flush=True)
    if not report['verified']:
        write_json(out.with_name('teacher_verification.failed.json'),report)
        volume.commit()
        raise ValueError('T+ is not T0 + the published LoRA; refusing to use the hub teacher pair')
    write_json(out,report)
    volume.commit()
    return report


@app.function(gpu='A100-80GB',cpu=2.0,memory=24576,retries=0,**dpo_limits)  # (10.2 GB; GPU 57.5/80)
def dpo_teacher(revision: str) -> None:
    """T+: DPO of T0 on UltraFeedback train_prefs; the frozen DPO reference is exactly T0."""
    volume.reload()
    train_checkpoint(revision,'train-teacher-dpo',f'{CHECKPOINTS}/teacher_base','local',
                     f'{DATA}/ultrafeedback_train.jsonl',f'{CHECKPOINTS}/teacher_plus')


def write_pilot_configs() -> dict[str,str]:
    """Resolved pilot configs from the NARVAL protocol file; T+'s DPO reference must equal T0."""
    work = workdir()
    sys.path.insert(0,str(work))
    from chunkratio.pilot import HDA_CHECKPOINTS, HUB_TEACHERS, pilot_config
    assumption = None
    if HDA:
        # Every role loads from its pinned commit. The weight check supports T0 as T+'s DPO reference but
        # does not verify it, so the configs carry it as an explicit assumption with its evidence.
        evidence = json.loads((work/ROOT/'teacher_reference_evidence.json').read_text(encoding='utf-8'))
        verified = False
        assumption = {'status':'assumed','claim':"T0 is T+'s actual DPO reference",
                      'evidence':{'relative_l2_distance':evidence['relative_l2_distance'],'rule':evidence['rule']},
                      'not_verified_because':'T+ publishes no adapter, reference model name or training config'}
        checkpoints = {role:{'path':source['repo'],'revision':source['revision']} for role,source in HDA_CHECKPOINTS.items()}
    elif HUB:
        # Hub teachers load from their pinned commits; verification is the weight check above.
        verified = json.loads((work/ROOT/'teacher_verification.json').read_text(encoding='utf-8'))['verified']
        checkpoints = {role:{'path':HUB_TEACHERS[role]['repo'],'revision':HUB_TEACHERS[role]['revision']}
                       for role in ('teacher_base','teacher_plus')}
    else:
        manifests = {role:json.loads((work/CHECKPOINTS/role/'preparation_manifest.json').read_text(encoding='utf-8'))
                     for role in ('teacher_base','teacher_plus')}
        verified = manifests['teacher_plus']['teacher_dpo_reference_fingerprint'] == manifests['teacher_base']['model_fingerprint']
        checkpoints = {role:{'path':f'{CHECKPOINTS}/{role}','revision':manifests[role]['model_fingerprint']} for role in manifests}
    if not HDA:
        rho = json.loads((work/CHECKPOINTS/'student_ref'/'preparation_manifest.json').read_text(encoding='utf-8'))
        checkpoints['student_ref'] = {'path':f'{CHECKPOINTS}/student_ref','revision':rho['model_fingerprint']}
    split = json.loads((work/DATA/'ultrafeedback_train.splits.json').read_text(encoding='utf-8'))['assignment_sha256']
    protocol = json.loads((work/'config/crpd/narval'/f'{PAIR}.json').read_text(encoding='utf-8'))
    paths = {}
    for variant in VARIANTS:
        path = work/f'{ROOT}/configs/{variant}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        cfg = pilot_config(protocol,variant,checkpoints,_revisions(),split,verified,RUN_PAIR,assumption)
        if SIZES:
            cfg.update({k:SIZES[k] for k in ('steps','batch_size','selection_interval','states_per_prompt',
                                             'policy_microbatch_size','selection_baseline','max_prompt_tokens',
                                             'generation_batch_size','selection_generation_batch_size',
                                             'correction_generation_batch_size','batch_candidate_scoring',
                                             'head_feature_cache_mb','prefix_generation_batch_size',
                                             'score_tokens_per_batch','gpu_sampling','generation_tokens_per_batch',
                                             'generation_max_rows','policy_tokens_per_microbatch',
                                             'shared_prefix_scoring','score_max_states','score_end_token','pair_sources') if k in SIZES})
            cfg['pilot_assumptions'] = {**cfg['pilot_assumptions'],'pilot_subset':f"{SIZES['subset']} prompts ({SCALE} scale)",
                                        'states_per_prompt':str(SIZES['states_per_prompt']),
                                        'policy_steps':f"{SIZES['steps']} steps x {SIZES['batch_size']} pairs; RM selection every {SIZES['selection_interval']} steps"}
        if COMPACT:
            cfg.update(COMPACT_SETTINGS)
            cfg['pilot_assumptions'] = {**cfg['pilot_assumptions'],
                                        'compact':('K=2 (1 rho + 1 T+), correction from nonterminal endpoints only '
                                                   '(endpoint_fraction 1), head on chunk-boundary states only, '
                                                   '2 states per prompt, selection every 50 steps; a user-requested '
                                                   'deviation from the K=4 protocol (DECISIONS)')}
        if GPU_TAG or CACHE_SUFFIX:
            cfg.update(output=f'{ROOT}/{variant}{GPU_TAG}/seed0',cache_root=f'{ROOT}/cache{CACHE_SUFFIX}')
        # Turbo raw/suffix runs (no head): collect the train split only and skip the duplicate student.pt
        # (the HF export holds the same weights). Train pairs and targets are unchanged (DECISIONS 87).
        if TURBO and variant in ('crpd_raw','crpd_suffix'):
            cfg.update(collection_splits=['train'],save_student_state_dict=False)
        if SWEEP_LITE:
            cfg['selection_baseline'] = False
        path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        paths[variant] = str(path.relative_to(work))
        for alpha in ALPHAS:
            for gamma in GAMMAS or (1.,):
                name = run_name(variant,alpha,gamma)
                sweep = {**cfg,'alpha':alpha,'output':f'{ROOT}/{name}{GPU_TAG}/seed0','selection_baseline':not SWEEP_LITE,
                         'pilot_assumptions':{**cfg['pilot_assumptions'],
                                              'alpha_sweep':f'{ALPHA_TEXT}; chosen on select-split RM only (DECISIONS 55)'}}
                if gamma != 1:
                    sweep['teacher_gap_scale'] = gamma
                    sweep['pilot_assumptions'] = {**sweep['pilot_assumptions'],
                                                  'gamma_sweep':f'{GAMMA_TEXT}; t = gamma * delta_T, crpd_raw only; '
                                                                'chosen on select-split RM only (DECISIONS)'}
                sweep_path = path.with_name(f'{name}.json')
                sweep_path.write_text(json.dumps(sweep,indent=2),encoding='utf-8')
                paths[name] = str(sweep_path.relative_to(work))
    return paths


@app.function(cpu=4.0,memory=49152,timeout=2*3600,**{k:v for k,v in common.items() if k != 'timeout'})
def verify_hda_teachers(revision: str) -> dict:
    """hda trio: T+ has no published LoRA, so check from the weights that T0 is nearer to T+ than the
    Base checkpoint it was trained from and the official Qwen3-1.7B (CPU). This supports, but does not
    verify, that T0 was T+'s DPO reference: runs record it as an assumption, never as verified."""
    volume.reload()
    out = workdir()/ROOT/'teacher_reference_evidence.json'
    sys.path.insert(0,str(workdir()))
    from chunkratio.pilot import HDA_CHECKPOINTS, TEACHER_BASE_MODEL
    # Earlier evidence counts only for exactly these checkpoints (repos and revisions): replacing a teacher
    # in HDA_CHECKPOINTS must check the new pair instead of reusing the old pair's result.
    if out.exists():
        earlier = json.loads(out.read_text(encoding='utf-8'))
        if earlier.get('supported') and earlier.get('sources') == HDA_CHECKPOINTS:
            print('skip verify_hda_teachers',flush=True)
            return earlier
        print('teacher_reference_evidence.json is for other checkpoints; checking again',flush=True)
    from huggingface_hub import snapshot_download
    from chunkratio.artifacts import write_json
    from chunkratio.hub_teachers import nearest_reference
    revisions = _revisions()
    folder = lambda repo,rev: Path(snapshot_download(repo,revision=rev,allow_patterns=['*.json','*.safetensors']))
    report = nearest_reference(folder(HDA_CHECKPOINTS['teacher_plus']['repo'],HDA_CHECKPOINTS['teacher_plus']['revision']),
                               folder(HDA_CHECKPOINTS['teacher_base']['repo'],HDA_CHECKPOINTS['teacher_base']['revision']),
                               {'qwen3_1p7b_base':folder(TEACHER_BASE_MODEL,revisions['qwen3_1p7b_base']),
                                'qwen3_1p7b':folder('Qwen/Qwen3-1.7B',revisions['qwen3_1p7b'])})
    report = {**report,'supported':report['nearest'],'verified':False,'sources':HDA_CHECKPOINTS,'code_revision':revision}
    print(json.dumps({'teacher_reference_evidence':{k:v for k,v in report.items() if k != 'sources'}}),flush=True)
    if not report['supported']:
        write_json(out.with_name('teacher_reference_evidence.failed.json'),report)
        volume.commit()
        raise ValueError('T0 is not the nearest candidate to T+; refusing to use the hda teacher pair')
    write_json(out,report)
    volume.commit()
    return report


@app.function(gpu='L40S',cpu=2.0,memory=32768,**common)  # (19.5 GB)
def diagnose(revision: str) -> None:
    """Displacement diagnostic on 100 pilot train prompts before any student training."""
    volume.reload()
    # Output of the batched-sampling diagnostic; an earlier per-state run's directory is left as is.
    out = f'{ROOT}/diagnostic-batched'
    if done(f'{out}/displacement.json'):
        print('skip diagnose',flush=True)
        return
    configs = write_pilot_configs()
    prompts,replicates = ((SIZES['diagnostic_prompts'],SIZES['diagnostic_replicates']) if SIZES else (100,16))
    crpd(revision,'diagnose-displacement','--config',configs[VARIANTS[0]],'--prompts',str(prompts),
         '--replicates',str(replicates),'--output',out)
    volume.commit()


# 64 GB RAM: the FP32 8B selection RM passes through host memory while loading.
# 96 GB RAM: the FP32 8B selection RM waits in host memory during training (hda x10 peaked at 62 GB).
@app.function(gpu=CRPD_GPU,cpu=4.0,memory=98304,retries=1,**common)
def train_variants(revision: str) -> None:
    """crpd_main, crpd_raw, crpd_anchor (seed 0) sharing one cache_root; each resumes from its shards.

    Smoke runs every stage as its own command so resources.jsonl times collection, head, targets
    and training separately (each command also reloads the models; validate-config times that).
    """
    volume.reload()
    configs = write_pilot_configs()
    # gamma = 1 first: it collects, and the other gammas of the same alpha reuse that collection.
    runs = ([(variant,run_name(variant,alpha,gamma)) for variant in VARIANTS for alpha in ALPHAS
             for gamma in sorted(GAMMAS or (1.,),reverse=True)] or [(v,v) for v in VARIANTS])
    for variant,name in runs:
        config = json.loads((workdir()/configs[name]).read_text(encoding='utf-8'))
        if finished(config,'diagnostics.json',revision) or (SWEEP_LITE and finished(config,'student/config.json',revision)):
            print('skip',name,flush=True)
        else:
            stages = (('validate-config','collect-prefixes','collect-candidates','collect-correction','fit-head',
                       'build-targets','train-policy','evaluate') if SMALL else ('evaluate',))
            # Variants that bypass the head (pipeline.py) would only load three models in these two stages.
            if config['variant'] in ('crpd_raw','crpd_suffix'):
                stages = tuple(s for s in stages if s not in ('collect-correction','fit-head'))
            # One pipeline call loads the models once; per-phase collection seconds stay in timing_*.json.
            if ONE_PROCESS:
                stages = ('validate-config','evaluate')
            stages = ('train-policy',) if SWEEP_LITE else stages
            for stage in stages:
                crpd(revision,stage,'--config',configs[name])
                volume.commit()
        if HF_PUSH:
            push_hf.spawn(revision,f"{config['output']}/student",name,config['output'])


@app.function(cpu=2.0,memory=8192,timeout=4*3600,retries=2,secrets=hf_secret,image=image,volumes={VOLUME_PATH:volume})
def push_hf(revision: str, folder: str, role: str, extras: str | None = None) -> str:
    """Upload one finished model folder to a public Hugging Face repo <user>/<prefix>-<role>, once."""
    volume.reload()
    work = workdir()
    marker = work/ROOT/'hf_pushed'/f'{role}.json'
    if marker.exists():
        print('skip push',role,flush=True)
        return json.loads(marker.read_text(encoding='utf-8'))['repo']
    from huggingface_hub import HfApi
    api = HfApi()
    repo = f"{os.environ.get('HF_NAMESPACE') or api.whoami()['name']}/{HF_PREFIX}-{role.replace('_','-')}"
    source = work/folder
    if not (source/'config.json').exists():
        raise FileNotFoundError(f'{source} is not a saved Hugging Face model')
    api.create_repo(repo,repo_type='model',private=False,exist_ok=True)
    prepared = json.loads((work/DATA/'prepared.json').read_text(encoding='utf-8'))
    card = '\n'.join([
        '---','license: apache-2.0','library_name: transformers',
        f"base_model: {'Qwen/Qwen3-0.6B-Base' if role in ('student_ref',)+VARIANTS else 'Qwen/Qwen3-1.7B-Base'}",
        'datasets:',f"- {'HuggingFaceH4/ultrachat_200k' if HUB else 'hkust-nlp/deita-10k-v0'}",
        '- HuggingFaceH4/ultrafeedback_binarized',
        'tags:','- chunkratio-pd','- preference-distillation','---','',
        f'# {HF_PREFIX}-{role}','',
        f"ChunkRatio-PD {f'{SCALE.upper()} TEST (small data, not a usable model)' if SMALL else 'pilot'} artifact, role `{role}`, "
        'pair Qwen3-1.7B -> Qwen3-0.6B. Pilot settings are assumptions, not NARVAL\'s; see `pilot_assumptions` '
        'in the bundled config. No benchmark result is claimed.','',
        f'- code revision: `{revision}`',f"- data revisions: `{json.dumps(prepared['revisions'])}`",
        f'- scale: `{SCALE}`; EOS `<|im_end|>`; chat template with `enable_thinking=false`',''])
    (source/'README.md').write_text(card,encoding='utf-8')
    api.upload_folder(repo_id=repo,folder_path=str(source),commit_message=f'{role} from {revision}',
                      ignore_patterns=['*.config.json'])
    if extras:
        for name in ('diagnostics.json','resolved_config.json','manifest.json'):
            if (work/extras/name).exists():
                api.upload_file(path_or_fileobj=str(work/extras/name),path_in_repo=f'crpd/{name}',repo_id=repo)
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({'repo':repo,'role':role,'revision':revision}),encoding='utf-8')
    volume.commit()
    print('pushed',role,'->',f'https://huggingface.co/{repo}',flush=True)
    return repo


def push_finished(revision: str) -> list:
    """Spawn pushes for every finished checkpoint and student (already-pushed ones are skipped)."""
    calls = []
    # Smoke pushes only the small 0.6B models (the 7 GB smoke teachers use the same upload path);
    # hub teachers are already public.
    for role in (() if HDA else ('student_ref',) if SMALL or HUB else ('teacher_base','student_ref','teacher_plus')):
        if done(f'{CHECKPOINTS}/{role}/preparation_manifest.json'):
            calls.append(push_hf.spawn(revision,f'{CHECKPOINTS}/{role}',role))
    for variant in VARIANTS:
        output = f'{ROOT}/{variant}/seed0'
        if done(f'{output}/diagnostics.json') and done(f'{output}/student/config.json'):
            calls.append(push_hf.spawn(revision,f'{output}/student',variant,output))
    return calls


@app.function(cpu=0.25,memory=1024,**common)
def finish(revision: str) -> None:
    """Phase 2 orchestrator: diagnostic, then the three variants."""
    diagnose.remote(revision)
    train_variants.remote(revision)


@app.function(cpu=0.25,memory=1024,**common)
def run_pilot(revision: str) -> None:
    """Phase 1 orchestrator; spawns phase 2 so each phase has its own 24 h budget."""
    prepare.remote(revision)
    # Hub: verify the published teachers (CPU) while rho trains; no teacher SFT or DPO.
    calls = ([verify_hda_teachers.spawn(revision)] if HDA else
             [verify_teachers.spawn(revision),sft_student.spawn(revision)] if HUB else
             [sft_teacher.spawn(revision),sft_student.spawn(revision)])
    for call in calls:
        call.get()
    if not HUB and not HDA:
        dpo_teacher.remote(revision)
    if HF_PUSH:
        volume.reload()
        push_finished(revision)
    finish.spawn(revision)


@app.function(cpu=0.25,memory=1024,**common)
def push_all(revision: str) -> None:
    """Re-push every finished artifact; already-pushed roles are skipped by their markers."""
    volume.reload()
    for call in push_finished(revision):
        print(call.get(),flush=True)


def local_revision() -> str:
    """Git HEAD of the uploaded code, marked dirty when the work tree has changes."""
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    return head+('-dirty' if dirty else '')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'start'
    functions = {'start':'run_pilot','finish':'finish','push':'push_all'}
    if command not in functions:
        raise SystemExit('usage: python script/modal/qwen3_pilot.py [start|finish|push]')
    if command == 'push' and not HF_PUSH:
        raise SystemExit('set CRPD_HF_PUSH=1 (and deploy with it) to push to Hugging Face')
    function = modal.Function.from_name(APP_NAME,functions[command])
    call = function.spawn(local_revision())
    print(f'spawned {command} ({SCALE}) on Modal app {APP_NAME}: {call.object_id}')
    print(f'logs: modal app logs {APP_NAME}    files: modal volume ls {APP_NAME} {ROOT}')
