"""CRPD on a multi-GPU server (2-4 GPUs, no Modal) for the three NARVAL Table 1 pairs (DECISIONS 93).

Stages, each idempotent and resumable (rerun the same command after an interruption):
  prepare     CPU: pin revisions, download UltraFeedback, tokenize with the pair's template, choose the prompt subset,
              warm the Hugging Face cache with the trio and the selection RM
  verify      CPU: weight-space evidence that T0 is nearest to T+ (recorded as an assumption, never as verified)
  collect     one worker per GPU, each collecting its shard of the prompts (collection_shard); the union equals a
              single-process collection, so the number of GPUs never changes the data
  train       one run per (gamma) config on its own GPU, all sharing the collection; RM checkpoint selection on the
              64 select prompts
  heldout     greedy generation on the 64 test prompts: RM vs rho, EOS, length, repetition (script/server/heldout_eval.py)
  benchmarks  Winogrande + TruthfulQA-MC2 vs rho and the riskKD student, with --eval-python (script/server/benchmarks.py)

Examples (repository root; Linux: script/server/run_pair.sh, Windows: script/server/run_pair.ps1):
  python script/server/run_pair.py --pair qwen3_1p7b_to_0p6b --scale x10 --gpus 0,1 --gammas 0.05 --dry-run
  python script/server/run_pair.py --pair qwen3_8b_to_1p7b --scale x10 --gpus 0,1,2,3 --gammas 0.1,0.05,0.03
  python script/server/run_pair.py --pair llama31_8b_to_llama32_1b --scale full --gpus 0,1,2,3 --gammas 0.05 \
      --eval-python ../venv-eval/bin/python
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
STAGES = ('prepare','verify','collect','train','heldout','benchmarks')
DEFAULT_STEPS = {'x10':(150,150),'x30':(450,150)}  # (steps, selection interval); full: --passes over the pairs


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}",flush=True)


class StageFailed(Exception):
    """A subprocess failed; `oom` when its log shows a CUDA out-of-memory error."""
    def __init__(self, command: list[str], status: int, logfile: Path, oom: bool) -> None:
        super().__init__(f'failed ({status}): {" ".join(command)} -- see {logfile}')
        self.oom = oom


def run(command: list[str], gpu: str | None, logfile: Path, dry: bool) -> None:
    """One subprocess on one GPU (or CPU), output appended to its log file; raises StageFailed on failure."""
    log(('DRY ' if dry else '')+(f'GPU {gpu}: ' if gpu is not None else 'CPU: ')+' '.join(command)+f'  > {logfile}')
    if dry:
        return
    logfile.parent.mkdir(parents=True,exist_ok=True)
    env = {**os.environ,'TOKENIZERS_PARALLELISM':'false','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    start = logfile.stat().st_size if logfile.exists() else 0
    with open(logfile,'a',encoding='utf-8') as out:
        status = subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
    if status:
        with open(logfile,encoding='utf-8',errors='replace') as source:
            source.seek(start)
            tail = source.read()[-20000:]
        raise StageFailed(command,status,logfile,'out of memory' in tail.lower() or 'OutOfMemoryError' in tail)


def parallel(jobs: list[tuple[list[str],Path]], gpus: list[str], per_gpu: int, dry: bool) -> None:
    """Jobs over `per_gpu` slots on each GPU. A job that runs out of GPU memory is retried alone on a GPU after
    the others (results do not depend on how many processes share a GPU); any other failure stops the run."""
    slots = [g for _ in range(per_gpu) for g in gpus]
    retry = []
    def worker(index_job: tuple[int,tuple[list[str],Path]]) -> None:
        index,(command,logfile) = index_job
        try:
            run(command,slots[index % len(slots)],logfile,dry)
        except StageFailed as error:
            if not (error.oom and per_gpu > 1):
                raise
            log(f'out of GPU memory with {per_gpu} processes per GPU; will retry alone: {logfile.name}')
            retry.append((command,logfile))
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        list(pool.map(worker,enumerate(jobs)))
    if retry:
        parallel(retry,gpus,1,dry)


def gpu_memory_gb(gpus: list[str]) -> float:
    """Smallest total memory among the chosen GPUs (nvidia-smi), in GB."""
    out = subprocess.run(['nvidia-smi','--query-gpu=index,memory.total','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True).stdout
    sizes = {line.split(',')[0].strip():float(line.split(',')[1])/1024 for line in out.strip().splitlines()}
    return min(sizes[g] for g in gpus)


def number(value: float) -> str:
    """0.05 -> 0p05 (run names)."""
    return f'{value:g}'.replace('.','p')


def main() -> int:
    from chunkratio.riskkd import RISKKD, SCALES, UF_TEST_ROWS
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pair',required=True,choices=sorted(RISKKD))
    parser.add_argument('--scale',default='x10',choices=sorted(SCALES))
    parser.add_argument('--gpus',default='0',help='GPU indices, e.g. 0,1,2,3')
    parser.add_argument('--variant',default='crpd_raw',choices=('crpd_raw','crpd_main'))
    parser.add_argument('--alpha',type=float,default=0.03)
    parser.add_argument('--gammas',default='0.05',help='Comma list; every gamma trains on the same collection')
    parser.add_argument('--pair-sources',default='all',choices=('all','within','student_ref'))
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--steps',type=int,default=None,help='Override; default 150 (x10), 450 (x30), --passes (full)')
    parser.add_argument('--selection-interval',type=int,default=None)
    parser.add_argument('--passes',type=float,default=1.0,help='full scale: passes over the training pairs')
    parser.add_argument('--selections',type=int,default=3,help='full scale: RM selections over the run')
    parser.add_argument('--gpu-memory-gb',type=float,default=None,help='Per-GPU memory for the memory plan (default: nvidia-smi)')
    parser.add_argument('--trainings-per-gpu',type=int,default=None,help='Override the memory plan')
    parser.add_argument('--policy-tokens',type=int,default=None,help='policy_tokens_per_microbatch (8192 measured on A100)')
    parser.add_argument('--generation-tokens',type=int,default=None,help='Override the planned generation_tokens_per_batch')
    parser.add_argument('--root',default='output/server')
    parser.add_argument('--stages',default=','.join(STAGES[:5]),help=f'Comma list from {STAGES}')
    parser.add_argument('--eval-python',default=None,help='Python of the lm-eval environment (benchmarks stage)')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    stages = [s for s in args.stages.split(',') if s]
    if set(stages)-set(STAGES):
        raise SystemExit(f'unknown stage in {stages}')
    if 'benchmarks' in stages and not args.eval_python:
        raise SystemExit('benchmarks needs --eval-python (an environment with requirements-eval.txt)')
    gpus = [g for g in args.gpus.split(',') if g]
    gammas = [float(g) for g in args.gammas.split(',') if g]
    if not gpus or not gammas or not all(0 < g <= 1 for g in gammas):
        raise SystemExit('need at least one GPU and gammas in (0, 1]')
    if args.variant != 'crpd_raw' and (args.pair_sources != 'all' or any(g != 1 for g in gammas)):
        raise SystemExit('gamma < 1 and --pair-sources are implemented for crpd_raw only')
    spec,scale = RISKKD[args.pair],SCALES[args.scale]
    py = sys.executable
    base = Path(args.root)/args.pair
    rows = scale['uf_train_rows']
    data = base/'data'/(f'rows{rows}' if rows else 'all')
    work = base/args.scale
    logs = work/'logs'
    revisions_path,evidence_path = base/'data'/'revisions.json',base/'teacher_reference_evidence.json'
    train_file,test_file = data/f'ultrafeedback_train_{args.scale}.jsonl',data/'ultrafeedback_test.jsonl'
    log(f'pair {args.pair}, scale {args.scale}, GPUs {gpus}, variant {args.variant}, alpha {args.alpha}, '
        f'gammas {gammas}, pair_sources {args.pair_sources}, stages {stages}')

    # prepare (CPU)
    if 'prepare' in stages and not args.dry_run and not (data/'prepared.json').exists():
        from datasets import load_dataset
        from huggingface_hub import HfApi, snapshot_download
        from chunkratio.artifacts import read_jsonl, write_json
        from chunkratio.pilot import PREFERENCE_DATASET, REWARD_MODEL, pilot_subset
        api = HfApi()
        ckpt = spec['checkpoints']
        for role,source in ckpt.items():
            if api.model_info(source['repo'],revision=source['revision']).sha != source['revision']:
                raise SystemExit(f"{source['repo']} does not serve revision {source['revision']}")
        if revisions_path.exists():
            revisions = json.loads(revisions_path.read_text(encoding='utf-8'))
        else:
            revisions = {'reward_model':api.model_info(REWARD_MODEL).sha,
                         'preference_dataset':api.dataset_info(PREFERENCE_DATASET).sha,
                         'sft_dataset':api.dataset_info('HuggingFaceH4/deita-10k-v0-sft').sha,
                         'teacher_base_model':ckpt['teacher_base']['revision'],
                         'teacher_plus_model':ckpt['teacher_plus']['revision'],
                         'student_base_model':ckpt['student_ref']['revision']}
            write_json(revisions_path,revisions)
        raw = base/'data'/'raw'
        raw.mkdir(parents=True,exist_ok=True)
        for name,split in (('uf_train_prefs.jsonl','train_prefs'),('uf_test_prefs.jsonl','test_prefs')):
            if not (raw/name).exists():
                log(f'download {PREFERENCE_DATASET} {split}')
                load_dataset(PREFERENCE_DATASET,split=split,revision=revisions['preference_dataset']).to_json(
                    str(raw/name),lines=True,force_ascii=False)
        data.mkdir(parents=True,exist_ok=True)
        def head(source: Path, target: Path, count: int | None) -> Path:
            """The first `count` rows (all when None), as the Modal pilot scales take them."""
            if count is None:
                return source
            if not target.exists():
                with open(source,encoding='utf-8') as src, open(target,'w',encoding='utf-8') as out:
                    for i,line in enumerate(src):
                        if i >= count:
                            break
                        out.write(line)
            return target
        train_raw = head(raw/'uf_train_prefs.jsonl',data/'uf_train_prefs.jsonl',rows)
        test_raw = head(raw/'uf_test_prefs.jsonl',data/'uf_test_prefs.jsonl',UF_TEST_ROWS)
        tokenizer = ['--tokenizer',ckpt['teacher_base']['repo'],'--revision',ckpt['teacher_base']['revision'],
                     '--eos-token',spec['eos'],'--chat-template-kwargs',json.dumps(spec['chat_template_kwargs'])]
        if not (data/'ultrafeedback_train.jsonl').exists():
            run([py,'-m','chunkratio','prepare-data','--input',str(train_raw),'--output',str(data/'ultrafeedback_train.jsonl'),
                 '--format','ultrafeedback',*tokenizer],None,logs/'prepare.log',False)
        if not test_file.exists():
            run([py,'-m','chunkratio','prepare-data','--input',str(test_raw),'--output',str(test_file),
                 '--format','ultrafeedback','--assign-split','test',*tokenizer],None,logs/'prepare.log',False)
        all_rows = read_jsonl(data/'ultrafeedback_train.jsonl')
        if scale['subset']['train'] is None:
            # Full: every eligible train and head_val prompt, the same 64 hash-ranked select prompts as the pilots.
            subset = sorted([r for r in all_rows if r['split'] in ('train','head_val')]+
                            pilot_subset(all_rows,sizes={'select':64}),key=lambda r:r['id'])
        else:
            subset = pilot_subset(all_rows,sizes=scale['subset'])
        train_file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),encoding='utf-8')
        log(f'subset {args.scale}: {len(subset)} prompts')
        log('warm the Hugging Face cache (trio and selection RM)')
        for source in ckpt.values():
            snapshot_download(source['repo'],revision=source['revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model'])
        snapshot_download(REWARD_MODEL,revision=revisions['reward_model'],allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
        write_json(data/'prepared.json',{'pair':args.pair,'scale':args.scale,'rows':len(subset),'revisions':revisions})

    # verify (CPU)
    if 'verify' in stages and not args.dry_run:
        from chunkratio.artifacts import write_json
        sources = {role:spec['checkpoints'][role] for role in ('teacher_base','teacher_plus')}
        earlier = json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.exists() else {}
        if earlier.get('sources') != sources:
            from huggingface_hub import snapshot_download
            from chunkratio.hub_teachers import nearest_reference
            folder = lambda repo,rev=None: Path(snapshot_download(repo,revision=rev,allow_patterns=['*.json','*.safetensors']))
            alternatives = {}
            for repo in spec['reference_alternatives']:
                try:
                    alternatives[repo] = folder(repo)
                except Exception as error:  # gated or unreachable: recorded, the check uses the others
                    log(f'reference alternative {repo} unavailable: {type(error).__name__}')
            if not alternatives:
                raise SystemExit('no reference alternative could be downloaded (Llama: accept the licence, set HF_TOKEN)')
            log('weight-space reference check (CPU)')
            report = nearest_reference(folder(sources['teacher_plus']['repo'],sources['teacher_plus']['revision']),
                                       folder(sources['teacher_base']['repo'],sources['teacher_base']['revision']),alternatives)
            write_json(evidence_path,{**report,'sources':sources,'alternatives':list(alternatives)})
            log(f"evidence: {report['relative_l2_distance']}, nearest {report['nearest']}")
        if not json.loads(evidence_path.read_text(encoding='utf-8'))['nearest']:
            raise SystemExit('T0 is not the nearest checkpoint to T+: the DPO-reference assumption fails')

    # configs
    from chunkratio.artifacts import write_json
    from chunkratio.riskkd import server_config
    from chunkratio.config import validate
    protocol = json.loads((ROOT/'config/crpd/narval'/f"{spec['protocol']}.json").read_text(encoding='utf-8-sig'))
    if args.dry_run and not (data/'prepared.json').exists():
        revisions,split_hash,evidence = {k:'<pinned at prepare>' for k in ('reward_model','preference_dataset','sft_dataset',
                                         'teacher_base_model','student_base_model')},'<split hash>',{
                                         'relative_l2_distance':{},'rule':'<verify>'}
    else:
        revisions = json.loads(revisions_path.read_text(encoding='utf-8'))
        split_hash = json.loads((data/'ultrafeedback_train.splits.json').read_text(encoding='utf-8'))['assignment_sha256']
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    # Memory plan (DECISIONS 94): collection budgets and training processes per GPU from the GPU size.
    from chunkratio.riskkd import memory_plan
    gpu_gb = args.gpu_memory_gb or (180. if args.dry_run else gpu_memory_gb(gpus))
    plan = memory_plan(args.pair,gpu_gb)
    per_gpu = args.trainings_per_gpu or plan['trainings_per_gpu']
    speed = {**plan['collection'],**{k:v for k,v in (('policy_tokens_per_microbatch',args.policy_tokens),
                                                     ('generation_tokens_per_batch',args.generation_tokens)) if v}}
    log(f"memory plan for {gpu_gb:.0f} GB GPUs: {speed}, {per_gpu} training process(es) per GPU")
    splits = ['train'] if args.variant == 'crpd_raw' else None
    steps,interval = (args.steps,args.selection_interval) if args.steps else DEFAULT_STEPS.get(args.scale,(None,None))
    cache = str(work/'cache')
    def config(gamma: float, output: str, **extra) -> dict:
        cfg = server_config(args.pair,args.variant,revisions,split_hash,str(train_file),output,cache,alpha=args.alpha,
                            gamma=gamma,pair_sources=args.pair_sources,steps=steps or 1,batch_size=args.batch_size,
                            selection_interval=interval or steps or 1,protocol=protocol,speed=speed,
                            collection_splits=splits,evidence=evidence)
        cfg.update(extra)
        if not args.dry_run:
            validate(cfg)
        return cfg
    names = {g:f"{args.variant}-alpha{number(args.alpha)}-gamma{number(g)}-{args.pair_sources}" for g in gammas}
    paths = {}
    for g,name in names.items():
        paths[g] = work/'configs'/f'{name}.json'
        if not args.dry_run:
            write_json(paths[g],config(g,str(work/'runs'/name/'seed0')))

    # collect (one worker per GPU)
    if 'collect' in stages:
        last = ['collect-candidates']+(['collect-correction'] if args.variant == 'crpd_main' else [])
        jobs = []
        for index,gpu in enumerate(gpus):
            shard = {'index':index,'count':len(gpus)} if len(gpus) > 1 else None
            path = work/'configs'/f'collect-shard{index}.json'
            if not args.dry_run:
                write_json(path,config(gammas[0],str(work/'workers'/f'shard{index}'),
                                       **({'collection_shard':shard} if shard else {})))
            commands = [[py,'-m','chunkratio',stage,'--config',str(path)] for stage in last]
            jobs.append((commands,logs/f'collect-shard{index}.log',gpu))
        def worker(job: tuple[list[list[str]],Path,str]) -> None:
            for command in job[0]:
                for attempt in range(3):
                    try:
                        run(command,job[2],job[1],args.dry_run)
                        break
                    except StageFailed as error:
                        if not error.oom or attempt == 2:
                            raise
                        # Halve the collection budgets and resume from the shards already written.
                        path = Path(command[command.index('--config')+1])
                        cfg = json.loads(path.read_text(encoding='utf-8'))
                        for key in ('generation_tokens_per_batch','generation_max_rows','score_tokens_per_batch',
                                    'score_max_states','correction_generation_batch_size'):
                            if cfg.get(key):
                                cfg[key] = max(1,cfg[key]//2)
                        write_json(path,cfg)
                        log(f'out of GPU memory in {path.name}; halved the collection budgets and resuming')
        log(f'collect: {len(gpus)} worker(s)')
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker,jobs))

    # train (one config per GPU; full scale fixes the steps from the pair count first)
    if 'train' in stages:
        if steps is None and not args.dry_run:
            from chunkratio.artifacts import read_json
            from chunkratio.riskkd import steps_for
            for g,path in paths.items():
                run([py,'-m','chunkratio','build-targets','--config',str(path)],gpus[0],logs/f'{names[g]}.log',False)
                cfg = read_json(path)
                pairs = len(read_json(Path(read_json(Path(cfg['output'])/'target_index.json')['path']))['pairs'])
                total = steps_for(pairs,args.batch_size,args.passes)
                cfg.update(steps=total,selection_interval=max(1,-(-total//args.selections)))
                write_json(path,cfg)
                log(f'{names[g]}: {pairs} pairs -> {total} steps, selection every {cfg["selection_interval"]}')
        parallel([([py,'-m','chunkratio','evaluate','--config',str(paths[g])],logs/f'{names[g]}.log') for g in gammas],
                 gpus,per_gpu,args.dry_run)

    runs = [work/'runs'/names[g]/'seed0' for g in gammas]
    if 'heldout' in stages:
        parallel([([py,'script/server/heldout_eval.py','--run',str(r),'--test',str(test_file)],logs/f'heldout-{r.parent.name}.log')
                  for r in runs],gpus,per_gpu,args.dry_run)
    if 'benchmarks' in stages:
        run([args.eval_python,'script/server/benchmarks.py','--pair',args.pair,'--runs',*map(str,runs),
             '--gpus',','.join(g for _ in range(per_gpu) for g in gpus),'--output',str(work/'benchmarks')],
            None,logs/'benchmarks.log',args.dry_run)
    if not args.dry_run:
        for r in runs:
            if (r/'heldout.json').exists():
                h = json.loads((r/'heldout.json').read_text(encoding='utf-8'))
                d = h['rm_student_minus_rho']
                log(f"{r.parent.name}: held-out RM vs rho {d['mean']:+.3f} {d['bootstrap_95']} ({d['wins']}/{d['losses']}), "
                    f"EOS {h['eos_rate']['student']:.3f}, length {h['length_mean']['student']:.0f}")
    log('done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
