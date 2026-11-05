"""Isolated fast CRPD x10 on the existing HDA trio; never pushes models.

Deploy with ``modal deploy script/modal/crpd_fast_x10.py --profile quangltt280305``.
Then ``modal run --detach script/modal/crpd_fast_x10.py::run --revision COMMIT``.
Training: one A100-80GB, main then raw with shared immutable collection shards.
Benchmarks: one L40S per model, Winogrande 5-shot and TruthfulQA-MC2 0-shot.
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

APP_NAME = 'crpd-fast-hda-x10'
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME,create_if_missing=True)
source = modal.Volume.from_name('crpd-qwen3-pilot-hda-x10').read_only()
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/code')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0','transformers==4.52.4','datasets==3.5.0','numpy','pyyaml',
                      'safetensors','tokenizers','huggingface_hub','pytest')
         .env({'HF_HOME':'/vol/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
               'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
         .add_local_dir(REPO/'chunkratio','/code/chunkratio',ignore=['__pycache__'])
         .add_local_dir(REPO/'config','/code/config')
         .add_local_dir(REPO/'reference','/code/reference')
         .add_local_dir(REPO/'tests','/code/tests',ignore=['__pycache__']))
eval_image = (modal.Image.debian_slim(python_version='3.12')
              .pip_install('torch==2.6.0','transformers==4.56.2','datasets==3.5.0',
                           'lm-eval==0.4.9.2','sentencepiece')
              .env({'HF_HOME':'/vol/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}))
mounts = {'/vol':volume,'/source':source}


def work(revision: str) -> Path:
    root = Path('/root/work')
    root.mkdir(exist_ok=True)
    for name in ('chunkratio','config','reference','tests'):
        if not (root/name).exists():
            (root/name).symlink_to(Path('/code')/name)
    if not (root/'output').exists():
        (root/'output').symlink_to('/vol/output')
    Path('/vol/output').mkdir(exist_ok=True)
    os.chdir(root)
    sys.path.insert(0,str(root))
    os.environ['CRPD_CODE_REVISION'] = revision
    return root


@app.function(image=image,cpu=2,memory=8192,timeout=600,volumes=mounts)
def prepare(revision: str) -> str:
    work(revision)
    from chunkratio.artifacts import atomic_bytes, read_json, read_jsonl, source_digest, write_json
    from chunkratio.fast_run import fast_config
    root = Path('output/fast_hda_x10')/source_digest()[:16]
    root.mkdir(parents=True,exist_ok=True)
    original = Path('/source/output/narval/qwen3_1p7b_to_0p6b_hda/pilot/configs/crpd_main-alpha0p03.json')
    base = read_json(original)
    prompt_source = Path('/source')/base['prompts']
    snapshot = root/'input_prompts.jsonl'
    if snapshot.exists() and snapshot.read_bytes() != prompt_source.read_bytes():
        raise ValueError('x10 input changed; refuse to overwrite the isolated snapshot')
    atomic_bytes(snapshot,prompt_source.read_bytes())
    base['prompts'] = snapshot.as_posix()
    write_json(root/'base_config.json',base)
    for name in ('crpd_main','crpd_raw'):
        write_json(root/f'{name}.json',fast_config(base,root,root/'prompts.jsonl',name))
    # Fixed held-out sample, selected by ID hash before observing model outcomes.
    # This is student-held-out; teacher exposure is not certified by these artifacts.
    test = read_jsonl(Path('/source/data/narval/qwen3_1p7b_to_0p6b_hda/ultrafeedback_test.jsonl'))
    from chunkratio.artifacts import digest
    training_ids = {tuple(row['prompt_ids']) for row in read_jsonl(snapshot)}
    test = sorted((row for row in test if tuple(row['prompt_ids']) not in training_ids
                   and len(row['prompt_ids']) <= base.get('max_prompt_tokens',2048)),
                  key=lambda row:digest(row['prompt_ids']))[:64]
    if not test:
        raise ValueError('no disjoint held-out RM prompts')
    atomic_bytes(root/'test_prompts.jsonl',(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in test)).encode())
    write_json(root/'input_provenance.json',{'base_config_source':str(original.relative_to('/source')),
               'test_prompts':len(test),'test_selection':'first 64 by sha256(prompt_ids), disjoint from x10, prompt <=2048 tokens',
               'teacher_test_exposure':'not verified'})
    volume.commit()
    return root.as_posix()


@app.function(image=image,gpu='A100-80GB',cpu=4,memory=49152,timeout=1800,volumes=mounts)
def gpu_check(revision: str, root: str) -> dict:
    work(revision)
    import torch
    from chunkratio.artifacts import read_json, read_jsonl, write_json
    from chunkratio.fast import score_candidate_records
    from chunkratio.models import load_models
    from chunkratio.scoring import score_batch
    cfg = read_json(Path(root)/'crpd_main.json')
    models,student,_,info = load_models(cfg)
    rows = read_jsonl(Path(root)/'input_prompts.jsonl')
    rows = sorted((r for r in rows if r.get('chosen_ids')),key=lambda r:len(r['prompt_ids']))
    chosen = [rows[i*len(rows)//6] for i in range(6)]
    records = []
    for row in chosen:
        tokens = row['chosen_ids'][:32]
        if info['eos'] in tokens:
            tokens = tokens[:tokens.index(info['eos'])+1]
        records.append({'state':{'prefix_ids':row['prompt_ids']},'candidates':[{'tokens':tokens}]})
    expected = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    for name,model in models.items():
        expected[name] = [score_batch(model,[r['state']['prefix_ids']],[r['candidates'][0]['tokens']],info['eos']).item()
                          for r in records]
    torch.cuda.synchronize()
    slow = time.perf_counter()-start
    start = time.perf_counter()
    score_candidate_records(records,models,info['eos'],cfg['score_tokens_per_batch'])
    torch.cuda.synchronize()
    fast = time.perf_counter()-start
    error = max(abs(record['candidates'][0]['logp_'+name]-expected[name][i])
                for name in models for i,record in enumerate(records))
    if error > .002:
        raise ValueError(f'cross-state FP32 score drift exceeds 0.002: {error}')
    prefixes = [r['state']['prefix_ids'] for r in records[:2]]
    chunks = [r['candidates'][0]['tokens'] for r in records[:2]]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        a = score_batch(student,prefixes,chunks,info['eos'],frozen=False)
        b = score_batch(models['student_ref'],prefixes,chunks,info['eos'])
    margin = float((a-b).abs().max())
    if margin != 0.:
        raise ValueError(f'initial student/rho margin is not zero: {margin}')
    result = {'gpu':torch.cuda.get_device_name(),'rows':len(records),'max_score_error':error,
              'initial_autocast_margin':margin,'individual_seconds':slow,'batched_seconds':fast,
              'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    write_json(Path(root)/'gpu_preflight.json',result)
    volume.commit()
    return result


@app.function(image=image,gpu='A100-80GB',cpu=4,memory=49152,timeout=4*3600,retries=0,volumes=mounts)
def train(revision: str, root: str, variant: str) -> dict:
    work(revision)
    from chunkratio.artifacts import write_json
    from chunkratio.fast_run import run
    stop = threading.Event()
    samples = []
    def monitor():
        while not stop.wait(5):
            try:
                line = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                                '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                memory,util = map(float,line.split(','))
                samples.append({'elapsed_s':time.perf_counter()-start,'memory_mib':memory,'utilization':util})
            except (OSError,ValueError,subprocess.SubprocessError):
                pass
    start = time.perf_counter()
    thread = threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        result = run(Path(root)/'base_config.json',Path(root),variant)[variant]
        return {k:result[k] for k in ('counts','policy','cost')}
    finally:
        stop.set()
        thread.join(timeout=12)
        write_json(Path(root)/f'{variant}_resources.json',{'wall_seconds':time.perf_counter()-start,
                   'gpu':'A100-80GB','samples':samples})
        volume.commit()


@app.function(image=eval_image,gpu='L40S',cpu=2,memory=24576,timeout=3600,retries=0,volumes={'/vol':volume})
def benchmark(root: str, name: str) -> dict:
    volume.reload()
    model = (f'/vol/{root}/{name}/student' if name != 'rho' else
             'hoangducanh1865/qwen3-0.6b-deita-sft-student,revision=fc0f225cb63e5db6e84a7e5705c997891ec85d65')
    results = {}
    for task,fewshot in (('winogrande',5),('truthfulqa_mc2',0)):
        output = Path('/vol')/root/'benchmarks'/name/task
        subprocess.run([sys.executable,'-m','lm_eval','--model','hf','--model_args',f'pretrained={model},dtype=float32',
                        '--tasks',task,'--num_fewshot',str(fewshot),'--batch_size','32','--device','cuda',
                        '--log_samples','--output_path',str(output)],check=True)
        latest = sorted(output.rglob('results_*.json'))[-1]
        results[task] = json.loads(latest.read_text())['results'][task]
        volume.commit()
    return results


@app.function(image=image,cpu=.25,memory=2048,timeout=12*3600,volumes={'/vol':volume})
def run(revision: str) -> dict:
    root = prepare.remote(revision)
    check = gpu_check.remote(revision,root)
    if not (check['initial_autocast_margin'] == 0 and check['max_score_error'] <= .002):
        raise ValueError('GPU preflight failed')
    training = {}
    for variant in ('crpd_main','crpd_raw'):
        training[variant] = train.remote(revision,root,variant)
    # Benchmarks are evaluated only after validation has selected the exported checkpoint.
    calls = {name:benchmark.spawn(root,name) for name in ('rho','crpd_main','crpd_raw')}
    rm_call = heldout_rm.spawn(revision,root)
    benchmarks = {name:call.get() for name,call in calls.items()}
    heldout = rm_call.get()
    result = {'root':root,'gpu_preflight':check,'training':training,'benchmarks':benchmarks,'heldout_rm':heldout}
    volume.reload()
    work(revision)
    from chunkratio.artifacts import write_json
    output = Path('/vol')/root/'run_summary.json'
    write_json(output,result)
    volume.commit()
    return result


@app.function(image=image,gpu='A100-80GB',cpu=4,memory=49152,timeout=2*3600,retries=0,volumes=mounts)
def heldout_rm(revision: str, root: str) -> dict:
    """Report paired RM rewards on the fixed test prompts, never selecting checkpoints."""
    work(revision)
    import gc
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from chunkratio.artifacts import read_json, read_jsonl, write_json
    from chunkratio.evaluation import reward_selector
    from chunkratio.pilot import HDA_CHECKPOINTS
    cfg = read_json(Path(root)/'crpd_main.json')
    rows = read_jsonl(Path(root)/'test_prompts.jsonl')
    tokenizer = AutoTokenizer.from_pretrained(HDA_CHECKPOINTS['student_ref']['repo'],
                                               revision=HDA_CHECKPOINTS['student_ref']['revision'])
    tokenizer.eos_token = cfg['eos_token']
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selector,cost = reward_selector(cfg,[{**row,'split':'select'} for row in rows],tokenizer)
    results = {}
    for name in ('rho','crpd_main','crpd_raw'):
        path = HDA_CHECKPOINTS['student_ref']['repo'] if name == 'rho' else str(Path(root)/name/'student')
        kwargs = {'revision':HDA_CHECKPOINTS['student_ref']['revision']} if name == 'rho' else {}
        model = AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,**kwargs)
        model = model.to('cuda').eval().requires_grad_(False)
        score = selector(model)
        results[name] = {'mean':score,'scores':selector.last_scores.copy()}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write_json(Path(root)/'heldout_rm_partial.json',{'split':'test','results':results,'n':len(rows)})
        volume.commit()
    rng = np.random.default_rng(0)
    indices = rng.integers(0,len(rows),(10000,len(rows)))
    comparisons = {}
    for left,right in (('crpd_main','rho'),('crpd_raw','rho'),('crpd_main','crpd_raw')):
        delta = np.asarray(results[left]['scores'])-np.asarray(results[right]['scores'])
        low,high = np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()
        comparisons[left+'_vs_'+right] = {'mean_delta':float(delta.mean()),'paired_bootstrap_95':[low,high]}
    result = {'split':'test','n':len(rows),'prompt_ids':[row['id'] for row in rows],
              'results':results,'comparisons':comparisons,'cost':cost,
              'note':'fixed student-held-out prompts; teacher exposure not verified; not used for checkpoint selection'}
    write_json(Path(root)/'heldout_rm.json',result)
    volume.commit()
    return result
