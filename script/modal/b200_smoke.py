"""Modal: lightweight smoke test and GPU memory probe of the server launcher on one B200 per pair (DECISIONS 95).

For each pair, in its own container with one B200:
  1. script/server/run_pair.py --scale smoke (32 train prompts, train-only collection, 2 gammas trained at once on the
     GPU, 10 steps, held-out check on 16 test prompts), with --skip-verify (no 16-32 GB base-model download);
  2. script/server/memory_probe.py at the planned full batch sizes (the smoke batches are far smaller);
  3. an NVML sampler over the whole container records device memory used, with the stage each sample falls in.
Results: volume crpd-b200-smoke, output/<pair>/summary.json (and the run logs).

  modal deploy script/modal/b200_smoke.py
  python -c "import modal; [print(modal.Function.from_name('crpd-b200-smoke','smoke').spawn(p).object_id) for p in
             ('qwen3_1p7b_to_0p6b','qwen3_8b_to_1p7b','llama31_8b_to_llama32_1b')]"
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

APP = 'crpd-b200-smoke'
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/root/repo')
app = modal.App(APP)
volume = modal.Volume.from_name(APP,create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         # B200 (sm_100) needs a CUDA 12.8 build of PyTorch.
         .pip_install('torch==2.7.1',index_url='https://download.pytorch.org/whl/cu128')
         .pip_install('transformers==4.52.4','datasets==3.5.0','numpy','safetensors','tokenizers','huggingface_hub',
                      'hf_transfer','nvidia-ml-py','pyyaml')
         .env({'HF_HOME':'/vol/hf','HF_HUB_ENABLE_HF_TRANSFER':'1','TOKENIZERS_PARALLELISM':'false',
               'PYTHONUNBUFFERED':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
         .add_local_dir(REPO/'chunkratio','/root/repo/chunkratio',ignore=['__pycache__'])
         .add_local_dir(REPO/'script'/'server','/root/repo/script/server',ignore=['__pycache__'])
         .add_local_dir(REPO/'config'/'crpd','/root/repo/config/crpd')
         .add_local_dir(REPO/'reference','/root/repo/reference'))


@app.function(image=image,gpu='B200',cpu=8.0,memory=196608,timeout=3*3600,volumes={'/vol':volume})
def smoke(pair: str) -> dict:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    total = pynvml.nvmlDeviceGetMemoryInfo(handle).total/1e9
    stage,samples,stop = ['start'],[],threading.Event()
    def sample() -> None:
        while not stop.is_set():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            samples.append((time.time(),stage[0],info.used/1e9,util))
            time.sleep(.5)
    threading.Thread(target=sample,daemon=True).start()
    root = Path('/vol/output')/pair
    root.mkdir(parents=True,exist_ok=True)
    timings = {}
    def step(name: str, command: list[str]) -> int:
        stage[0] = name
        start = time.time()
        with open(root/f'{name}.log','w',encoding='utf-8') as out:
            status = subprocess.run(command,cwd='/root/repo',stdout=out,stderr=subprocess.STDOUT).returncode
        timings[name] = {'seconds':round(time.time()-start,1),'status':status}
        volume.commit()
        return status
    py = sys.executable
    base = [py,'script/server/run_pair.py','--pair',pair,'--scale','smoke','--gpus','0','--gammas','0.05,0.03',
            '--trainings-per-gpu','2','--skip-verify','--root',str(root/'server')]
    status = step('prepare',base+['--stages','prepare'])
    if status == 0:
        status = step('collect',base+['--stages','collect'])
    if status == 0:
        status = step('train_2_per_gpu',base+['--stages','train'])
    if status == 0:
        step('heldout',base+['--stages','heldout'])
    prepared = root/'server'/pair/'data'/'rows2048'/'ultrafeedback_train.jsonl'
    if prepared.exists():
        step('probe',[py,'script/server/memory_probe.py','--pair',pair,'--prompts',str(prepared),
                      '--output',str(root/'probe.json')])
    stop.set()
    time.sleep(1)
    by_stage = {}
    for _,name,used,util in samples:
        entry = by_stage.setdefault(name,{'peak_used_gb':0.,'util':[]})
        entry['peak_used_gb'] = max(entry['peak_used_gb'],used)
        entry['util'].append(util)
    nvml = {k:{'peak_used_gb':round(v['peak_used_gb'],2),'mean_util_pct':round(sum(v['util'])/len(v['util']),1)}
            for k,v in by_stage.items()}
    runs = {}
    for diag in sorted((root/'server').rglob('runs/*/seed0/diagnostics.json')):
        d = json.loads(diag.read_text())
        runs[diag.parent.parent.name] = {'peak_cuda_allocated_gb':round(d['cost']['peak_cuda_allocated_bytes']/1e9,2),
                                         'stage_seconds':{k:round(v) for k,v in d['cost'].get('stage_seconds',{}).items()},
                                         'selection':d['cost'].get('selection_this_invocation'),
                                         'pairs':d['counts']['pairs']}
    heldout = {}
    for path in sorted((root/'server').rglob('runs/*/seed0/heldout.json')):
        h = json.loads(path.read_text())
        heldout[path.parent.parent.name] = {k:h[k] for k in ('rm_student_minus_rho','eos_rate','length_mean')}
    probe = json.loads((root/'probe.json').read_text()) if (root/'probe.json').exists() else None
    summary = {'pair':pair,'gpu':pynvml.nvmlDeviceGetName(handle),'gpu_total_gb':round(total,1),'timings':timings,
               'nvml_by_stage':nvml,'runs':runs,'heldout':heldout,'probe':probe}
    (root/'summary.json').write_text(json.dumps(summary,indent=1))
    with open(root/'nvml_timeline.jsonl','w') as out:
        for t,name,used,util in samples:
            out.write(json.dumps({'t':round(t,1),'stage':name,'used_gb':round(used,2),'util':util})+'\n')
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    pairs = ('qwen3_1p7b_to_0p6b','qwen3_8b_to_1p7b','llama31_8b_to_llama32_1b')
    for result in smoke.map(pairs,return_exceptions=True):
        print(json.dumps(result,default=str)[:4000])
