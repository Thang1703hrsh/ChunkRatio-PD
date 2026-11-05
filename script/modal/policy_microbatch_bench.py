"""Modal benchmark: train-policy step time and peak GPU memory for microbatch settings.

Real rho states from the hda x10 prefix shards; each state's next 32 rho tokens form chunk c and a shifted
window chunk d, with random targets (timing depends on lengths only). The student is an exact rho copy
under the pilot's bf16 autocast, AdamW and a 128-pair batch drawn by the trainer's own sampler, so the
measured steps match train-policy except for checkpoint selection. Nothing is trained for real.

  modal deploy script/modal/policy_microbatch_bench.py && python script/modal/policy_microbatch_bench.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

APP_NAME = 'crpd-policy-microbatch-bench'
PREFIXES = 'output/narval/qwen3_1p7b_to_0p6b_hda/pilot/cache-bb150/prefixes-a81d86e91e4c6009f50c'
SETTINGS = [{'policy_tokens_per_microbatch':8192}]+\
           [{'policy_tokens_per_microbatch':t,'shared_prefix_scoring':True} for t in (8192,12288,16384)]
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/root/repo')

app = modal.App(APP_NAME)
volume = modal.Volume.from_name('crpd-qwen3-pilot-hda-x10')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0','transformers==4.52.4','numpy','safetensors','tokenizers','huggingface_hub')
         .env({'HF_HOME':'/vol/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
               'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
         .add_local_dir(REPO/'chunkratio','/root/repo/chunkratio',ignore=['__pycache__']))


@app.function(image=image,gpu='A100-80GB',cpu=4.0,memory=32768,timeout=3600,volumes={'/vol':volume})
def bench(batch_size: int = 128, steps: int = 3) -> dict:
    """Time `steps` optimizer steps (after one warm-up step) per microbatch setting."""
    import copy
    import random
    import time
    import torch
    sys.path.insert(0,'/root/repo')
    from transformers import AutoModelForCausalLM
    from chunkratio.artifacts import read_json
    from chunkratio.pilot import HDA_CHECKPOINTS
    from chunkratio.training import ChunkRatioTrainer
    rng = random.Random(0)
    pairs = []
    for path in sorted(Path('/vol',PREFIXES).glob('*.jsonl')):
        record = read_json(path)['payload']
        response = record['response']
        for state in record['states']:
            n = state['n']
            if len(response)-n < 40:
                continue
            pairs.append({'state':state,'weight':1.,'c':{'tokens':response[n:n+32],'target_score':rng.gauss(0,50)},
                          'd':{'tokens':response[n+8:n+40],'target_score':rng.gauss(0,50)}})
    lengths = sorted(len(p['state']['prefix_ids'])+32 for p in pairs)
    source = HDA_CHECKPOINTS['student_ref']
    rho = AutoModelForCausalLM.from_pretrained(source['repo'],revision=source['revision'],torch_dtype=torch.float32)
    rho = rho.to('cuda').eval().requires_grad_(False)
    student = copy.deepcopy(rho).requires_grad_(True)
    eos = 151645  # <|im_end|>
    base = {'seed':0,'alpha':.03,'device':'cuda','batch_size':batch_size,'policy_autocast_dtype':'bfloat16'}
    trainer = ChunkRatioTrainer(student,rho,{'route':'reference_mean','pairs':pairs},[],base,eos)
    optimizer = torch.optim.AdamW(student.parameters(),lr=1e-9,weight_decay=0.)
    results = []
    for setting in SETTINGS:
        trainer.cfg = {**base,**setting}
        trainer.rng = random.Random(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times, groups = [], []
        try:
            for step in range(steps+1):
                batch = trainer.sample()
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                micro = trainer.microbatches(batch)
                for part in micro:
                    loss,_ = trainer.batch_loss(part)
                    (loss*len(part)/len(batch)).backward()
                optimizer.step()
                torch.cuda.synchronize()
                if step:
                    times.append(time.perf_counter()-start)
                    groups.append(len(micro))
            row = {**setting,'seconds_per_step':sum(times)/len(times),'microbatches_per_step':sum(groups)/len(groups),
                   'peak_gib':torch.cuda.max_memory_allocated()/1024**3,'oom':False}
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            row = {**setting,'oom':True,'peak_gib':torch.cuda.max_memory_allocated()/1024**3}
        print(json.dumps(row),flush=True)
        results.append(row)
    report = {'gpu':torch.cuda.get_device_name(),'batch_size':batch_size,'pairs_available':len(pairs),
              'length_quantiles':{q:lengths[int(q*(len(lengths)-1))] for q in (.5,.9,.99,1.)},'results':results}
    out = Path('/vol/output/policy_microbatch_bench_shared.json')
    out.write_text(json.dumps(report,indent=1),encoding='utf-8')
    volume.commit()
    return report


@app.function(image=image,gpu='A100-80GB',cpu=4.0,memory=32768,timeout=3600,volumes={'/vol':volume})
def score_bench(states: int = 400) -> dict:
    """FP32 T+ scoring of 4 siblings per real state: cross-state rows vs one shared prefix per state."""
    import copy
    import time
    import torch
    sys.path.insert(0,'/root/repo')
    from transformers import AutoModelForCausalLM
    from chunkratio.artifacts import read_json
    from chunkratio.fast import score_candidate_records, score_candidate_siblings
    from chunkratio.pilot import HDA_CHECKPOINTS
    records = []
    for path in sorted(Path('/vol',PREFIXES).glob('*.jsonl')):
        payload = read_json(path)['payload']
        response = payload['response']
        for state in payload['states']:
            n = state['n']
            if len(response)-n >= 56:
                records.append({'state':state,'candidates':[{'tokens':response[n+8*k:n+8*k+32]} for k in range(4)]})
    records = records[:states]
    source = HDA_CHECKPOINTS['teacher_plus']
    model = AutoModelForCausalLM.from_pretrained(source['repo'],revision=source['revision'],torch_dtype=torch.float32)
    models = {'teacher_plus':model.to('cuda').eval().requires_grad_(False)}
    eos, results, scores = 151645, [], {}
    shared = lambda states: lambda rows,models,eos,budget: score_candidate_siblings(rows,models,eos,budget,states)
    for name,scorer,budget in (('cross_state',score_candidate_records,16384),('shared48',shared(48),16384),
                               ('shared48',shared(48),32768),('shared96',shared(96),32768)):
        rows = copy.deepcopy(records)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        scorer(rows,models,eos,budget)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        scores[(name,budget)] = [c['logp_teacher_plus'] for r in rows for c in r['candidates']]
        results.append({'scorer':name,'budget':budget,'seconds':seconds,'peak_gib':torch.cuda.max_memory_allocated()/1024**3})
        print(json.dumps(results[-1]),flush=True)
    reference = scores[('cross_state',16384)]
    for row in results:
        other = scores[(row['scorer'],row['budget'])]
        row['max_abs_diff_vs_cross_state'] = max(abs(a-b) for a,b in zip(reference,other))
    report = {'states':len(records),'rows':4*len(records),'results':results}
    Path('/vol/output/score_bench_shared.json').write_text(json.dumps(report,indent=1),encoding='utf-8')
    volume.commit()
    return report


if __name__ == '__main__':
    for name in ('bench','score_bench'):
        print('spawned',name,modal.Function.from_name(APP_NAME,name).spawn().object_id)
