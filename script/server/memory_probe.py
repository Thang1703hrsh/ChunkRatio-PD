"""GPU memory and throughput probe for one pair at full batch sizes (DECISIONS 95). Runs on one GPU, on Modal or on
the training server, and recommends budgets from measured bytes per token.

A small smoke run never fills the planned batches, so it cannot show the real peaks. This probe builds batches of
the planned size from real prepared prompts and measures each phase separately:
  load      the three FP32 roles
  generate  T+ (the largest KV cache) sampling 512 tokens for as many rows as the generation budget allows,
            at the planned budget and at 1.5x
  score     shared-prefix scoring of 4 siblings per state with each role, at the planned budget and at 2x
  train     the teachers dropped; one student microbatch (bf16 autocast, FP32 weights, AdamW) at 8192, 16384
            and 24576 policy tokens, with rho's scores
  rm        the FP32 8B selection RM scoring 1024-token responses
Peaks are torch's max allocated/reserved and NVML's device memory used. A phase that runs out of memory is
recorded as such and the probe goes on.

  python script/server/memory_probe.py --pair qwen3_8b_to_1p7b --prompts <prepared train jsonl> --output probe.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


class NvmlPeak:
    """Background sampler of device memory used (NVML), so peaks between torch calls are seen too."""
    def __init__(self, index: int = 0) -> None:
        import pynvml
        pynvml.nvmlInit()
        self.nvml,self.handle = pynvml,pynvml.nvmlDeviceGetHandleByIndex(index)
        self.total_gb = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total/1e9
        self.peak,self.stop = 0.,threading.Event()
        threading.Thread(target=self.loop,daemon=True).start()
    def loop(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak,self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used/1e9)
            time.sleep(.05)
    def reset(self) -> None:
        self.peak = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used/1e9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair',required=True)
    parser.add_argument('--prompts',required=True,help='Prepared UltraFeedback JSONL of the pair (prompt_ids)')
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpu-memory-gb',type=float,default=None)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    from chunkratio.artifacts import read_jsonl, write_json
    from chunkratio.pilot import REWARD_MODEL
    from chunkratio.riskkd import RISKKD, memory_plan
    from chunkratio.sampling import generate_batch, stream
    from chunkratio.scoring import sibling_logprobs
    spec = RISKKD[args.pair]
    nvml = NvmlPeak()
    # Plan with run_pair.py's convention (nvidia-smi MiB / 1024), so the probe measures the budgets the server uses.
    gpu_gb = args.gpu_memory_gb or nvml.total_gb*1e9/2**30
    plan = memory_plan(args.pair,gpu_gb)
    report = {'pair':args.pair,'gpu':torch.cuda.get_device_name(0),'gpu_total_gb':round(nvml.total_gb,1),
              'plan':plan,'phases':{}}
    prompts = [r['prompt_ids'] for r in read_jsonl(Path(args.prompts)) if len(r['prompt_ids']) <= 2048]
    lengths = sorted(len(p) for p in prompts)
    # A long-ish prompt (90th percentile) makes every row near the worst case the budgets allow.
    long = lengths[int(.9*(len(lengths)-1))]
    rows_of = lambda n: [p for p in prompts if len(p) >= long][:1]*n
    report['prompt_tokens'] = {'median':statistics.median(lengths),'p90':long,'max':lengths[-1]}

    def phase(name: str, fn, **info) -> None:
        """Run one phase with fresh peak counters; record memory, time and out-of-memory."""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        nvml.reset()
        start = time.perf_counter()
        entry = dict(info)
        try:
            extra = fn() or {}
            torch.cuda.synchronize()
            entry.update(ok=True,**extra)
        except torch.cuda.OutOfMemoryError:
            entry.update(ok=False,error='out of memory')
        entry.update(seconds=round(time.perf_counter()-start,2),
                     torch_peak_allocated_gb=round(torch.cuda.max_memory_allocated()/1e9,2),
                     torch_peak_reserved_gb=round(torch.cuda.max_memory_reserved()/1e9,2),
                     nvml_peak_used_gb=round(nvml.peak,2))
        report['phases'][name] = entry
        print(json.dumps({name:entry}),flush=True)
        torch.cuda.empty_cache()

    from transformers import AutoTokenizer
    ckpt = spec['checkpoints']
    tok = AutoTokenizer.from_pretrained(ckpt['teacher_base']['repo'],revision=ckpt['teacher_base']['revision'])
    eos = tok.convert_tokens_to_ids(spec['eos'])
    models = {}
    def load() -> dict:
        for role,source in ckpt.items():
            models[role] = AutoModelForCausalLM.from_pretrained(source['repo'],revision=source['revision'],
                                                                torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
        return {'resident_gb':round(torch.cuda.memory_allocated()/1e9,2)}
    phase('load',load)
    resident = torch.cuda.memory_allocated()

    collection = plan['collection']
    for factor in (1.,1.5):
        budget = int(collection['generation_tokens_per_batch']*factor)
        rows = min(int(collection['generation_max_rows']*factor),budget//(long+512))
        def generate(rows: int = rows) -> dict:
            rngs = [stream(0,f'probe/{i}') for i in range(rows)]  # CPU generators, as in collection
            out = generate_batch(models['teacher_plus'],rows_of(rows),[512]*rows,eos,rngs,gpu_sampling=True)
            new = sum(len(o) for o in out)
            return {'rows':rows,'new_tokens':new}
        phase(f'generate_x{factor:g}',generate,budget_tokens=budget,prompt_tokens=long)
        entry = report['phases'][f'generate_x{factor:g}']
        if entry.get('ok'):
            entry['new_tokens_per_s'] = round(entry['new_tokens']/entry['seconds'])

    for factor in (1.,2.):
        budget = int(collection['score_tokens_per_batch']*factor)
        states = max(1,min(int(collection['score_max_states']*factor),budget//(long+32)))
        def score(states: int = states) -> dict:
            prefix = rows_of(states)
            chunks = [[[(i*7+j) % 1000+10 for j in range(32)] for i in range(4)] for _ in range(states)]
            with torch.no_grad():
                for role in ('teacher_plus','teacher_base','student_ref'):
                    sibling_logprobs(models[role],prefix,chunks,eos)
            return {'states':states,'chunks':4*states}
        phase(f'score_x{factor:g}',score,budget_tokens=budget,prompt_tokens=long)

    # Training: the teachers are released as in the server configs.
    for role in ('teacher_plus','teacher_base'):
        del models[role]
    import gc
    gc.collect()
    rho = models['student_ref']
    student = AutoModelForCausalLM.from_pretrained(ckpt['student_ref']['repo'],revision=ckpt['student_ref']['revision'],
                                                   torch_dtype=torch.float32).to('cuda').eval()
    optimizer = torch.optim.AdamW(student.parameters(),lr=1e-6,weight_decay=0.)
    depth = int(statistics.median(lengths))+256  # a mid-response state
    state = (rows_of(1)[0]*4)[:depth]
    for budget in (8192,16384,24576):
        pairs = max(1,budget//(depth+32))
        def train(pairs: int = pairs) -> dict:
            prefix = [state]*pairs
            chunks = [[[(i*11+j) % 1000+10 for j in range(32)] for i in range(2)] for _ in range(pairs)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                s = torch.stack([x for x in sibling_logprobs(student,prefix,chunks,eos)])
                with torch.no_grad():
                    r = torch.stack([x for x in sibling_logprobs(rho,prefix,chunks,eos)])
            loss = torch.nn.functional.softplus(-.03*((s[:,0]-r[:,0])-(s[:,1]-r[:,1]))).mean()
            loss.backward()
            optimizer.step()
            return {'pairs':pairs,'state_tokens':depth}
        phase(f'train_{budget}',train,policy_tokens=budget)
        entry = report['phases'][f'train_{budget}']
        if entry.get('ok'):
            entry['pairs_per_s'] = round(entry['pairs']/entry['seconds'],1)
    del student,optimizer
    gc.collect()

    def rm() -> dict:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(REWARD_MODEL).sha
        model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL,revision=revision,
                                                                   torch_dtype=torch.float32).to('cuda').eval()
        with torch.no_grad():
            for _ in range(8):
                model(torch.randint(10,1000,(1,long+1024),device='cuda'))
        return {'scored':8,'tokens_each':long+1024}
    phase('rm',rm)

    # Recommendation: bytes per token measured above, filled to 90% of the device.
    usable = .9*nvml.total_gb
    rec = {}
    # A larger budget is recommended only when the larger probe was measurably faster (B200, 6 November 2026:
    # generation x1.5 was 13% slower per token and scoring x2 only 1.6% faster, so memory was not the limit).
    phases = report['phases']
    def rate(name: str, unit: str) -> float | None:
        p = phases.get(name,{})
        return p[unit]/p['seconds'] if p.get('ok') and p.get('seconds') else None
    g = phases.get('generate_x1',{})
    if g.get('ok'):
        per_token = max(1e-9,(g['torch_peak_reserved_gb']-resident/1e9)/(g['rows']*(long+512)))
        fill = int((usable-resident/1e9)/per_token)//8192*8192
        small,large = rate('generate_x1','new_tokens'),rate('generate_x1.5','new_tokens')
        rec['generation_tokens_per_batch'] = (fill if small and large and large > 1.05*small
                                              else collection['generation_tokens_per_batch'])
        rec['generation_speedup_x1.5'] = round(large/small,3) if small and large else None
    small,large = rate('score_x1','chunks'),rate('score_x2','chunks')
    if small and large:
        rec['score_speedup_x2'] = round(large/small,3)
        rec['score_tokens_per_batch'] = collection['score_tokens_per_batch']*(2 if large > 1.05*small else 1)
    peaks = {k:v['nvml_peak_used_gb'] for k,v in report['phases'].items() if k.startswith('train_') and v.get('ok')}
    if peaks:
        rec['training_peak_gb_at_8192'] = peaks.get('train_8192')
        rec['trainings_per_gpu_at_8192'] = int(usable//max(peaks.get('train_8192',usable),1)) if peaks.get('train_8192') else None
    report['recommendation'] = rec
    write_json(Path(args.output),report)
    print(json.dumps({'recommendation':rec}),flush=True)
    nvml.stop.set()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
