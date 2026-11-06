"""Run several NARVAL pairs on one multi-GPU server with no idle GPU: a dependency-aware job scheduler over
script/server/run_pair.py (DECISIONS 98).

Each pair becomes jobs:
  prep           CPU lane: run_pair --stages prepare,verify (downloads, tokenization, weight check)
  collect/i      one hash shard of the collection on one GPU (--collect-shards K): shards of a pair spread over
                 every free GPU and fill gaps beside other pairs' training; any K gives the same data
  train/g        one gamma on one GPU (half a GPU where the memory plan fits two trainings per GPU)
  heldout/g      after train/g
  bench-base     rho and NARVAL benchmarks, right after prep (they do not depend on any run)
  bench          the students' benchmarks and the summary, after every heldout of the pair
  push           CPU lane, after bench, when --hub-namespace is given
A GPU has two units: collection takes a whole GPU (it is compute-bound), a training takes 2 / trainings_per_gpu,
held-out and benchmark jobs one. When a GPU frees, the highest-priority ready job that fits starts:
train > heldout > collect > bench > bench-base, then pair order (put the pair with the longest training first).
A job that shared a GPU and ran out of memory is retried once on a whole GPU. A failed job stops only its own
pair's later jobs. Finished jobs are marked in <root>/<pair>/<scale>/schedule/, so rerunning the same command
skips them (each run_pair stage also resumes on its own); --fresh ignores the marks.

  python script/server/run_all.py --gpus 0,1 --scale full --pair-sources within \
      --pair qwen3_8b_to_1p7b:0.1 --pair llama31_8b_to_llama32_1b:0.05 --pair qwen3_1p7b_to_0p6b:0.1 \
      --eval-python ../venv-eval/bin/python --hub-namespace <user>
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
UNITS = 2
RANK = {'train':0,'heldout':1,'collect':2,'bench':3,'bench-base':4,'prep':5,'push':6}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}",flush=True)


@dataclass
class Job:
    name: str                  # '<pair>/<kind>[/<detail>]'
    kind: str
    pair_rank: int
    order: int
    command: Callable[[str | None],list[str]]   # GPU id (None for CPU jobs) -> command
    units: int                 # 0: CPU lane
    deps: list[str] = field(default_factory=list)
    log: Path | None = None
    marker: Path | None = None
    state: str = 'waiting'     # waiting, running, done, failed, skipped
    retried: bool = False

    def key(self) -> tuple:
        return (RANK[self.kind],self.pair_rank,self.order)


def oom(logfile: Path | None) -> bool:
    if not logfile or not logfile.exists():
        return False
    tail = logfile.read_text(encoding='utf-8',errors='replace')[-50000:]
    return 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail


class Scheduler:
    """Greedy list scheduler over GPU units plus one CPU lane. `execute(command, gpu, log) -> exit status`."""
    def __init__(self, jobs: list[Job], gpus: list[str], execute: Callable[[list[str],str | None,Path | None],int],
                 fresh: bool = False, poll: float = 1.) -> None:
        self.jobs = {j.name:j for j in jobs}
        self.free = {g:UNITS for g in gpus}
        self.cpu_busy = False
        self.execute,self.poll = execute,poll
        self.lock = threading.Lock()
        self.events: list[tuple[str,str,str | None]] = []   # (event, job, gpu), for the record and the tests
        for job in jobs:
            if not fresh and job.marker and job.marker.exists() and \
                    json.loads(job.marker.read_text(encoding='utf-8')).get('command') == self.signature(job):
                job.state = 'skipped'

    @staticmethod
    def signature(job: Job) -> str:
        """The command with the GPU left out: a finished job is the same job on any GPU."""
        return ' '.join(job.command('<gpu>' if job.units else None))

    def finished(self, name: str) -> bool:
        return self.jobs[name].state in ('done','skipped')

    def blocked(self, job: Job) -> bool:
        return any(self.jobs[d].state == 'failed' or (self.jobs[d].state == 'cancelled') for d in job.deps)

    def ready(self) -> list[Job]:
        return sorted((j for j in self.jobs.values() if j.state == 'waiting' and all(self.finished(d) for d in j.deps)),
                      key=Job.key)

    def start(self, job: Job, gpu: str | None) -> None:
        job.state = 'running'
        if gpu is None:
            self.cpu_busy = True
        else:
            self.free[gpu] -= job.units
        self.events.append(('start',job.name,gpu))
        log(f"start {job.name}"+(f' on GPU {gpu} ({job.units}/{UNITS})' if gpu is not None else ' (CPU)'))
        def body() -> None:
            began = time.time()
            status = self.execute(job.command(gpu),gpu,job.log)
            with self.lock:
                if gpu is None:
                    self.cpu_busy = False
                else:
                    self.free[gpu] += job.units
                if status == 0:
                    job.state = 'done'
                    if job.marker:
                        job.marker.parent.mkdir(parents=True,exist_ok=True)
                        job.marker.write_text(json.dumps({'command':self.signature(job),'gpu':gpu,
                                                          'seconds':round(time.time()-began)}),encoding='utf-8')
                    self.events.append(('done',job.name,gpu))
                    log(f'done  {job.name} in {(time.time()-began)/60:.1f} min')
                elif job.units and job.units < UNITS and not job.retried and oom(job.log):
                    job.units,job.retried,job.state = UNITS,True,'waiting'
                    self.events.append(('oom-retry',job.name,gpu))
                    log(f'{job.name}: out of GPU memory beside another job; retrying on a whole GPU')
                else:
                    job.state = 'failed'
                    self.events.append(('failed',job.name,gpu))
                    log(f'FAILED {job.name} (exit {status}); see {job.log}')
        threading.Thread(target=body,daemon=True).start()

    def run(self) -> bool:
        while True:
            with self.lock:
                for job in self.jobs.values():
                    if job.state == 'waiting' and self.blocked(job):
                        job.state = 'cancelled'
                        log(f'skip  {job.name}: a job it needs failed')
                for job in self.ready():
                    if job.units == 0:
                        if not self.cpu_busy:
                            self.start(job,None)
                        continue
                    # The GPU with the least room that still fits keeps whole GPUs free for collection.
                    fits = sorted((u,g) for g,u in self.free.items() if u >= job.units)
                    if fits:
                        self.start(job,fits[0][1])
                active = [j for j in self.jobs.values() if j.state in ('waiting','running')]
                if not active:
                    return all(j.state in ('done','skipped') for j in self.jobs.values())
                if not any(j.state == 'running' for j in active) and not self.ready():
                    for j in active:
                        j.state = 'cancelled'
                    return False
            time.sleep(self.poll)


def build_jobs(args: argparse.Namespace, gpu_gb: float) -> list[Job]:
    from chunkratio.riskkd import RISKKD, memory_plan
    py,jobs = sys.executable,[]
    common = ['--scale',args.scale,'--root',args.root,'--variant','crpd_raw','--alpha',str(args.alpha),
              '--passes',str(args.passes),'--selections',str(args.selections),'--gpu-memory-gb',str(gpu_gb),
              *(['--skip-verify'] if args.skip_verify else []),*shlex.split(args.run_pair_args or '')]
    hub = (['--hub-namespace',args.hub_namespace]+(['--hub-public'] if args.hub_public else [])+
           (['--hub-allow-pilot'] if args.hub_allow_pilot else [])) if args.hub_namespace else []
    for rank,item in enumerate(args.pair):
        parts = item.split(':')
        pair,gammas = parts[0],[g for g in (parts[1] if len(parts) > 1 else '0.05').split(',') if g]
        sources = parts[2] if len(parts) > 2 else args.pair_sources
        if pair not in RISKKD:
            raise SystemExit(f'unknown pair {pair}; choose from {sorted(RISKKD)}')
        work = Path(args.root)/pair/args.scale
        logs,marks = work/'logs',work/'schedule'
        base = [py,'script/server/run_pair.py','--pair',pair,'--pair-sources',sources,*common]
        per_gpu = memory_plan(pair,gpu_gb)['trainings_per_gpu']
        def job(kind: str, detail: str, command: Callable[[str | None],list[str]], units: int, deps: list[str],
                order: int = 0) -> str:
            name = f'{pair}/{kind}'+(f'/{detail}' if detail else '')
            slug = name.replace('/','_')
            jobs.append(Job(name,kind,rank,order,command,units,deps,logs/f'sched-{slug}.log',marks/f'{slug}.json'))
            return name
        all_g = ','.join(gammas)
        prep = job('prep','',lambda gpu,b=base,a=all_g: b+['--gpus',args.gpus,'--gammas',a,'--stages','prepare,verify'],0,[])
        shards = [job('collect',str(i),lambda gpu,b=base,a=all_g,i=i: b+['--gpus',gpu,'--gammas',a,'--stages','collect',
                      '--collect-shards',str(args.collect_shards),'--shard-indices',str(i)],UNITS,[prep],i)
                  for i in range(args.collect_shards)]
        heldouts = []
        for order,g in enumerate(gammas):
            one = ['--gammas',g,'--trainings-per-gpu','1']
            train = job('train',g,lambda gpu,b=base,o=one: b+['--gpus',gpu,*o,'--stages','train'],
                        max(1,UNITS//per_gpu),shards,order)
            heldouts.append(job('heldout',g,lambda gpu,b=base,o=one: b+['--gpus',gpu,*o,'--stages','heldout'],1,[train],order))
        final = list(heldouts)
        if args.eval_python:
            out = str(work/'benchmarks')
            base_bench = job('bench-base','',lambda gpu,p=pair,o=out: [args.eval_python,'script/server/benchmarks.py','--pair',p,
                             '--gpus',gpu,'--output',o],1,[prep])
            final = [job('bench','',lambda gpu,b=base,a=all_g: b+['--gpus',gpu,'--gammas',a,'--trainings-per-gpu','1',
                         '--stages','benchmarks','--eval-python',args.eval_python],1,heldouts+[base_bench])]
        if hub:
            job('push','',lambda gpu,b=base,a=all_g: b+['--gpus',args.gpus,'--gammas',a,'--stages','push',*hub],0,final)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pair',action='append',required=True,
                        help='pair:gamma[,gamma...][:pair_sources]; repeat; order = priority (longest training first)')
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--scale',default='full')
    parser.add_argument('--pair-sources',default='within',choices=('all','within','student_ref'))
    parser.add_argument('--alpha',type=float,default=0.03)
    parser.add_argument('--passes',type=float,default=1.0)
    parser.add_argument('--selections',type=int,default=3)
    parser.add_argument('--collect-shards',type=int,default=6,
                        help='Collection jobs per pair (more: finer gap filling, one model load each)')
    parser.add_argument('--root',default='output/server')
    parser.add_argument('--eval-python',default=None,help='lm-eval environment; without it no benchmark jobs')
    parser.add_argument('--hub-namespace',default=None,help='Push each pair after its benchmarks (private)')
    parser.add_argument('--hub-public',action='store_true')
    parser.add_argument('--hub-allow-pilot',action='store_true')
    parser.add_argument('--skip-verify',action='store_true',help='Smoke only')
    parser.add_argument('--run-pair-args',default=None,help='Extra run_pair.py arguments for every job (quoted)')
    parser.add_argument('--gpu-memory-gb',type=float,default=None,help='Default: nvidia-smi')
    parser.add_argument('--fresh',action='store_true',help='Ignore the finished-job marks')
    parser.add_argument('--dry-run',action='store_true',help='List the jobs and their dependencies only')
    args = parser.parse_args()
    gpus = [g for g in args.gpus.split(',') if g]
    if args.collect_shards < 1 or not gpus:
        raise SystemExit('need --collect-shards >= 1 and at least one GPU')
    if args.gpu_memory_gb:
        gpu_gb = args.gpu_memory_gb
    elif args.dry_run:
        gpu_gb = 179.
    else:
        sys.path.insert(0,str(ROOT/'script'/'server'))
        from run_pair import gpu_memory_gb
        gpu_gb = gpu_memory_gb(gpus)
    jobs = build_jobs(args,gpu_gb)
    if args.dry_run:
        flow = ['prep','collect','train','heldout','bench-base','bench','push']
        for j in sorted(jobs,key=lambda j:(j.pair_rank,flow.index(j.kind),j.order)):
            where = f'GPU {j.units}/{UNITS}' if j.units else 'CPU'
            log(f"{j.name:<48} {where:<9} after {', '.join(j.deps) or '-'}")
            log(f"    {' '.join(j.command('<gpu>' if j.units else None))}")
        return 0
    def execute(command: list[str], gpu: str | None, logfile: Path | None) -> int:
        logfile.parent.mkdir(parents=True,exist_ok=True)
        with open(logfile,'a',encoding='utf-8') as out:
            out.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(command)}\n")
            out.flush()
            return subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT).returncode
    began = time.time()
    ok = Scheduler(jobs,gpus,execute,fresh=args.fresh,poll=5.).run()
    states = {j.name:j.state for j in jobs}
    log(f"{'all jobs finished' if ok else 'some jobs failed'} in {(time.time()-began)/3600:.2f} h: "+
        json.dumps({s:sum(v == s for v in states.values()) for s in sorted(set(states.values()))}))
    for name,state in states.items():
        if state in ('failed','cancelled'):
            log(f'  {state}: {name}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
