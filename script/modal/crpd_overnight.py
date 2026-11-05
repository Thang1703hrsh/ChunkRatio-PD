"""Budgeted, unattended CRPD smoke suite. No model uploads, no test-based selection.

Each GPU call has an absolute deadline and zero application retries. A CPU-only
non-preemptible controller reserves worst-case runtime before submitting work.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

if not modal.is_local():
    sys.path.insert(0,'/code')
from script.modal import crpd_fast_x10 as fast
from script.overnight_plan import SMOKE_TRIALS, configure_trial, promotion_candidates, reserve, validation_delta

APP_NAME = 'crpd-overnight-budget'
app = modal.App(APP_NAME)
image = fast.image.add_local_dir(fast.REPO/'script','/code/script',ignore=['__pycache__'])
eval_image = (fast.eval_image.add_local_dir(fast.REPO/'script','/code/script',ignore=['__pycache__'])
              .add_local_dir(fast.REPO/'chunkratio','/code/chunkratio',ignore=['__pycache__'])
              .add_local_dir(fast.REPO/'config','/code/config')
              .add_local_dir(fast.REPO/'reference','/code/reference')
              .add_local_dir(fast.REPO/'tests','/code/tests',ignore=['__pycache__']))


def utc_now() -> float:
    return time.time()


def run_process(args: list[str], log: Path, deadline: float) -> dict:
    """Terminate the actual GPU subprocess on expiry, including after a preemption replay."""
    remaining = deadline-utc_now()
    if remaining < 10:
        return {'status':'deadline_expired'}
    log.parent.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    with log.open('a',encoding='utf-8') as handle:
        process = subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=remaining)
            status = 'success' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code,status = None,'timeout'
    return {'status':status,'returncode':code,'wall_seconds':time.perf_counter()-start,
            'log_tail':log.read_text(encoding='utf-8',errors='replace')[-3000:]}


@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(65536,65536),timeout=1230,
              startup_timeout=120,retries=0,max_containers=1,single_use_containers=True,volumes=fast.mounts)
def trial(revision: str, cfg_path: str, deadline: float) -> dict:
    fast.work(revision)
    from chunkratio.artifacts import code_revision, digest, read_json, write_json
    from chunkratio.fast_run import output_fingerprints
    cfg = read_json(Path(cfg_path))
    output = Path(cfg['output'])
    output.mkdir(parents=True,exist_ok=True)
    marker = output/'trial_completed.json'
    try:
        if marker.exists():
            done = read_json(marker)
            if done['config_hash'] != digest(cfg) or done['code_revision'] != code_revision():
                raise ValueError('incompatible completed trial')
            if done['artifacts'] != output_fingerprints(output):
                raise ValueError('completed artifacts changed')
            return done['result']
        result = run_process([sys.executable,'-m','chunkratio','evaluate','--config',cfg_path],
                             output/'execution.log',deadline)
        result.update(variant=cfg['variant'],output=cfg['output'])
        if result['status'] == 'success':
            data = read_json(output/'diagnostics.json')
            result.update(policy={k:v for k,v in data['policy'].items() if k != 'history'},counts=data['counts'])
            write_json(marker,{'config_hash':digest(cfg),'code_revision':code_revision(),
                              'artifacts':output_fingerprints(output),'result':result})
        write_json(output/'execution_result.json',result)
        return result
    finally:
        fast.volume.commit()


@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(65536,65536),timeout=2730,
              startup_timeout=120,retries=0,max_containers=1,single_use_containers=True,volumes=fast.mounts)
def medium_trial(revision: str, cfg_path: str, deadline: float) -> dict:
    return trial.local(revision,cfg_path,deadline)


@app.function(image=eval_image,gpu='L40S',cpu=(2,2),memory=(24576,24576),timeout=1230,
              startup_timeout=120,retries=0,max_containers=1,single_use_containers=True,
              volumes={'/vol':fast.volume})
def evaluate_benchmarks(root: str, name: str, model: str, deadline: float) -> dict:
    fast.volume.reload()
    results = {}
    for task,shots in (('winogrande',5),('truthfulqa_mc2',0)):
        out = Path('/vol')/root/'benchmarks'/name/task
        command=[sys.executable,'-m','lm_eval','--model','hf','--model_args',f'pretrained={model},dtype=float32',
                 '--tasks',task,'--num_fewshot',str(shots),'--batch_size','32','--device','cuda',
                 '--log_samples','--output_path',str(out)]
        result=run_process(command,out/'execution.log',deadline)
        if result['status']=='success':
            files=sorted(out.rglob('results_*.json'))
            result['metrics']=json.loads(files[-1].read_text())['results'][task]
        results[task]=result
        fast.volume.commit()
        if result['status']!='success':
            break
    return results


@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(65536,65536),timeout=1230,
              startup_timeout=120,retries=0,max_containers=1,single_use_containers=True,volumes=fast.mounts)
def evaluate_rm(revision: str, root: str, models: dict, deadline: float) -> dict:
    fast.work(revision)
    # Run as a subprocess so the fixed deadline also covers downloads and RM scoring.
    program = """import json,gc
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from chunkratio.artifacts import read_json,read_jsonl,write_json
from chunkratio.evaluation import reward_selector
root=Path(__import__('sys').argv[1]); entries=json.loads(__import__('sys').argv[2])
cfg=read_json(root/'rm_config.json'); rows=read_jsonl(root/'test_prompts.jsonl')
rho=cfg['models']['student_ref']; tok=AutoTokenizer.from_pretrained(rho['path'],revision=rho['revision'])
tok.eos_token=cfg['eos_token']
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
selector,cost=reward_selector(cfg,[{**r,'split':'select'} for r in rows],tok)
results={}
for name,entry in entries.items():
 model=AutoModelForCausalLM.from_pretrained(entry['path'],revision=entry.get('revision'),torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
 score=selector(model); results[name]={'mean':score,'scores':selector.last_scores.copy()}
 write_json(root/'rm_partial.json',{'split':'test','results':results,'n':len(rows)})
 del model;gc.collect();torch.cuda.empty_cache()
rng=np.random.default_rng(0); ix=rng.integers(0,len(rows),(10000,len(rows))); comparisons={}
for name in results:
 if name=='rho':continue
 delta=np.array(results[name]['scores'])-np.array(results['rho']['scores'])
 comparisons[name]={'mean_delta':float(delta.mean()),'paired_bootstrap_95':np.quantile(delta[ix].mean(axis=1),[.025,.975]).tolist()}
write_json(root/'heldout_rm.json',{'split':'test','results':results,'comparisons':comparisons,'n':len(rows),'cost':cost,'teacher_test_exposure':'unverified','not_used_for_selection':True})
"""
    try:
        result=run_process([sys.executable,'-c',program,root,json.dumps(models)],Path(root)/'rm_execution.log',deadline)
        from chunkratio.artifacts import read_json
        if result['status']=='success':
            result['metrics']=read_json(Path(root)/'heldout_rm.json')
        return result
    finally:
        fast.volume.commit()


@app.function(image=image,cpu=(.25,.25),memory=(1024,1024),timeout=8*3600,
              startup_timeout=120,retries=0,nonpreemptible=True,single_use_containers=True,volumes=fast.mounts)
def overnight(revision: str, nonce: str, limit_cents: int = 1455, allow_compute: bool = False,
              wait_for_smoke_app: bool = True) -> dict:
    """Reserve before submit; serial calls, fixed deadlines, no automatic restarts/sweeps."""
    fast.work(revision)
    from chunkratio.artifacts import atomic_bytes, read_json, read_jsonl, source_digest, write_json
    from chunkratio.pilot import pilot_subset
    if not 0 < limit_cents <= 2400:
        raise ValueError('new-compute reservation must be at most USD24')
    base_root=Path('output/fast_hda_x10')/source_digest()[:16]
    # Reuse immutable inputs across reviewed code revisions, never a scoring/cache namespace.
    if not (base_root/'base_config.json').exists():
        previous=Path('output/fast_hda_x10/e78dfc27a2e939b9')
        base_root.mkdir(parents=True,exist_ok=True)
        for name in ('base_config.json','input_prompts.jsonl','test_prompts.jsonl'):
            atomic_bytes(base_root/name,(previous/name).read_bytes())
    root=base_root/'overnight_smoke_v1'
    root.mkdir(parents=True,exist_ok=True)
    ledger_path=root/'budget.json'
    if ledger_path.exists():
        return {'status':'already_submitted','ledger':read_json(ledger_path)}
    base=read_json(base_root/'base_config.json')
    full=read_jsonl(base_root/'input_prompts.jsonl')
    for level,sizes in (('smoke',{'train':64,'head_val':16,'select':16}),
                        ('medium',{'train':256,'head_val':64,'select':32})):
        rows=pilot_subset(full,0,sizes)
        atomic_bytes(root/level/'prompts.jsonl',(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)).encode())
    test=read_jsonl(base_root/'test_prompts.jsonl')[:32]
    atomic_bytes(root/'test_prompts.jsonl',(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in test)).encode())
    write_json(root/'rm_config.json',base)
    ledger={'nonce':nonce,'limit_cents':limit_cents,'reserved_cents':0,'jobs':{},
            'past_and_other_apps':'not controlled by this ledger','status':'planned',
            'deadline_utc':dt.datetime.fromtimestamp(utc_now()+7*3600,dt.timezone.utc).isoformat(),
            'reservation_note':'full timeout +180s; USD4.50/h A100 and USD3/h L40S; no refunds',
            'provider_billing_cap_set':False,'results':{},'promoted':{}}
    write_json(root/'plan.json',{'trials':SMOKE_TRIALS,'data':'64/16/16 smoke; optional 256/64/32 medium',
               'selection':'select-split RM only; promote positive gain, exploratory; test reporting afterward',
               'source_digest':source_digest(),'limit_cents':limit_cents,'allow_compute':allow_compute})
    if not allow_compute:
        fast.volume.commit()
        return {'status':'plan_only','root':str(root),'reason':'compute requires explicit launch flag after shared-budget check'}
    ledger['status']='waiting_for_other_smoke' if wait_for_smoke_app else 'running'
    write_json(ledger_path,ledger);fast.volume.commit()
    global_deadline=utc_now()+7*3600
    if wait_for_smoke_app:
        wait_until=utc_now()+3600
        functions=['run_pilot','finish','train_variants','diagnose','verify_hda_teachers','prepare']
        while utc_now()<wait_until:
            state={}
            for name in functions:
                try:
                    stats=modal.Function.from_name('crpd-qwen3-pilot-hda-smoke',name).get_current_stats()
                    state[name]={'running':stats.num_running_inputs,'backlog':stats.backlog}
                except Exception as error:
                    state[name]={'running':1,'error':str(error)[:200]}
            ledger['other_smoke_state']=state
            write_json(ledger_path,ledger);fast.volume.commit()
            if all(row['running']==0 and row.get('backlog',0)==0 for row in state.values()):break
            time.sleep(30)
        else:
            ledger['status']='stopped_other_smoke_still_active'
            write_json(ledger_path,ledger);fast.volume.commit()
            return {'root':str(root),'status':ledger['status'],'reserved_cents':0}
    ledger['status']='running'
    write_json(ledger_path,ledger);fast.volume.commit()
    def submit(key,seconds,gpu,function,args):
        if utc_now()+seconds+180>global_deadline:
            return {'status':'skipped_deadline'}
        try:reserve(ledger,key,seconds,gpu)
        except ValueError as error:return {'status':'skipped_budget','reason':str(error)}
        entry=ledger['jobs'][key]
        entry['absolute_deadline']=utc_now()+seconds
        write_json(ledger_path,ledger);fast.volume.commit()
        call=function.spawn(*args,entry['absolute_deadline'])
        entry.update(call_id=call.object_id,status='running')
        write_json(ledger_path,ledger);fast.volume.commit()
        try:
            result=call.get(timeout=max(1,entry['absolute_deadline']-utc_now()))
        except TimeoutError:
            call.cancel(terminate_containers=True)
            result={'status':'timeout_cancelled'}
        except Exception as error:
            call.cancel(terminate_containers=True)
            result={'status':'failed','error':str(error)[:2000]}
        fast.volume.reload()
        entry['status']=result.get('status','completed')
        entry['completed_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(ledger_path,ledger);fast.volume.commit()
        return result
    # Under USD15 reserve room for final evaluation; K4 cases are optional under the larger allowance.
    chosen_trials=list(SMOKE_TRIALS) if limit_cents>=2200 else [SMOKE_TRIALS[i] for i in (0,1,2,5)]
    for spec in chosen_trials:
        cfg=configure_trial(base,'smoke',dict(spec),root.as_posix())
        path=root/'configs'/f"smoke_{spec['name']}.json";write_json(path,cfg);fast.volume.commit()
        result=submit('smoke_'+spec['name'],1200,'A100-80GB',trial,(revision,str(path)))
        result['variant']=spec['variant']
        ledger['results'][spec['name']]=result
        write_json(ledger_path,ledger);fast.volume.commit()
    promoted=promotion_candidates(ledger['results'])
    if limit_cents<2200:
        promoted=sorted(promoted,key=lambda n:validation_delta(ledger['results'][n]),reverse=True)[:1]
    for name in promoted:
        spec=dict(next(s for s in SMOKE_TRIALS if s['name']==name))
        cfg=configure_trial(base,'medium',spec,root.as_posix())
        path=root/'configs'/f'medium_{name}.json';write_json(path,cfg);fast.volume.commit()
        result=submit('medium_'+name,2700,'A100-80GB',medium_trial,(revision,str(path)))
        result['variant']=spec['variant'];ledger['promoted'][name]=result
        write_json(ledger_path,ledger);fast.volume.commit()
    winners={n:r for n,r in ledger['promoted'].items() if validation_delta(r)>0}
    if not winners:
        successes=any(r.get('status')=='success' for r in ledger['results'].values())
        ledger['status']='completed_no_validation_gain' if successes else 'failed_all_smokes'
    else:
        # At most one candidate per family; selected before observing any test metric.
        models={'rho':{k:base['models']['student_ref'][k] for k in ('path','revision')}}
        for name,row in winners.items():models[name]={'path':row['output']+'/student'}
        ledger['heldout_rm']=submit('heldout_rm',1200,'A100-80GB',evaluate_rm,(revision,str(root),models))
        ledger['benchmarks']={}
        for name,entry in models.items():
            model=('/vol/'+entry['path'] if name!='rho' else entry['path']+',revision='+entry['revision'])
            ledger['benchmarks'][name]=submit('benchmark_'+name,1200,'L40S',evaluate_benchmarks,(str(root),name,model))
        ledger['status']=('completed' if all(j['status'] in ('success','completed') for j in ledger['jobs'].values())
                          else 'completed_with_partial_results')
    write_json(ledger_path,ledger)
    write_json(root/'summary.json',ledger)
    fast.volume.commit()
    return {'root':str(root),'status':ledger['status'],'reserved_cents':ledger['reserved_cents'],
            'jobs':{k:v['status'] for k,v in ledger['jobs'].items()}}
