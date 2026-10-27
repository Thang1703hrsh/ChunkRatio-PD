"""Modal GPU check: CRPD on CUDA, first with tiny Qwen3-style models, then with real Qwen3 weights.

A functional check, not a result. Part B uses Qwen3-1.7B-Base as T0, Qwen3-1.7B (post-trained,
not a DPO of T0) as T+ and Qwen3-0.6B-Base as rho, 24 hand-written prompts, H=64, L=16 and
20 policy steps with the tuned pilot settings. Selection is the synthetic pair-BCE rule (no RM).

Usage (repository root, Python with `modal`; one L40S, hard 30-minute cap):

  modal run script/modal/crpd_gpu_check.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = 'crpd-gpu-check'
REMOTE_REPO = '/root/repo'
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_REPO)
VOLUME_PATH = '/vol'

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME,create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0','transformers==4.52.4','numpy','safetensors','tokenizers')
         .env({'HF_HOME':f'{VOLUME_PATH}/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'})
         .add_local_dir(REPO,REMOTE_REPO,ignore=['output','data','.git','mtbench','winrate_eval','diversity_metrics',
                                                 'promt','prompts','env','reports','**/__pycache__','.pytest_cache']))

PROMPTS = [
    ('Explain why the sky looks blue during the day.','Sunlight scatters off air molecules, and blue light scatters most, so the sky looks blue.','Because it reflects the ocean.'),
    ('Give three tips for staying focused while studying.','Remove distractions, work in short timed blocks, and take brief breaks.','Just study harder.'),
    ('What is the difference between a list and a tuple in Python?','Lists are mutable and tuples are immutable; tuples can be dictionary keys.','There is no difference.'),
    ('Write a short haiku about autumn leaves.','Red leaves drift and fall / the cold wind carries them home / quiet earth below','Leaves are red in fall.'),
    ('How do vaccines help the immune system?','They expose it to a harmless antigen so it builds memory cells before real infection.','They kill all germs.'),
    ('Summarize the water cycle in two sentences.','Water evaporates, condenses into clouds and falls as precipitation. It then collects and evaporates again.','Rain happens.'),
    ('Suggest a healthy breakfast for a busy morning.','Greek yogurt with fruit and oats is quick, filling and high in protein.','Skip breakfast.'),
    ('Why is regular exercise important?','It strengthens the heart, improves mood and lowers the risk of chronic disease.','It is not important.'),
    ('Translate "good morning" into French and Spanish.','French: "bonjour". Spanish: "buenos dias".','Hello.'),
    ('What causes the seasons on Earth?','The tilt of Earth\'s axis changes how directly sunlight hits each hemisphere through the year.','Distance from the sun.'),
    ('Give an example of a renewable energy source.','Solar power, which converts sunlight into electricity.','Coal.'),
    ('How can I improve my sleep quality?','Keep a regular schedule, avoid screens before bed and keep the room dark and cool.','Drink coffee at night.'),
    ('Explain what a prime number is.','A whole number greater than 1 whose only divisors are 1 and itself.','An even number.'),
    ('Describe a polite way to decline an invitation.','Thank them, say you cannot attend this time, and wish them a great event.','Ignore the message.'),
    ('What is photosynthesis?','The process plants use to turn light, water and carbon dioxide into sugar and oxygen.','Plants eating soil.'),
    ('List two benefits of reading books.','Reading builds vocabulary and improves concentration.','None.'),
    ('How does a refrigerator keep food cold?','A refrigerant absorbs heat inside and releases it outside through compression and expansion.','It makes cold air from ice.'),
    ('Why should passwords be long?','Longer passwords have far more combinations, so they are much harder to guess.','Short is safer.'),
    ('What is the capital of Japan, and what is it known for?','Tokyo, known for its technology, food and mix of modern and traditional culture.','Kyoto is the capital.'),
    ('Give advice for a first job interview.','Research the company, prepare examples of your work and arrive a little early.','Be late on purpose.'),
    ('Explain gravity to a child.','Gravity is an invisible pull that makes things fall down and keeps us on the ground.','Gravity is magic.'),
    ('How do I make a simple cup of tea?','Boil water, steep the tea bag for three to five minutes, then remove it.','Use cold water.'),
    ('What is the main idea of recycling?','Turning used materials into new products to save resources and reduce waste.','Throwing things away.'),
    ('Why do we need to drink water?','Water carries nutrients, regulates temperature and keeps organs working.','We do not need it.'),
]


def ultrafeedback_rows(path: Path) -> None:
    """Hand-written UltraFeedback-format rows (not a benchmark)."""
    lines = []
    for i,(prompt,good,bad) in enumerate(PROMPTS):
        user = {'role':'user','content':prompt}
        lines.append(json.dumps({'prompt_id':f'p{i}','prompt':prompt,'chosen':[user,{'role':'assistant','content':good}],
                                 'rejected':[user,{'role':'assistant','content':bad}],'score_chosen':8.,'score_rejected':2.}))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def timed(log: dict, name: str, function, *args):
    """Run one step and record its wall time."""
    start = time.perf_counter()
    value = function(*args)
    log[name] = round(time.perf_counter()-start,1)
    print(f'[{name}] {log[name]} s',flush=True)
    return value


@app.function(image=image,gpu='L40S',cpu=4.0,memory=32768,timeout=30*60,volumes={VOLUME_PATH:volume})
def check(revision: str) -> dict:
    """Part A (tiny Qwen3, CUDA) and part B (real Qwen3 weights, CUDA); returns measured facts."""
    import torch
    work = Path('/root/work')
    work.mkdir(exist_ok=True)
    for name in ('chunkratio','config','reference','script'):
        if not (work/name).exists():
            (work/name).symlink_to(Path(REMOTE_REPO)/name)
    os.chdir(work)
    sys.path.insert(0,str(work))
    os.environ['CRPD_CODE_REVISION'] = revision
    from chunkratio.artifacts import read_json
    from chunkratio.config import validate
    from chunkratio.data import prepare_data
    from chunkratio.diagnostics import diagnose
    from chunkratio.models import load_models
    from chunkratio.pilot import POLICY_TUNING
    from chunkratio.pipeline import pipeline
    from chunkratio.sampling import generate, stream
    from chunkratio.tiny import build_tiny_qwen3, synthetic_ultrafeedback
    report = {'revision':revision,'gpu':torch.cuda.get_device_name(0),'times_s':{}}
    times = report['times_s']

    # Part A: the narval-tiny smoke stages, on CUDA.
    tiny = Path('output/gpu-check/tiny')
    paths = build_tiny_qwen3(tiny/'models')
    synthetic_ultrafeedback(tiny/'data'/'uf_train.jsonl')
    prepare_data(tiny/'data'/'uf_train.jsonl',str(paths['student_ref']),'local',tiny/'data'/'train.jsonl',0,
                 template_kwargs={'enable_thinking':False},head_val_fraction=.2,select_fraction=.1)
    cfg = read_json(Path('config/crpd/narval/qwen3_1p7b_to_0p6b.json'))
    cfg.update(synthetic=True,device='cuda',horizon=8,chunk_length=2,batch_size=4,steps=2,head_epochs=2,
               head_batch_size=2,generation_batch_size=4,states_per_prompt=2,selection_interval=1,
               prompts=str(tiny/'data'/'train.jsonl'),cache_root=str(tiny/'cache'),**POLICY_TUNING,
               models={role:{'path':str(path),'revision':'local'} for role,path in paths.items()})
    validate(cfg)
    timed(times,'A_diagnose',diagnose,cfg,2,4,tiny/'diagnostics')
    report['A_tiny_cuda'] = {}
    for variant in ('crpd_main','crpd_raw','crpd_anchor'):
        run = {**cfg,'variant':variant,'output':str(tiny/variant)}
        result = timed(times,f'A_{variant}',pipeline,run,'evaluate')
        assert result['route'] == 'reference_mean' and result['policy']['optimizer_steps'] == run['steps']
        report['A_tiny_cuda'][variant] = {'route':result['route'],'steps':result['policy']['optimizer_steps'],
                                          'pairs':result['counts']['pairs']}

    # Part B: real Qwen3 weights, small scale.
    real = Path('output/gpu-check/qwen3')
    ultrafeedback_rows(real/'data'/'uf.jsonl')
    tokenizer = 'Qwen/Qwen3-0.6B-Base'
    timed(times,'B_prepare_data',prepare_data,real/'data'/'uf.jsonl',tokenizer,'main',real/'data'/'train.jsonl',0,
          'ultrafeedback',None,{'enable_thinking':False},.2,.1,'<|im_end|>')
    # Hub Base tokenizers declare <|endoftext|>; the pilot's SFT checkpoints save theirs with
    # <|im_end|> as EOS, so do the same here instead of loosening the preflight.
    from transformers import AutoTokenizer
    shared = AutoTokenizer.from_pretrained(tokenizer,revision='main')
    shared.eos_token = '<|im_end|>'
    shared.save_pretrained(real/'tokenizer')
    tokenizer = str(real/'tokenizer')
    roles = {'teacher_plus':'Qwen/Qwen3-1.7B','teacher_base':'Qwen/Qwen3-1.7B-Base','student_ref':'Qwen/Qwen3-0.6B-Base'}
    cfg = read_json(Path('config/crpd/narval/qwen3_1p7b_to_0p6b.json'))
    cfg.update(synthetic=True,device='cuda',horizon=64,chunk_length=16,batch_size=16,steps=20,head_epochs=5,
               head_batch_size=8,generation_batch_size=16,states_per_prompt=2,selection_interval=10,
               prompts=str(real/'data'/'train.jsonl'),cache_root=str(real/'cache'),**POLICY_TUNING,
               models={role:{'path':path,'revision':'main','tokenizer':tokenizer,'tokenizer_revision':'local'}
                       for role,path in roles.items()})
    validate(cfg)
    torch.cuda.reset_peak_memory_stats()
    report['B_diagnostic'] = timed(times,'B_diagnose',diagnose,cfg,4,4,real/'diagnostics')
    run = {**cfg,'variant':'crpd_main','output':str(real/'crpd_main')}
    result = timed(times,'B_crpd_main_all_stages',pipeline,run,'evaluate')
    volume.commit()
    history = list(result['policy'].get('history',[]))
    report['B_crpd_main'] = {'route':result['route'],'counts':result['counts'],
                             'optimizer_steps':result['policy']['optimizer_steps'],'history':history,
                             'peak_cuda_memory_gb':round(torch.cuda.max_memory_allocated()/2**30,2)}
    assert result['route'] == 'reference_mean' and result['policy']['optimizer_steps'] == run['steps']
    assert all(torch.isfinite(torch.tensor(h['loss'])) for h in history)

    # Standalone export: reload the trained student and sample next to rho (same RNG stream).
    models,student,tokenizers,info = load_models(run)
    student.load_state_dict(torch.load(real/'crpd_main'/'student.pt',map_location='cuda',weights_only=True)['state_dict'])
    tok = tokenizers[2]
    prompt = tok.apply_chat_template([{'role':'user','content':'Give two tips for learning a new language.'}],
                                     tokenize=True,add_generation_prompt=True,enable_thinking=False)
    samples = {}
    for name,model in (('rho',models['student_ref']),('crpd_student',student)):
        tokens = generate(model,prompt,64,info['eos'],stream(7,'gpu-check-sample'))
        samples[name] = {'text':tok.decode(tokens),'tokens':len(tokens),'ended_with_eos':tokens[-1] == info['eos']}
    report['B_samples'] = samples
    report['status'] = 'passed'
    print('GPU CHECK PASSED',flush=True)
    return report


@app.function(image=image,gpu='L40S',cpu=4.0,memory=32768,timeout=15*60,volumes={VOLUME_PATH:volume})
def profile(revision: str, horizon: int) -> dict:
    """Per-stage wall time on real Qwen3 weights with the pilot's L=32, K=4 and 4 states per prompt."""
    import torch
    work = Path('/root/work')
    work.mkdir(exist_ok=True)
    for name in ('chunkratio','config','reference','script'):
        if not (work/name).exists():
            (work/name).symlink_to(Path(REMOTE_REPO)/name)
    os.chdir(work)
    sys.path.insert(0,str(work))
    os.environ['CRPD_CODE_REVISION'] = revision
    import chunkratio.pipeline as stages_module
    from transformers import AutoTokenizer
    from chunkratio.artifacts import read_json
    from chunkratio.config import validate
    from chunkratio.data import prepare_data
    from chunkratio.pilot import POLICY_TUNING
    seconds = {}
    def timer(name, function):
        """Accumulate synchronized wall time per stage and print it as soon as the stage ends."""
        def wrapped(*args, **kwargs):
            """Timed call."""
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = function(*args,**kwargs)
            torch.cuda.synchronize()
            seconds[name] = round(seconds.get(name,0.)+time.perf_counter()-start,1)
            print(json.dumps({'stage':name,'seconds':seconds[name],
                              'peak_gb':round(torch.cuda.max_memory_allocated()/2**30,2)}),flush=True)
            return value
        return wrapped
    for name in ('load_models','eligible_prompts','collect_prefixes','collect_candidates','collect_correction',
                 'fit_head','build_targets'):
        setattr(stages_module,name,timer(name,getattr(stages_module,name)))
    stages_module.ChunkRatioTrainer.train = timer('train_policy',stages_module.ChunkRatioTrainer.train)
    real = Path(f'output/gpu-profile/h{horizon}')
    ultrafeedback_rows(real/'data'/'uf.jsonl')
    shared = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base',revision='main')
    shared.eos_token = '<|im_end|>'
    shared.save_pretrained(real/'tokenizer')
    tokenizer = str(real/'tokenizer')
    prepare_data(real/'data'/'uf.jsonl',tokenizer,'local',real/'data'/'train.jsonl',0,
                 'ultrafeedback',None,{'enable_thinking':False},.2,.1,'<|im_end|>')
    roles = {'teacher_plus':'Qwen/Qwen3-1.7B','teacher_base':'Qwen/Qwen3-1.7B-Base','student_ref':'Qwen/Qwen3-0.6B-Base'}
    cfg = read_json(Path('config/crpd/narval/qwen3_1p7b_to_0p6b.json'))
    cfg.update(synthetic=True,device='cuda',horizon=horizon,batch_size=16,steps=10,head_epochs=20,head_batch_size=64,
               generation_batch_size=64,selection_interval=10,prompts=str(real/'data'/'train.jsonl'),
               cache_root=str(real/'cache'),variant='crpd_main',output=str(real/'crpd_main'),**POLICY_TUNING,
               models={role:{'path':path,'revision':'main','tokenizer':tokenizer,'tokenizer_revision':'local'}
                       for role,path in roles.items()})
    validate(cfg)
    start = time.perf_counter()
    result = stages_module.pipeline(cfg,'evaluate')
    total = time.perf_counter()-start
    report = {'gpu':torch.cuda.get_device_name(0),'horizon':horizon,
              'settings':{k:cfg[k] for k in ('chunk_length','candidates','states_per_prompt','correction_replicates',
                                             'endpoint_fraction','batch_size','steps','policy_microbatch_size',
                                             'generation_batch_size','head_epochs')},
              'train_prompts':sum(1 for line in open(cfg['prompts'],encoding='utf-8') if '"train"' in line),
              'stage_seconds':seconds,'pipeline_seconds':round(total,1),'counts':result['counts'],
              'cost':result['cost'],'last_loss':result['policy']['history'][-1]}
    print('PROFILE DONE',flush=True)
    return report


@app.local_entrypoint()
def main(mode: str = 'check', horizon: int = 128) -> None:
    """Run the check (default) or `--mode profile --horizon H` once and print its report."""
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    report = check.remote(revision) if mode == 'check' else profile.remote(revision,horizon)
    print(json.dumps(report,indent=2,ensure_ascii=False))
