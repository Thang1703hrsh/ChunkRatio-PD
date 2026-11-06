"""Server launcher for the three riskKD (NARVAL) pairs (DECISIONS 93)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chunkratio.config import validate
from chunkratio.pilot import PAIRS
from chunkratio.riskkd import RISKKD, SCALES, server_config, steps_for

REVISIONS = {'reward_model':'r','preference_dataset':'p','sft_dataset':'d','teacher_base_model':'t','student_base_model':'s'}
EVIDENCE = {'relative_l2_distance':{'reference':1e-3,'base':1e-2},'rule':'nearest'}


@pytest.mark.parametrize('pair',sorted(RISKKD))
@pytest.mark.parametrize('variant',['crpd_raw','crpd_main'])
def test_server_configs_validate_for_every_pair(pair: str, variant: str) -> None:
    """Every pair gives a valid config with its pinned trio, end-of-turn EOS, the assumption with evidence, the
    speed settings and the run's choices; raw may use gamma, pair_sources and train-only collection."""
    spec = RISKKD[pair]
    protocol = json.loads(Path(f"config/crpd/narval/{spec['protocol']}.json").read_text(encoding='utf-8-sig'))
    raw = variant == 'crpd_raw'
    cfg = server_config(pair,variant,REVISIONS,'h','data/train.jsonl','out/run','out/cache',alpha=.03,
                        gamma=.05 if raw else 1.,pair_sources='within' if raw else 'all',steps=150,protocol=protocol,
                        collection_splits=['train'] if raw else None,evidence=EVIDENCE)
    validate(cfg)
    assert {r:(m['path'],m['revision']) for r,m in cfg['models'].items()} == \
           {r:(s['repo'],s['revision']) for r,s in spec['checkpoints'].items()}
    assert cfg['eos_token'] == spec['eos'] and cfg['teacher_dpo_reference_verified'] is False
    assert cfg['teacher_dpo_reference_assumption']['evidence']['relative_l2_distance'] == EVIDENCE['relative_l2_distance']
    assert cfg['shared_prefix_scoring'] and cfg['gpu_sampling'] and cfg['states_per_prompt'] == 2
    assert PAIRS[spec['run_pair']]['tokenizer'] == spec['checkpoints']['teacher_base']['repo']
    if raw:
        assert cfg['teacher_gap_scale'] == .05 and cfg['pair_sources'] == 'within' and cfg['collection_splits'] == ['train']


def test_steps_and_scales() -> None:
    """Steps cover the requested passes; x10/x30 share the 20,480 UltraFeedback rows of the Modal pilots."""
    assert steps_for(7119,128,2.7) == 151 and steps_for(10,128,1.) == 1
    assert SCALES['x10']['uf_train_rows'] == SCALES['x30']['uf_train_rows'] == 20480
    assert SCALES['full']['subset']['train'] is None and SCALES['full']['subset']['select'] == 64


@pytest.mark.parametrize('pair',sorted(RISKKD))
def test_run_pair_dry_run_plans_every_stage(pair: str) -> None:
    """The dry run prints one collection worker per GPU and one training run per gamma, without side effects."""
    out = subprocess.run([sys.executable,'script/server/run_pair.py','--pair',pair,'--scale','x10','--gpus','0,1,2',
                          '--gammas','0.1,0.05','--stages','collect,train,heldout','--dry-run','--root','output/dry-run-test'],
                         capture_output=True,text=True,check=True).stdout
    assert out.count('collect-candidates') == 3 and out.count(' evaluate ') == 2 and out.count('heldout_eval.py') == 2
    assert 'GPU 2:' in out and not Path('output/dry-run-test').exists()


def test_memory_plan_scales_with_gpu_memory() -> None:
    """An A100-80GB gets the measured 0.6B settings; a 180 GB GPU gets larger collection budgets and two training
    processes for the small pairs; FP32 8B teachers that do not fit fail clearly (DECISIONS 94)."""
    from chunkratio.riskkd import memory_plan
    a100 = memory_plan('qwen3_1p7b_to_0p6b',80)
    assert (a100['collection']['generation_tokens_per_batch'],a100['collection']['score_tokens_per_batch'],
            a100['collection']['score_max_states'],a100['trainings_per_gpu']) == (163840,16384,48,1)
    b200 = memory_plan('qwen3_1p7b_to_0p6b',180)
    assert b200['collection']['generation_tokens_per_batch'] > 2*163840 and b200['trainings_per_gpu'] == 2
    assert memory_plan('qwen3_8b_to_1p7b',180)['trainings_per_gpu'] == 1
    assert memory_plan('llama31_8b_to_llama32_1b',180)['trainings_per_gpu'] == 2
    for pair in ('qwen3_8b_to_1p7b','llama31_8b_to_llama32_1b'):
        with pytest.raises(ValueError,match='not enough'):
            memory_plan(pair,80)


def test_release_teachers_in_server_configs() -> None:
    """Server configs drop the teachers after targets; the flag is validated."""
    spec = RISKKD['qwen3_8b_to_1p7b']
    protocol = json.loads(Path(f"config/crpd/narval/{spec['protocol']}.json").read_text(encoding='utf-8-sig'))
    cfg = server_config('qwen3_8b_to_1p7b','crpd_raw',REVISIONS,'h','d.jsonl','o','c',alpha=.03,protocol=protocol,
                        evidence=EVIDENCE)
    assert cfg['release_teachers'] is True
    with pytest.raises(ValueError,match='release_teachers'):
        validate({**cfg,'release_teachers':'yes'})


def fake_run(root: Path, selected: str = 'checkpoint', heldout: bool = True) -> Path:
    """A finished run folder with the files push_hub.py reads (tiny fake weights)."""
    run = root/'runs'/'crpd_raw-alpha0p03-gamma0p1-within'/'seed0'
    (run/'student').mkdir(parents=True)
    (run/'student'/'config.json').write_text('{}',encoding='utf-8')
    (run/'student'/'model.safetensors').write_bytes(b'weights')
    (run/'student'/'tokenizer.json').write_text('{}',encoding='utf-8')
    (run/'diagnostics.json').write_text(json.dumps({'policy':{'selected':selected,'optimizer_steps':10,
        'selection_rule':'select_mean_reward','selection_score':-3.,'reference_selection_score':-4.},
        'counts':{'pairs':100}}),encoding='utf-8')
    (run/'resolved_config.json').write_text(json.dumps({'variant':'crpd_raw','alpha':.03,'teacher_gap_scale':.1,
        'pair_sources':'within','batch_size':128,'models':{'student_ref':{'path':'u/rho','revision':'r'}}}),encoding='utf-8')
    (run/'policy_index.json').write_text(json.dumps({'code_revision':'abc+src.1'}),encoding='utf-8')
    if heldout:
        pair = {'rho':1.,'student':1.}
        (run/'heldout.json').write_text(json.dumps({'prompts':64,'rm_mean':pair,'eos_rate':pair,'length_mean':pair,
            'repetition_4gram':pair,'per_prompt':{},'rm_student_minus_rho':{'mean':.5,'bootstrap_95':[.1,.9],
            'wins':40,'losses':20}}),encoding='utf-8')
    return run


class FakeHub:
    """HfApi stand-in: records the commit and serves the committed files' sha256 (or a corrupted one)."""
    def __init__(self, corrupt: bool = False) -> None:
        self.corrupt,self.repos,self.commits = corrupt,[],[]
    def create_repo(self, repo, repo_type, private, exist_ok):
        self.repos.append((repo,private))
    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append(operations)
        from types import SimpleNamespace
        return SimpleNamespace(oid=f'c{len(self.commits)}')
    def model_info(self, repo, revision, files_metadata):
        import hashlib
        from types import SimpleNamespace
        siblings = []
        for op in self.commits[-1]:
            lfs = None
            if op.path_in_repo.endswith('.safetensors'):
                digest = hashlib.sha256(Path(op.path_or_fileobj).read_bytes()).hexdigest()
                lfs = SimpleNamespace(sha256='0'*64 if self.corrupt else digest)
            siblings.append(SimpleNamespace(rfilename=op.path_in_repo,lfs=lfs))
        return SimpleNamespace(siblings=siblings)


def test_push_hub_private_verified_and_idempotent(tmp_path: Path) -> None:
    """One private commit with weights, card and provenance; the Hub sha256 is checked and recorded; a second push
    of unchanged weights commits nothing; --update-card commits again."""
    sys.path.insert(0,'script/server')
    from push_hub import push
    run,hub = fake_run(tmp_path),FakeHub()
    record = push(run,'me/crpd-x',api=hub)
    assert hub.repos == [('me/crpd-x',True)] and record['verified'] and record['revision'] == 'c1'
    names = {op.path_in_repo for op in hub.commits[0]}
    assert names == {'config.json','model.safetensors','tokenizer.json','README.md','crpd_provenance.json'}
    card = next(op for op in hub.commits[0] if op.path_in_repo == 'README.md').path_or_fileobj.decode()
    assert 'base_model: u/rho' in card and 'No test or benchmark result was used' in card and '+0.500' in card
    assert json.loads((run/'hub.json').read_text())['revision'] == 'c1'
    push(run,'me/crpd-x',api=hub)
    assert len(hub.commits) == 1
    push(run,'me/crpd-x',api=hub,update_card=True)
    assert len(hub.commits) == 2


def test_push_hub_refusals(tmp_path: Path) -> None:
    """No push when selection kept rho, without held-out results, or when the Hub weights differ."""
    sys.path.insert(0,'script/server')
    from push_hub import push
    with pytest.raises(SystemExit,match='student is rho'):
        push(fake_run(tmp_path/'a',selected='reference'),'me/x',api=FakeHub())
    with pytest.raises(SystemExit,match='heldout'):
        push(fake_run(tmp_path/'b',heldout=False),'me/x',api=FakeHub())
    run = fake_run(tmp_path/'c')
    with pytest.raises(SystemExit,match='differs'):
        push(run,'me/x',api=FakeHub(corrupt=True))
    assert json.loads((run/'hub.json').read_text())['verified'] is False


def test_run_pair_push_guards_and_plan() -> None:
    """push needs a namespace and full scale (or --hub-allow-pilot); the dry run lists one private push per run."""
    def plan(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable,'script/server/run_pair.py','--pair','qwen3_1p7b_to_0p6b','--gpus','0,1',
                               '--gammas','0.1,0.05','--dry-run',*extra],capture_output=True,text=True)
    assert 'hub-namespace' in plan('--scale','full','--stages','push').stderr
    assert 'hub-allow-pilot' in plan('--scale','x10','--stages','push','--hub-namespace','me').stderr
    out = plan('--scale','full','--stages','push','--hub-namespace','me')
    assert out.returncode == 0, out.stderr
    pushes = [l for l in out.stdout.splitlines() if 'push_hub.py' in l]
    assert len(pushes) == 2 and all('--public' not in l for l in pushes)
    assert 'me/crpd-qwen3_1p7b_to_0p6b-full-crpd_raw-alpha0p03-gamma0p1-all' in pushes[0]


def scheduler_jobs(tmp_path: Path, durations: dict[str,float]):
    """Two pairs of fake jobs with the run_all.py shape: prep -> 3 collect shards -> train -> heldout."""
    sys.path.insert(0,'script/server')
    from run_all import Job, UNITS
    jobs = []
    for rank,(pair,train_units) in enumerate((('big',UNITS),('small',1))):
        def add(kind,detail,units,deps,order=0):
            name = f'{pair}/{kind}'+(f'/{detail}' if detail else '')
            jobs.append(Job(name,kind,rank,order,lambda gpu,n=name: [n,str(gpu)],units,deps,
                            tmp_path/f'{name.replace("/","_")}.log',tmp_path/'marks'/f'{name.replace("/","_")}.json'))
            return name
        prep = add('prep','',0,[])
        shards = [add('collect',str(i),UNITS,[prep],i) for i in range(3)]
        add('heldout','g',1,[add('train','g',train_units,shards)])
    return jobs


def fake_execute(durations: dict[str,float], record: list, fail: set = frozenset(), oom_once: set = frozenset()):
    import threading
    import time
    lock,seen = threading.Lock(),set()
    def execute(command,gpu,logfile):
        name = command[0]
        with lock:
            record.append(('start',name,gpu,time.time()))
        time.sleep(durations.get(name.split('/')[1],.01))
        with lock:
            record.append(('end',name,gpu,time.time()))
        if name in oom_once and name not in seen:
            seen.add(name)
            logfile.write_text('torch.OutOfMemoryError: CUDA out of memory',encoding='utf-8')
            return 1
        return 1 if name in fail else 0
    return execute


def test_scheduler_fills_both_gpus_and_respects_dependencies(tmp_path: Path) -> None:
    """Collection shards of one pair run on both GPUs at once; while the big pair trains on one GPU the other GPU
    collects the small pair; no GPU ever holds more than two units; every job starts after its dependencies;
    a rerun skips finished jobs."""
    sys.path.insert(0,'script/server')
    from run_all import Scheduler
    durations = {'prep':.02,'collect':.05,'train':.3,'heldout':.02}
    record = []
    jobs = scheduler_jobs(tmp_path,durations)
    assert Scheduler(jobs,['0','1'],fake_execute(durations,record),poll=.002).run()
    starts = {n:(g,t) for e,n,g,t in record if e == 'start'}
    ends = {n:t for e,n,g,t in record if e == 'end'}
    for job in jobs:
        assert all(ends[d] <= starts[job.name][1] for d in job.deps), job.name
    assert {starts[f'big/collect/{i}'][0] for i in range(3)} == {'0','1'}
    t0,t1 = starts['big/train/g'][1],ends['big/train/g']
    small = [n for n in starts if n.startswith('small/collect') and t0 <= starts[n][1] < t1]
    assert small and all(starts[n][0] != starts['big/train/g'][0] for n in small)
    units = {j.name:j.units for j in jobs}
    for gpu in ('0','1'):
        for _,n,g,t in record:
            if g == gpu:
                load = sum(units[m] for m,(h,s) in starts.items() if h == gpu and s <= t < ends[m])
                assert load <= 2
    record2 = []
    again = scheduler_jobs(tmp_path,durations)
    assert Scheduler(again,['0','1'],fake_execute(durations,record2),poll=.002).run()
    assert record2 == [] and all(j.state == 'skipped' for j in again)


def test_scheduler_oom_retry_and_failure_isolation(tmp_path: Path) -> None:
    """A half-GPU job that runs out of memory reruns on a whole GPU; a failed job cancels only its own pair."""
    sys.path.insert(0,'script/server')
    from run_all import Scheduler
    durations = {'prep':.01,'collect':.01,'train':.02,'heldout':.01}
    record = []
    jobs = scheduler_jobs(tmp_path,durations)
    ok = Scheduler(jobs,['0','1'],fake_execute(durations,record,fail={'big/collect/1'},oom_once={'small/train/g'}),
                   poll=.002).run()
    states = {j.name:j.state for j in jobs}
    assert not ok and states['big/collect/1'] == 'failed' and states['big/train/g'] == 'cancelled'
    assert states['small/heldout/g'] == 'done'
    assert next(j for j in jobs if j.name == 'small/train/g').units == 2
    assert sum(1 for e,n,g,t in record if e == 'start' and n == 'small/train/g') == 2


def test_run_all_dry_run_plans_pairs() -> None:
    """The dry run lists every job of every pair with the run_pair.py command it will run."""
    out = subprocess.run([sys.executable,'script/server/run_all.py','--gpus','0,1','--pair','qwen3_8b_to_1p7b:0.1',
                          '--pair','qwen3_1p7b_to_0p6b:0.1,0.05:all','--collect-shards','4','--eval-python','EVAL',
                          '--hub-namespace','me','--dry-run'],capture_output=True,text=True)
    assert out.returncode == 0, out.stderr
    import re
    assert len(re.findall(r'/collect/\d+ +GPU 2/2',out.stdout)) == 8 and 'qwen3_1p7b_to_0p6b/train/0.05' in out.stdout
    assert '--collect-shards 4 --shard-indices 3' in out.stdout and '--pair-sources all' in out.stdout
    assert 'qwen3_8b_to_1p7b/push' in out.stdout and 'benchmarks.py --pair qwen3_8b_to_1p7b --gpus <gpu>' in out.stdout
