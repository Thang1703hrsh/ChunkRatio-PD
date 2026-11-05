"""Displacement diagnostic converges to exact tree quantities (method eqs. 4, 15a)."""
import importlib.util
import math
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch',reason='diagnostic requires PyTorch')
from chunkratio.diagnostics import gap_statistics, state_report
from test_reference import ROOT

PROMPT = 3


def reference() -> object:
    """Load the supplied finite-tree reference implementation (§12)."""
    spec = importlib.util.spec_from_file_location('tree_reference_diag',ROOT/'reference/ChunkRatio_Math_Checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TreeLM(torch.nn.Module):
    """Causal LM whose next-token law is an exact tree table; token 3 is the prompt (§12)."""

    def __init__(self, table: dict) -> None:
        """Wrap per-prefix probabilities over tokens 0, 1 and EOS=2."""
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.table = table
        self.config = SimpleNamespace(max_position_embeddings=64,hidden_size=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object) -> SimpleNamespace:
        """Logits from the valid (unmasked) history of each row; no cache."""
        rows,steps = input_ids.shape
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask[:,-steps:]
        logits = torch.zeros(rows,steps,4,dtype=torch.float64)
        for b in range(rows):
            history = []
            for t in range(steps):
                if mask[b,t]:
                    history.append(int(input_ids[b,t]))
                probs = self.table.get(tuple(history[1:]))
                if probs is not None:
                    logits[b,t,:3] = torch.tensor(probs,dtype=torch.float64).log()
                    logits[b,t,3] = -40.
        return SimpleNamespace(logits=logits)


def exact_gap(world: object, ref: object, u: tuple) -> float:
    """phi - mu_rho, exactly zero at terminal endpoints."""
    if ref.terminal(u):
        return 0.
    stats = world.stats(u)
    return stats['phi']-stats['mu_rho']


def test_estimates_converge_to_exact_tree_values() -> None:
    """Mean d, Jensen gap and sibling omega approach exact phi/mu_rho as R grows (§12)."""
    ref = reference()
    T0,Tp,rho = ref.make_world(random.Random(31),.5,.5)
    world = ref.World(T0,Tp,rho)
    models = {'teacher_plus':TreeLM(Tp),'teacher_base':TreeLM(T0),'student_ref':TreeLM(rho)}
    state = {'id':'root','prompt_id':'p','prefix_ids':[PROMPT],'response_ids':[],'n':0}
    cfg = {'horizon':ref.H,'chunk_length':2,'collection_seed':0}
    mu, gap = world.stats(())['mu_rho'], exact_gap(world,ref,())
    errors = {}
    for replicates in (20,1000):
        report = state_report(models,state,cfg,ref.EOS,replicates)
        sibling_errors = [abs(e['jensen_gap']-exact_gap(world,ref,tuple(e['tokens']))) for e in report['sibling_endpoints']]
        exact_gaps = [exact_gap(world,ref,tuple(e['tokens'])) for e in report['sibling_endpoints']]
        low,high = report['jensen_gap_bootstrap_95']
        errors[replicates] = {'mean':abs(report['mean']-mu),'gap':abs(report['jensen_gap']-gap),
                              'gap_ci_width':high-low,'sibling':max(sibling_errors),
                              'omega':abs(report['omega_s']-(max(exact_gaps)-min(exact_gaps))),
                              'se':math.sqrt(report['variance']/replicates)}
    small, large = errors[20], errors[1000]
    # The plug-in gap log mean exp - mean is biased down by O(var/R): grade it by its own bootstrap width.
    assert large['mean'] < 4*large['se']
    assert large['gap'] < 1.5*large['gap_ci_width'] and large['gap_ci_width'] < small['gap_ci_width']
    assert large['gap'] < small['gap'] and large['sibling'] < small['sibling']
    assert large['sibling'] < .02 and large['omega'] < .03
    print('diagnostic convergence',errors)


def test_batched_state_reports_equal_per_state_reports() -> None:
    """Batching rollouts across states and endpoints changes no token and no statistic."""
    from chunkratio.diagnostics import state_reports
    ref = reference()
    T0,Tp,rho = ref.make_world(random.Random(7),.5,.5)
    models = {'teacher_plus':TreeLM(Tp),'teacher_base':TreeLM(T0),'student_ref':TreeLM(rho)}
    states = [{'id':'root','prompt_id':'p','prefix_ids':[PROMPT],'response_ids':[],'n':0},
              {'id':'one','prompt_id':'p','prefix_ids':[PROMPT,1],'response_ids':[1],'n':1},
              {'id':'zero','prompt_id':'q','prefix_ids':[PROMPT,0],'response_ids':[0],'n':1}]
    cfg = {'horizon':ref.H,'chunk_length':2,'collection_seed':0,'generation_batch_size':5}
    assert state_reports(models,states,cfg,ref.EOS,6) == [state_report(models,s,cfg,ref.EOS,6) for s in states]


def run_diagnose(monkeypatch: pytest.MonkeyPatch, output: object, rho_seed: int, prompt: list[int] | None = None,
                 tokenizer_hash: str = 'tok', revision: str = 'rev') -> dict:
    """diagnose() on tiny models with a chosen rho; returns its displacement.json."""
    import chunkratio.models
    import chunkratio.pipeline
    from chunkratio.artifacts import read_json
    from chunkratio.diagnostics import diagnose
    from chunkratio.tiny import TinyLM
    def build(seed: int) -> TinyLM:
        """Tiny model with weights fixed by the seed."""
        torch.manual_seed(seed)
        return TinyLM()
    models = {'teacher_plus':build(1),'teacher_base':build(2),'student_ref':build(rho_seed)}
    info = {'tokenizer_hash':tokenizer_hash,'eos':0,'output_dim':11}
    rows = [{'id':'p','split':'train','prompt_ids':prompt or [1,2,3]}]
    monkeypatch.setattr(chunkratio.models,'load_models',lambda cfg:(models,None,[None]*4,info))
    monkeypatch.setattr(chunkratio.pipeline,'eligible_prompts',lambda cfg,m,t:(rows,[]))
    monkeypatch.setenv('CRPD_CODE_REVISION',revision)
    cfg = {'horizon':12,'chunk_length':3,'collection_seed':0,'generation_batch_size':4,'variant':'crpd_main'}
    diagnose(cfg,1,2,output)
    return read_json(output/'displacement.json')


def test_diagnostic_cache_follows_model_tokenizer_prompts_and_code(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Reusing an output folder after changing rho gives what a fresh run gives, never the old prefixes."""
    shared = tmp_path/'shared'
    first = run_diagnose(monkeypatch,shared,3)
    second = run_diagnose(monkeypatch,shared,4)
    fresh = run_diagnose(monkeypatch,tmp_path/'fresh',4)
    assert second == fresh and second != first
    assert second['models']['student_ref'] != first['models']['student_ref']
    assert second['code_revision'].startswith('rev+src.') and second['tokenizer']['tokenizer_hash'] == 'tok'
    def caches() -> int:
        """Prefix namespaces in the shared folder."""
        return len(list(shared.glob('prefixes-*')))
    assert caches() == 2
    run_diagnose(monkeypatch,shared,4)
    assert caches() == 2  # same inputs resume the same shards
    run_diagnose(monkeypatch,shared,4,tokenizer_hash='other')
    assert caches() == 3
    run_diagnose(monkeypatch,shared,4,revision='other')
    assert caches() == 4
    run_diagnose(monkeypatch,shared,4,prompt=[1,2,4])
    assert caches() == 5


def test_rare_tail_flag_and_interval() -> None:
    """One dominant exponential weight triggers the warning; the bootstrap interval covers the gap."""
    flagged = gap_statistics([0.,0.,0.,6.],0,'tail')
    assert flagged['rare_tail_warning'] and flagged['max_exponential_weight_share'] > .5
    calm = gap_statistics([.1,-.1,.05,-.05,0.],0,'calm')
    assert not calm['rare_tail_warning']
    low,high = calm['jensen_gap_bootstrap_95']
    assert low <= calm['jensen_gap'] + 1e-12 and high >= 0
    assert calm['range'] == pytest.approx(.2) and calm['mean'] == pytest.approx(0)


def test_correction_audit_reads_sibling_differences() -> None:
    """Flip share counts pairs whose preferred side mu_hat reverses; endpoint stats compare centred siblings."""
    from chunkratio.diagnostics import correction_audit
    def side(plus: float, mu: float, source: str) -> dict:
        """One scored candidate."""
        return {'logp_teacher_plus':plus,'logp_teacher_base':0.,'endpoint_mu':mu,'sources':[source]}
    pairs = [{'c':side(1.,0.,'student_ref'),'d':side(0.,2.,'teacher_plus'),'weight':.75},   # flipped by mu
             {'c':side(1.,1.,'student_ref'),'d':side(0.,0.,'student_ref'),'weight':.25}]   # kept
    report = correction_audit(pairs,[],None,None,0)
    assert report['pairs']['all']['flip_share'] == .75 and report['pairs']['rho+teacher']['flip_share'] == 1.
    assert report['pairs']['rho+rho']['flip_share'] == 0. and report['endpoints'] is None

    class FirstFeature(torch.nn.Module):
        """mu_hat = the first feature, so predictions can be set exactly."""
        def __init__(self) -> None:
            """One unused parameter gives the head a device."""
            super().__init__()
            self.unused = torch.nn.Parameter(torch.zeros(1))
        def forward(self, features: torch.Tensor, budgets: torch.Tensor) -> torch.Tensor:
            """First feature of every row."""
            return features[:,0]
    state = {'id':'s','prompt_id':'p','prefix_ids':[1,2],'seed_source':'prefix','split':'head_val'}
    rows = [{'seed':state,'labels':[0.],'budgets':[1.],'value':0.}]
    for i,(label,source) in enumerate(((1.,'student_ref'),(3.,'teacher_plus'),(5.,'student_ref'))):
        rows.append({'seed':{**state,'id':f'e{i}','prefix_ids':[1,2,3+i],'seed_source':[source]},
                     'labels':[label,0.],'budgets':[.5,.4],'value':label})
    features = lambda row: torch.tensor([[row['value'],0.],[0.,0.]])
    endpoints = correction_audit(pairs,rows,FirstFeature(),features,0)['endpoints']
    assert endpoints['states'] == 1 and endpoints['endpoints'] == 3
    assert abs(endpoints['all']['corr']-1) < 1e-12 and abs(endpoints['all']['r2']-1) < 1e-12
    assert endpoints['rho']['n'] == 2 and endpoints['teacher']['n'] == 1
    # Centred over the state, each source's subset is not centred: here both sources are perfectly
    # anti-correlated (Pearson -1) while the pooled sample is +0.8; an uncentred product reported +0.8 for both.
    rows = [{'seed':state,'labels':[0.],'budgets':[1.],'value':0.}]
    for i,(mu,label,source) in enumerate(((1.,2.,'student_ref'),(2.,1.,'student_ref'),
                                          (-1.,-2.,'teacher_plus'),(-2.,-1.,'teacher_plus'))):
        rows.append({'seed':{**state,'id':f'f{i}','prefix_ids':[1,2,7+i],'seed_source':[source]},
                     'labels':[label],'budgets':[.5],'value':mu})
    split = correction_audit(pairs,rows,FirstFeature(),features,0)['endpoints']
    assert abs(split['all']['corr']-.8) < 1e-12
    assert abs(split['rho']['corr']+1) < 1e-12 and abs(split['teacher']['corr']+1) < 1e-12
    # Endpoint-only correction (compact) has no prefix seeds: inferring parents from them finds no state,
    # while parents taken from the candidate records audit the same endpoints as before.
    endpoint_only = rows[1:]
    assert correction_audit(pairs,endpoint_only,FirstFeature(),features,0)['endpoints']['states'] == 0
    parents = {row['seed']['id']:state['id'] for row in endpoint_only}
    fixed = correction_audit(pairs,endpoint_only,FirstFeature(),features,0,parents=parents)['endpoints']
    assert fixed['states'] == 1 and fixed['endpoints'] == 4
    assert fixed['all'] == split['all'] and fixed['rho'] == split['rho'] and fixed['teacher'] == split['teacher']
