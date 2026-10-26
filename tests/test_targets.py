"""Frozen target table, endpoint correction and hierarchical measure (§§4,7.2,8)."""
import collections
import copy

import pytest

torch = pytest.importorskip('torch',reason='target construction requires PyTorch')
from chunkratio.config import load_config
from chunkratio.math import hierarchical_weights, target_score
from chunkratio.models import load_models
from chunkratio.scoring import freeze
from chunkratio.targets import build_targets
from chunkratio.tiny import TinyLM
from chunkratio.training import ChunkRatioTrainer
from chunkratio.value_head import MeanHead


class ConstantHead(MeanHead):
    """Frozen head returning a fixed mu_hat and recording its budget inputs (§6.2)."""

    def __init__(self, value: float, hidden_size: int = 12) -> None:
        """Real head parameters so the frozen-head checks still apply (§6.2)."""
        super().__init__(hidden_size)
        self.value, self.budgets = value, []

    def forward(self, features: torch.Tensor, budgets: torch.Tensor) -> torch.Tensor:
        """Constant prediction independent of features (§6.2)."""
        self.budgets.extend(budgets.tolist())
        return torch.full((len(features),),self.value)


def candidate(name: str, tokens: list[int], terminal: bool, plus: float, base: float) -> dict:
    """Cached frozen scores for one sibling (§8)."""
    return {'id':name,'tokens':tokens,'terminal':terminal,'sources':['student_ref'],
            'logp_teacher_plus':plus,'logp_teacher_base':base,'logp_student_ref':-2.}


def test_target_score_function() -> None:
    """t_s(c)=delta_T+mu_hat, with mu_hat forced to zero at terminal endpoints (§4)."""
    values = [torch.tensor([x],dtype=torch.float64) for x in (-1.,-1.8,.25)]
    assert target_score(*values,torch.tensor([False])).item() == pytest.approx(1.05,abs=1e-12)
    assert target_score(*values,torch.tensor([True])).item() == pytest.approx(.8,abs=1e-12)


def test_build_targets_sign_correction_terminal_and_budget() -> None:
    """Endpoint mu_hat is added to log T+ - log T0, zero at EOS/H, budget (H-n-|c|)/H (§§6.2,8)."""
    cfg = load_config('config/crpd/smoke.json')
    models,_,_,info = load_models(cfg)
    state = {'id':'s','prefix_ids':[1,2],'response_ids':[5],'prompt_id':'p','split':'train','n':1}
    rows = [candidate('c',[3,4],False,-1.,-1.8),candidate('d',[5,6],False,-2.,-1.8),
            candidate('e',[7,0],True,-.5,-.7)]
    head = freeze(ConstantHead(.25))
    result = build_targets([{'state':state,'candidates':rows,'skipped':False}],head,models,cfg,info['eos'],'reference_mean')
    by_id = {c['id']:c for c in result['records'][0]['candidates']}
    assert by_id['c']['endpoint_mu'] == pytest.approx(.25)
    assert by_id['c']['target_score'] == pytest.approx(.8+.25,abs=1e-7)
    assert by_id['d']['target_score'] == pytest.approx(-.2+.25,abs=1e-7)
    assert by_id['e']['endpoint_mu'] == 0 and by_id['e']['target_score'] == pytest.approx(.2,abs=1e-12)
    assert head.budgets == pytest.approx([(6-1-2)/6]*2)
    pair = {(p['c']['id'],p['d']['id']):p for p in result['pairs']}[('c','d')]
    assert pair['c']['target_score'] > pair['d']['target_score']
    raw = build_targets([{'state':state,'candidates':rows,'skipped':False}],head,models,
                        {**cfg,'variant':'crpd_raw'},info['eos'],'reference_mean')
    assert [c['target_score'] for c in raw['records'][0]['candidates']] == pytest.approx([.8,-.2,.2])


def hierarchical_records() -> list[dict]:
    """Two train prompts (states with K=2,3 and K=4), one skip and one head_val state (§7.2)."""
    def record(prompt: str, state: str, k: int, split: str = 'train', skipped: bool = False) -> dict:
        return {'state':{'id':state,'prefix_ids':[1],'response_ids':[],'prompt_id':prompt,'split':split,'n':0},
                'candidates':[candidate(state+str(i),[i+2],False,-1.,-1.) for i in range(k)],'skipped':skipped}
    return [record('a','a1',2),record('a','a2',3),record('a','a3',1,skipped=True),
            record('b','b1',4),record('c','c1',4,split='head_val')]


def test_pair_weights_are_hierarchical() -> None:
    """1/N_prompts * 1/N_states * 1/C(K,2), skips and non-train states excluded (§7.2)."""
    cfg = load_config('config/crpd/smoke.json')
    models,_,_,info = load_models(cfg)
    result = build_targets(hierarchical_records(),None,models,cfg,info['eos'],'exact_zero_residual')
    weights = collections.defaultdict(list)
    for pair in result['pairs']:
        weights[pair['state']['id']].append(pair['weight'])
    assert {k:len(v) for k,v in weights.items()} == {'a1':1,'a2':3,'b1':6}
    assert weights['a1'] == pytest.approx([1/2*1/2*1])
    assert weights['a2'] == pytest.approx([1/2*1/2*1/3]*3)
    assert weights['b1'] == pytest.approx([1/2*1/6]*6)
    torch.testing.assert_close(torch.tensor([p['weight'] for p in result['pairs']],dtype=torch.float64),
                               hierarchical_weights([[2,3],[4]]))


def test_sampler_matches_hierarchical_weights() -> None:
    """Prompt, then state, then pair uniform sampling reproduces the declared weights (§7.2)."""
    cfg = load_config('config/crpd/smoke.json')
    models,_,_,info = load_models(cfg)
    targets = build_targets(hierarchical_records(),None,models,cfg,info['eos'],'exact_zero_residual')
    torch.manual_seed(0)
    rho = freeze(TinyLM())
    trainer = ChunkRatioTrainer(copy.deepcopy(rho).requires_grad_(True),rho,targets,[],
                                {**cfg,'batch_size':40000},info['eos'])
    counts = collections.Counter(p['id'] for p in trainer.sample())
    for pair in targets['pairs']:
        assert counts[pair['id']]/40000 == pytest.approx(pair['weight'],abs=.01)


def test_audit_endpoints_cover_both_sources() -> None:
    """Replicated audits include teacher-proposed endpoints, not only the first rho siblings (§7.4)."""
    cfg = {**load_config('config/crpd/smoke.json'),'variant':'crpd_audit','audit_endpoints':2,'audit_replicates':2}
    models,_,_,info = load_models(cfg)
    def state(name: str, source: str) -> dict:
        rows = [{**candidate(name+str(i),[i+2],False,-1.,-1.),'sources':[source]} for i in range(3)]
        return {'state':{'id':name,'prefix_ids':[1],'response_ids':[],'prompt_id':name,'split':'train','n':0},
                'candidates':rows,'skipped':False}
    result = build_targets([state('r','student_ref'),state('t','teacher_plus')],None,models,cfg,
                           info['eos'],'exact_zero_residual')
    assert [a['sources'] for a in result['audits']] == [['student_ref'],['teacher_plus']]
    assert all(a['replicates'] == 2 and a['mu_hat'] == 0 for a in result['audits'])
