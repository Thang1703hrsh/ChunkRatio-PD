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


def test_rare_tail_flag_and_interval() -> None:
    """One dominant exponential weight triggers the warning; the bootstrap interval covers the gap."""
    flagged = gap_statistics([0.,0.,0.,6.],0,'tail')
    assert flagged['rare_tail_warning'] and flagged['max_exponential_weight_share'] > .5
    calm = gap_statistics([.1,-.1,.05,-.05,0.],0,'calm')
    assert not calm['rare_tail_warning']
    low,high = calm['jensen_gap_bootstrap_95']
    assert low <= calm['jensen_gap'] + 1e-12 and high >= 0
    assert calm['range'] == pytest.approx(.2) and calm['mean'] == pytest.approx(0)
