"""Revision-2 §§4,7,8,12 numeric, gradient and reduction fixtures."""
import math

import pytest

torch = pytest.importorskip("torch", reason="active interpreter has no compatible PyTorch installation")
from chunkratio.math import (diagnostics, head_objective, hierarchical_weights,
                            mu_targets, pair_loss, pair_target_logits, routing)


def tensor(x: object, grad: bool = False) -> torch.Tensor:
    """Construct the FP64 fixture (§12)."""
    return torch.tensor(x, dtype=torch.float64, requires_grad=grad)


def test_numeric_fixture() -> None:
    """Match target .24, probability, log2 and derivative (§8)."""
    m = tensor(0., True)
    loss, margin, u = pair_loss(m, tensor(0.), tensor(0.), tensor(0.), tensor(.9), tensor(.1), .3)
    assert u.item() == pytest.approx(.24, abs=1e-12)
    assert u.sigmoid().item() == pytest.approx(.5597136492671929, abs=1e-12)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-12)
    derivative, = torch.autograd.grad(loss, margin, retain_graph=True)
    assert derivative.item() == pytest.approx(-.0597136492671929, abs=1e-12)
    loss.backward()
    assert m.grad.item() < 0


def test_suffix_endpoint_exclusion() -> None:
    """Y_t labels state before y_t and terminal is zero (§7.3)."""
    actual = mu_targets(tensor([.3, -.2, .4]))
    torch.testing.assert_close(actual, tensor([.5, .2, .4, 0.]), atol=1e-12, rtol=0)
    assert actual[2] == pytest.approx(.4)


def test_symmetry_offset_and_frozen_targets() -> None:
    """Swapping siblings/common offsets preserve loss, target/reference detached (§8)."""
    c, d, r, s, tc, td = [tensor(v, True) for v in [-2., -3., -2.2, -2.4, .9, .1]]
    first = pair_loss(c,d,r,s,tc,td,.3)[0]
    swapped = pair_loss(d,c,s,r,td,tc,.3)[0]
    shifted = pair_loss(c,d,r,s,tc+7,td+7,.3)[0]
    torch.testing.assert_close(first, swapped)
    torch.testing.assert_close(first, shifted)
    first.backward()
    assert c.grad is not None and d.grad is not None
    assert all(x.grad is None for x in (r,s,tc,td))


def test_weights_and_accumulated_reductions() -> None:
    """Hierarchical full/accumulated gradients and fixed-H regression (§§7.2,7.4)."""
    w = hierarchical_weights([[2,4],[3]])
    assert w.sum().item() == pytest.approx(1)
    assert w[0].item() == pytest.approx(.25)
    x = tensor(list(range(len(w))), True)
    full = (x.square()*w).sum()
    g1, = torch.autograd.grad(full, x)
    accumulated = sum((x[i:i+2].square()*w[i:i+2]).sum() for i in range(0,len(w),2))
    g2, = torch.autograd.grad(accumulated, x)
    torch.testing.assert_close(g1,g2)
    value = head_objective([tensor([2.]),tensor([2.,2.])],[tensor([0.]),tensor([0.,0.])],4)
    assert value.item() == pytest.approx(1.5)


def test_label_error_and_entropy_floor() -> None:
    """Lipschitz label bound and excess BCE diagnostic (§§8,12)."""
    gen = torch.Generator().manual_seed(7)
    c, d = torch.randn((2,1000), generator=gen, dtype=torch.float64)
    eps = .07
    ec, ed = (torch.rand((2,1000), generator=gen, dtype=torch.float64)*2-1)*eps
    for alpha in (.3,1.):
        p = pair_target_logits(c,d,alpha).sigmoid()
        q = pair_target_logits(c+ec,d+ed,alpha).sigmoid()
        assert (p-q).abs().max() <= alpha*eps/2+1e-12
    u = tensor([.24])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(u,u.sigmoid(),reduction='none')
    assert diagnostics(bce,u,u)['excess_bce'].abs().max() < 1e-12


def test_exact_routes() -> None:
    """Effective identity routing has required precedence (§4.1)."""
    assert routing('a','a','a') == 'no_alignment_update'
    assert routing('b','a','a') == 'exact_zero_residual'
    assert routing('b','a','c') == 'reference_mean'
    with pytest.raises(ValueError):
        pair_target_logits(tensor(0.),tensor(0.),0)
