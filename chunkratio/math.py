"""Pure tensor contracts: coding spec §§4, 7–8; method equations (1), (7)–(10)."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

Route = Literal["no_alignment_update", "exact_zero_residual", "reference_mean"]


def stable(x: Tensor) -> Tensor:
    """Accumulate at least FP32, preserving FP64 reference precision (§6.1)."""
    assert x.is_floating_point() and torch.isfinite(x).all(), "nonfinite scores"
    return x if x.dtype == torch.float64 else x.float()


def log_ratio(logp: Tensor, logq: Tensor, mask: Tensor | None = None) -> Tensor:
    """Summed d_T/delta_T/a_theta over the last (token) axis (§4)."""
    assert logp.shape == logq.shape and logp.ndim >= 1
    values = stable(logp) - stable(logq)
    if mask is not None:
        assert mask.shape == values.shape and mask.dtype == torch.bool
        values = values.masked_fill(~mask, 0)
    return values.sum(-1)


def suffix_returns(increments: Tensor) -> Tensor:
    """Return Y_t before each emitted token, without terminal row (§7.3)."""
    assert increments.ndim >= 1
    return stable(increments).flip((-1,)).cumsum(-1).flip((-1,))


def mu_targets(increments: Tensor) -> Tensor:
    """Suffix targets followed by exact terminal zero (§7.3, equation 10)."""
    y = suffix_returns(increments)
    return torch.cat((y, y.new_zeros((*y.shape[:-1], 1))), dim=-1)


def routing(teacher_plus_id: str, teacher_base_id: str, student_ref_id: str) -> Route:
    """Route verified effective fingerprints, never approximate scores (§4.1)."""
    assert teacher_plus_id and teacher_base_id and student_ref_id
    if teacher_plus_id == teacher_base_id:
        return "no_alignment_update"
    if student_ref_id == teacher_base_id:
        return "exact_zero_residual"
    return "reference_mean"


def target_score(logp_plus: Tensor, logp_base: Tensor, mu_hat: Tensor,
                 terminal: Tensor) -> Tensor:
    """Frozen t_s(c)=delta_T(c|s)+mu_hat(sc), terminal mu=0 (§8)."""
    assert logp_plus.shape == logp_base.shape == mu_hat.shape == terminal.shape
    assert terminal.dtype == torch.bool
    return (stable(logp_plus) - stable(logp_base)
            + stable(mu_hat).masked_fill(terminal, 0)).detach()


def pair_target_logits(t_c: Tensor, t_d: Tensor, alpha: float) -> Tensor:
    """Frozen alpha*(t_c-t_d), with a single shared positive alpha (§8)."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    assert t_c.shape == t_d.shape
    return (alpha * (stable(t_c) - stable(t_d))).detach()


def pair_loss(logp_student_c: Tensor, logp_student_d: Tensor,
              logp_ref_c: Tensor, logp_ref_d: Tensor,
              t_c: Tensor, t_d: Tensor, alpha: float) -> tuple[Tensor, Tensor, Tensor]:
    """Unreduced soft sibling BCE; never detach student gradients (§8)."""
    shape = logp_student_c.shape
    assert all(t.shape == shape for t in (logp_student_d, logp_ref_c, logp_ref_d, t_c, t_d))
    u = pair_target_logits(t_c, t_d, alpha)
    margin = alpha * ((stable(logp_student_c) - stable(logp_ref_c.detach()))
                      - (stable(logp_student_d) - stable(logp_ref_d.detach())))
    return F.binary_cross_entropy_with_logits(margin, u.sigmoid(), reduction="none"), margin, u


def hierarchical_weights(candidate_counts: Sequence[Sequence[int]]) -> Tensor:
    """Flatten all pairs with prompt/state/pair probability, excluding empty prompts (§7.2)."""
    assert candidate_counts and all(states for states in candidate_counts)
    values = []
    for states in candidate_counts:
        for k in states:
            assert k >= 2
            n = math.comb(k, 2)
            values.extend([1 / len(candidate_counts) / len(states) / n] * n)
    result = torch.tensor(values, dtype=torch.float64)
    assert abs(result.sum().item() - 1) < 1e-12
    return result


def head_objective(predictions: Sequence[Tensor], labels: Sequence[Tensor], horizon: int) -> Tensor:
    """Mean over trajectories of SSE/fixed H, never realized length (§7.4)."""
    assert horizon > 0 and len(predictions) == len(labels) > 0
    terms = []
    for pred, label in zip(predictions, labels):
        assert pred.shape == label.shape and pred.ndim == 1
        terms.append((stable(pred) - stable(label.detach())).square().sum() / horizon)
    return torch.stack(terms).mean()


def diagnostics(loss: Tensor, margin: Tensor, target_logit: Tensor) -> dict[str, Tensor]:
    """Excess BCE, logit error and saturation without zero-loss expectation (§8)."""
    u = target_logit.detach()
    p = u.sigmoid()
    entropy = F.binary_cross_entropy_with_logits(u, p, reduction="none")
    return {"excess_bce": (loss.detach() - entropy),
            "logit_error": (margin.detach() - u).abs(),
            "saturation": ((p < .01) | (p > .99)).float()}
