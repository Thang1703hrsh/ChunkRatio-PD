"""ChunkRPO loss: chunk-level, one-sided SBA Bregman ratio matching.

Implements Sections 3.2--3.4 of the ChunkRPO paper on top of per-token
policy/reference log-ratios and the deterministic chunk-id partition produced in
``preference_datasets.py``.

Pipeline (all vectorized over a padded ``(B, max_chunks)`` chunk layout):

  1. ``aggregate_chunk_logratios`` -- scatter per-token log-ratios into per-chunk
     sums + token counts, giving length-normalized chunk log-ratios

         r_i = (beta / |c_i|^eta) * sum_{t in c_i} log( pi_theta / pi_ref ).

  2. ``trimmed_mean`` -- the alpha-trimmed mean of a response's chunk values,
     used as the *cross-response* baseline (Eq. 5). For very short responses
     (fewer than ``n_min`` chunks) we fall back to the median (the alpha->1/2
     limit), which degrades gracefully.

  3. ``group_relative_advantages`` -- z^w_i = r^w_i - sg(mu^l), z^l_j = sg(mu^w)
     - r^l_j (Eq. 6), with optional robust (MAD) standardization (Eq. 8). The
     baseline enters under stop-gradient, so the advantage gradient is a positive
     multiple of the chunk's own log-ratio gradient (Lemma 1, leakage-free).

  4. ``chunkrpo_loss`` -- map to ratio space R_hat = exp(z), apply the one-sided
     SBA Bregman divergence D^+_{h_lambda}(R_C, R_hat) (Eqs. 9--11), average over
     valid chunks per side and sum the two sides.

The SBA generator and its divergence reuse the same family as ``h_function.py``;
we implement the one-sided divergence directly here in log-ratio space because
the loss is gated on ``R_hat < R_C`` and is most stable when written in terms of
``z`` and ``tau_C``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Chunk aggregation
# ----------------------------------------------------------------------------- #
def aggregate_chunk_logratios(
    per_token_logratio: torch.Tensor,
    chunk_ids: torch.Tensor,
    beta: float,
    eta: float,
    max_chunks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-token log-ratios into length-normalized per-chunk values.

    Args:
        per_token_logratio: ``(B, L)`` log( pi_theta / pi_ref ) per response token
            (already restricted to label positions; value is arbitrary where
            ``chunk_ids < 0`` and is masked out here).
        chunk_ids: ``(B, L)`` integer chunk index per token; ``< 0`` (e.g. -100)
            marks prompt/padding positions to ignore.
        beta: preference scale.
        eta: length-normalization exponent in [0.5, 1].
        max_chunks: width of the output chunk axis. If ``None``, inferred from the
            data as ``chunk_ids.max() + 1``.

    Returns:
        ``(r, chunk_mask)`` where
          - ``r``: ``(B, C)`` length-normalized chunk log-ratios (0 where invalid),
            differentiable w.r.t. the policy through ``per_token_logratio``;
          - ``chunk_mask``: ``(B, C)`` bool, True for chunks that contain >=1 token.
    """
    B, L = chunk_ids.shape
    valid = chunk_ids >= 0
    safe_ids = torch.where(valid, chunk_ids, torch.zeros_like(chunk_ids))

    if max_chunks is None:
        max_chunks = int(chunk_ids.max().item()) + 1 if valid.any() else 1
    C = max_chunks

    masked_logratio = per_token_logratio * valid

    # per-chunk sum of log-ratios
    chunk_sum = per_token_logratio.new_zeros(B, C)
    chunk_sum.scatter_add_(1, safe_ids, masked_logratio)

    # per-chunk token counts
    ones = valid.to(per_token_logratio.dtype)
    chunk_count = per_token_logratio.new_zeros(B, C)
    chunk_count.scatter_add_(1, safe_ids, ones)

    chunk_mask = chunk_count > 0
    denom = chunk_count.clamp_min(1.0).pow(eta)
    r = beta * chunk_sum / denom
    r = r * chunk_mask  # zero out empty chunk slots
    return r, chunk_mask


# ----------------------------------------------------------------------------- #
# Robust cross-response baseline
# ----------------------------------------------------------------------------- #
def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-wise median over valid entries, fully vectorized.

    Sorts each row ascending with invalid entries pushed to +inf, so the valid
    values occupy columns ``[0, n)``. The median is then the middle order
    statistic (average of the two middle ones for even ``n``). Rows with no valid
    entries return 0.
    """
    B, C = values.shape
    counts = mask.sum(dim=1)  # (B,)
    pos_inf = torch.finfo(values.dtype).max
    filled = torch.where(mask, values, values.new_full((), pos_inf))
    sorted_asc, _ = torch.sort(filled, dim=1, descending=False)  # (B, C)

    n = counts.clamp_min(1)
    lo = (n - 1) // 2  # lower-middle index
    hi = n // 2  # upper-middle index (== lo when n is odd)
    g_lo = sorted_asc.gather(1, lo.unsqueeze(1)).squeeze(1)
    g_hi = sorted_asc.gather(1, hi.unsqueeze(1)).squeeze(1)
    med = 0.5 * (g_lo + g_hi)
    return torch.where(counts > 0, med, values.new_zeros(B))


def trimmed_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    n_min: int = 4,
) -> torch.Tensor:
    """Row-wise alpha-trimmed mean over valid entries (Eq. 5), fully vectorized.

    Drops the ``floor(alpha * n)`` smallest and largest valid entries, then
    averages the rest. ``alpha == 0`` recovers the raw mean. For rows with fewer
    than ``n_min`` valid entries (or where trimming would empty the row) the
    trimmed mean is ill-defined, so we fall back to the median (the alpha -> 1/2
    limit). Returns a ``(B,)`` tensor; rows with no valid entries return 0.

    The result is used only as a stop-gradient baseline (the caller detaches it),
    so a purely functional, branch-free formulation is fine.
    """
    B, C = values.shape
    counts = mask.sum(dim=1)  # (B,)

    if alpha <= 0.0:
        # raw masked mean
        s = (values * mask).sum(dim=1)
        m = s / counts.clamp_min(1).to(values.dtype)
        return torch.where(counts > 0, m, values.new_zeros(B))

    # Sort ascending with invalid entries pushed to +inf -> valid values fill
    # columns [0, n) in ascending order per row.
    pos_inf = torch.finfo(values.dtype).max
    filled = torch.where(mask, values, values.new_full((), pos_inf))
    sorted_asc, _ = torch.sort(filled, dim=1, descending=False)  # (B, C)

    n = counts  # (B,)
    g = (alpha * n.to(values.dtype)).floor().to(n.dtype)  # (B,) trim count per end
    # rows where trimming is valid: enough chunks and a non-empty kept window
    use_trim = (n >= n_min) & (n - 2 * g > 0)

    # build a [g, n-g) keep-window mask over the ascending sort
    col = torch.arange(C, device=values.device).unsqueeze(0)  # (1, C)
    keep = (col >= g.unsqueeze(1)) & (col < (n - g).unsqueeze(1))
    # only meaningful at finite (valid) sorted positions
    keep = keep & (col < n.unsqueeze(1))

    keep_f = keep.to(values.dtype)
    kept_sum = (sorted_asc * keep_f).sum(dim=1)
    kept_cnt = keep_f.sum(dim=1).clamp_min(1.0)
    trimmed = kept_sum / kept_cnt

    median = _masked_median(values, mask)
    out = torch.where(use_trim, trimmed, median)
    return torch.where(counts > 0, out, values.new_zeros(B))


def _masked_mad(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise median absolute deviation over valid entries (for Eq. 8)."""
    med = _masked_median(values, mask)  # (B,)
    abs_dev = (values - med.unsqueeze(1)).abs()
    mad = _masked_median(abs_dev, mask)  # (B,)
    return mad + eps


def group_relative_advantages(
    r_w: torch.Tensor,
    mask_w: torch.Tensor,
    r_l: torch.Tensor,
    mask_l: torch.Tensor,
    alpha: float,
    n_min: int = 4,
    standardize: bool = False,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute group-relative chunk advantages z^w, z^l (Eqs. 6, 8).

    The cross-response baseline of the chosen side is the trimmed mean of the
    *rejected* chunk values (and vice-versa), entered under stop-gradient so no
    gradient leaks through the comparison set (Lemma 1).

    Returns ``(z_w, z_l)`` each ``(B, C)``; values at invalid chunk slots are 0.
    """
    mu_l = trimmed_mean(r_l, mask_l, alpha, n_min=n_min).detach()  # baseline for chosen
    mu_w = trimmed_mean(r_w, mask_w, alpha, n_min=n_min).detach()  # baseline for rejected

    z_w = r_w - mu_l.unsqueeze(1)
    z_l = mu_w.unsqueeze(1) - r_l

    if standardize:
        mad_l = _masked_mad(r_l, mask_l, eps=eps).detach()
        mad_w = _masked_mad(r_w, mask_w, eps=eps).detach()
        z_w = z_w / mad_l.unsqueeze(1)
        z_l = z_l / mad_w.unsqueeze(1)

    z_w = z_w * mask_w
    z_l = z_l * mask_l
    return z_w, z_l


# ----------------------------------------------------------------------------- #
# One-sided SBA Bregman divergence in log-ratio space
# ----------------------------------------------------------------------------- #
def one_sided_sba_divergence(
    z: torch.Tensor,
    tau_c: float,
    lam: float,
    s: float,
) -> torch.Tensor:
    """One-sided SBA Bregman divergence D^+_{h_lambda}(R_C, R_hat) (Eqs. 9--11).

    With R_hat = exp(z) and R_C = exp(tau_c), and the SBA generator

        h_lambda(R) = (R^{1+lambda} - R) / (s * lambda * (lambda+1)),
        h'_lambda(R) = (R^lambda - 1/(1+lambda)) / (s * lambda),

    the Bregman divergence is

        D(R_C, R_hat) = h(R_C) - h(R_hat) - h'(R_hat) (R_C - R_hat),

    gated by the indicator 1[R_hat < R_C] so already-separated chunks (z >= tau_c)
    contribute exactly 0 (Proposition 1). The limit lambda -> 0 uses the
    KL/logistic generator h(R) = (R log R - R) / s (a numerically-safe special
    case). Computation is clamped so R_hat does not overflow.
    """
    active = z < tau_c  # 1[R_hat < R_C]

    # Clamp z to the active region for the divergence value (no effect on the gate
    # since inactive entries are zeroed anyway) to keep exp() bounded by R_C.
    z_clamped = torch.clamp(z, max=tau_c)

    R_hat = torch.exp(z_clamped)
    R_c = float(torch.exp(torch.tensor(tau_c)).item())

    if abs(lam) < 1e-6:
        # KL / logistic generator: h(R) = (R*logR - R)/s, h'(R) = logR / s.
        # D = [h(R_c) - h(R_hat) - h'(R_hat)(R_c - R_hat)]
        #   = (1/s)[ R_c*tau_c - R_c - (R_hat*z - R_hat) - z*(R_c - R_hat) ]
        #   = (1/s)[ R_c*(tau_c - z) - R_c + R_hat ]
        div = (R_c * (tau_c - z_clamped) - R_c + R_hat) / s
    else:
        c0 = 1.0 / (s * lam * (lam + 1.0))
        h_Rc = (R_c ** (1.0 + lam) - R_c) * c0
        h_Rhat = (R_hat.pow(1.0 + lam) - R_hat) * c0
        hp_Rhat = (R_hat.pow(lam) - 1.0 / (1.0 + lam)) / (s * lam)
        div = h_Rc - h_Rhat - hp_Rhat * (R_c - R_hat)

    # numerical floor: Bregman divergence is >= 0 by convexity
    div = div.clamp_min(0.0)
    return div * active


# ----------------------------------------------------------------------------- #
# Saliency weights (optional ablation, Eq. 12)
# ----------------------------------------------------------------------------- #
def _entmax_uniform(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked softmax over valid chunks (stand-in for Entmax in Eq. 12).

    A plain softmax is the alpha=1 member of the entmax family; it keeps full
    coverage and is dependency-free. Returns a per-row distribution over valid
    chunks (0 on invalid slots).
    """
    neg_inf = torch.finfo(scores.dtype).min
    masked = torch.where(mask, scores, scores.new_full((), neg_inf))
    w = torch.softmax(masked, dim=1)
    w = w * mask
    return w


def saliency_weights(
    z: torch.Tensor,
    mask: torch.Tensor,
    tau_c: float,
    rho: float,
    temp: float,
) -> torch.Tensor:
    """Coverage-preserving saliency weights (Eq. 12).

    omega = (1 - rho)/n + rho * Entmax(e / T_s), with deficit e = [tau_c - sg(z)]_+
    and a uniform floor (1-rho)/n that prevents gradient starvation. Returns
    ``(B, C)`` weights that sum to ~1 over valid chunks per row (uniform when
    rho == 0).
    """
    n = mask.sum(dim=1, keepdim=True).clamp_min(1).to(z.dtype)  # (B,1)
    uniform = mask.to(z.dtype) / n
    if rho <= 0.0:
        return uniform
    deficit = (tau_c - z.detach()).clamp_min(0.0)
    sal = _entmax_uniform(deficit / temp, mask)
    return (1.0 - rho) * uniform + rho * sal


# ----------------------------------------------------------------------------- #
# Top-level loss
# ----------------------------------------------------------------------------- #
def chunkrpo_loss(
    per_token_logratio: torch.Tensor,
    chunk_ids: torch.Tensor,
    batch_size: int,
    beta: float,
    eta: float,
    alpha: float,
    tau_c: float,
    lam: float,
    s: float,
    n_min: int = 4,
    standardize: bool = False,
    rho: float = 0.0,
    saliency_temp: float = 1.0,
):
    """Compute the ChunkRPO objective (Eq. 7) for a concatenated chosen/rejected batch.

    Args:
        per_token_logratio: ``(2B, L)`` per-token log( pi_theta / pi_ref ), with the
            first ``B`` rows = chosen, last ``B`` rows = rejected.
        chunk_ids: ``(2B, L)`` chunk index per token (``< 0`` = ignore), same split.
        batch_size: ``B``.
        beta, eta: chunk log-ratio scale / length exponent (Eq. 4).
        alpha: trimming fraction for the cross-response baseline (Eq. 5).
        tau_c: target log-margin; ``R_C = exp(tau_c)`` (>= 0).
        lam, s: SBA generator parameters (Eq. 9).
        n_min: chunk-count threshold below which the baseline falls back to median.
        standardize: if True, robust-standardize advantages by MAD (Eq. 8).
        rho, saliency_temp: saliency-weighting strength / temperature (Eq. 12);
            ``rho == 0`` gives uniform weights (the default objective).

    Returns:
        ``(losses, info)`` where ``losses`` is a ``(B,)`` per-pair loss (mean of the
        chosen-side and rejected-side averages) and ``info`` is a dict of detached
        diagnostics.
    """
    # shared chunk axis width across chosen+rejected so trimmed means line up
    valid_all = chunk_ids >= 0
    max_chunks = int(chunk_ids.max().item()) + 1 if valid_all.any() else 1

    r_all, mask_all = aggregate_chunk_logratios(
        per_token_logratio, chunk_ids, beta=beta, eta=eta, max_chunks=max_chunks
    )

    r_w, r_l = r_all[:batch_size], r_all[batch_size:]
    mask_w, mask_l = mask_all[:batch_size], mask_all[batch_size:]

    z_w, z_l = group_relative_advantages(
        r_w, mask_w, r_l, mask_l, alpha=alpha, n_min=n_min, standardize=standardize
    )

    d_w = one_sided_sba_divergence(z_w, tau_c=tau_c, lam=lam, s=s)
    d_l = one_sided_sba_divergence(z_l, tau_c=tau_c, lam=lam, s=s)

    omega_w = saliency_weights(z_w, mask_w, tau_c, rho, saliency_temp)
    omega_l = saliency_weights(z_l, mask_l, tau_c, rho, saliency_temp)

    # per-side averages over chunks (omega sums to ~1 over valid chunks, so this is
    # a weighted mean; with rho=0 it reduces to (1/m) sum as in Eq. 7).
    loss_w = (omega_w * d_w).sum(dim=1)
    loss_l = (omega_l * d_l).sum(dim=1)
    losses = loss_w + loss_l  # (B,)

    with torch.no_grad():
        active_w = ((z_w < tau_c) & mask_w).float()
        active_l = ((z_l < tau_c) & mask_l).float()
        denom_w = mask_w.sum(dim=1).clamp_min(1)
        denom_l = mask_l.sum(dim=1).clamp_min(1)
        info = {
            "z_w_mean": (z_w * mask_w).sum() / mask_w.sum().clamp_min(1),
            "z_l_mean": (z_l * mask_l).sum() / mask_l.sum().clamp_min(1),
            "frac_active_w": (active_w.sum() / mask_w.sum().clamp_min(1)),
            "frac_active_l": (active_l.sum() / mask_l.sum().clamp_min(1)),
            "n_chunks_w": mask_w.sum(dim=1).float(),
            "n_chunks_l": mask_l.sum(dim=1).float(),
            "loss_w": loss_w.detach(),
            "loss_l": loss_l.detach(),
            "r_w_mean_per_pair": (r_w * mask_w).sum(dim=1) / denom_w,
            "r_l_mean_per_pair": (r_l * mask_l).sum(dim=1) / denom_l,
        }

    return losses, info
