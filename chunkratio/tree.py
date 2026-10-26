"""Independent FP64 finite-tree algebra, method equations (2)–(15), spec §12."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache

import torch
from torch import Tensor

State = tuple[int, ...]
Model = Mapping[State, list[float]]


class TensorWorld:
    """Enumerable variable-EOS world; all expectations use torch.float64 (§12)."""

    def __init__(self, T0: Model, Tp: Model, rho: Model, horizon: int = 5, eos: int = 2) -> None:
        """Accept the exact reference distributions with explicit stopping (§5)."""
        self.T0, self.Tp, self.rho = T0, Tp, rho
        self.horizon, self.eos = horizon, eos
        self.vocab = tuple(range(len(T0[()])))

    def terminal(self, s: State) -> bool:
        """EOS or global horizon is terminal (§5.2)."""
        return len(s) >= self.horizon or bool(s and s[-1] == self.eos)

    @lru_cache(maxsize=None)
    def completions(self, s: State) -> tuple[State, ...]:
        """Enumerate prefix-free suffix support including early EOS (§12)."""
        if self.terminal(s):
            return ((),)
        return tuple((a,)+z for a in self.vocab for z in self.completions(s+(a,)))

    def logp(self, model: Model, s: State, z: State) -> Tensor:
        """Token-sum log probability without length averaging (§4)."""
        values = [model[s+z[:i]][a] for i,a in enumerate(z)]
        return torch.tensor(values,dtype=torch.float64).log().sum()

    def d(self, s: State, z: State) -> float:
        """Teacher displacement delta_T (§4)."""
        return float(self.logp(self.Tp,s,z)-self.logp(self.T0,s,z))

    def ell(self, s: State, z: State) -> float:
        """Reference mismatch for the v1.0 diagnostic (method equation 4)."""
        return float(self.logp(self.rho,s,z)-self.logp(self.T0,s,z))

    @lru_cache(maxsize=None)
    def stats(self, s: State) -> dict[str,float]:
        """Both residuals, means, variance and range (method equation 4)."""
        if self.terminal(s):
            return dict.fromkeys(('phi','phi_alt','mu_rho','var_rho','mu_tp','d_range'),0.)
        suffixes = self.completions(s)
        lr, lp, lb = [torch.stack([self.logp(m,s,z) for z in suffixes]) for m in (self.rho,self.Tp,self.T0)]
        d, ell = lp-lb, lr-lb
        mu = (lr.exp()*d).sum()
        return dict(phi=float(torch.logsumexp(lr+d,0)),
                    phi_alt=float(torch.logsumexp(lp+ell,0)),mu_rho=float(mu),
                    var_rho=float((lr.exp()*(d-mu).square()).sum()),
                    mu_tp=float((lp.exp()*ell).sum()),d_range=float(d.max()-d.min()))

    def phi(self, s: State) -> float:
        """Exact log-exponential continuation (method equation 4)."""
        return self.stats(s)['phi']

    def target(self) -> dict[State,float]:
        """Normalized full sequence target (method equation 2)."""
        suffixes = self.completions(())
        scores = torch.stack([self.logp(self.rho,(),z)+self.logp(self.Tp,(),z)-self.logp(self.T0,(),z) for z in suffixes])
        return dict(zip(suffixes,scores.softmax(0).tolist()))

    def chunks(self, s: State, rule: Callable[[State],int]) -> list[State]:
        """Enumerate complete causal chunk actions (§5)."""
        out = []
        def grow(c: State) -> None:
            """Expand until causal cutoff (§5.2)."""
            if c and (self.terminal(s+c) or len(c) >= rule(s)):
                out.append(c)
                return
            for a in self.vocab:
                grow(c+(a,))
        grow(())
        return out

    def chunk_policy(self, s: State, rule: Callable[[State],int],
                     head: Callable[[State],float]) -> dict[State,float]:
        """Full-support normalization for reference tests ONLY (method equation 9)."""
        chunks = self.chunks(s,rule)
        values = torch.stack([self.logp(self.rho,s,c)+self.d(s,c)
                              +(0 if self.terminal(s+c) else head(s+c)) for c in chunks])
        return dict(zip(chunks,values.softmax(0).tolist()))
