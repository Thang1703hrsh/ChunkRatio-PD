"""Measured forward/prefill token accounting for all four CRPD roles (§11.4)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ForwardMeter:
    """Count actual nonpadding input tokens, including repeated prefix prefill (§11)."""

    def __init__(self, models: dict[str,nn.Module]) -> None:
        """Attach measurement-only pre-hooks; parameters and outputs remain unchanged (§11.4)."""
        self.counts = {role:{'forward_calls':0,'input_tokens':0,'with_grad_input_tokens':0} for role in models}
        self.handles = []
        for role,model in models.items():
            self.handles.append(model.register_forward_pre_hook(self._hook(role),with_kwargs=True))

    def _hook(self, role: str) -> Any:
        """Create a per-role counter that does not retain model references (§11.4)."""
        def count(module: nn.Module, args: tuple[Any,...], kwargs: dict[str,Any]) -> None:
            """Count positional attention validity, never EOS token identity (§11.4)."""
            ids = args[0] if args else kwargs.get('input_ids')
            if ids is None:
                return
            attention = kwargs.get('attention_mask')
            tokens = int(attention.sum()) if attention is not None else ids.numel()
            self.counts[role]['forward_calls'] += 1
            self.counts[role]['input_tokens'] += tokens
            if torch.is_grad_enabled():
                self.counts[role]['with_grad_input_tokens'] += tokens
        return count

    def close(self) -> None:
        """Detach measurement hooks after the invocation (§11.4)."""
        for handle in self.handles:
            handle.remove()
