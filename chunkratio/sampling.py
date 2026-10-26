"""Raw collection sampling and global response stopping (§§5,7)."""
from __future__ import annotations

import torch
from torch import nn

from .artifacts import digest


def stream(seed: int, name: str, device: str = 'cpu') -> torch.Generator:
    """Independent deterministic RNG by semantic source/job (§10)."""
    number = int(digest([seed,name])[:15],16)
    return torch.Generator(device=device).manual_seed(number)


def terminal(response: list[int], horizon: int, eos: int) -> bool:
    """Only sampled EOS/global horizon, not a chunk boundary, terminates (§5.2)."""
    return len(response) >= horizon or eos in response


@torch.no_grad()
def generate(model: nn.Module, prefix: list[int], remaining: int, eos: int,
             rng: torch.Generator, chunk_length: int | None = None) -> list[int]:
    """Temperature-1 untruncated categorical from raw logits; no artificial EOS (§5.3)."""
    assert prefix and remaining >= 0 and (chunk_length is None or chunk_length > 0)
    limit = min(remaining,chunk_length) if chunk_length else remaining
    context = list(prefix)
    result = []
    model.eval()
    device = next(model.parameters()).device
    for _ in range(limit):
        ids = torch.tensor([context],device=device)
        logits = model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float()
        assert torch.isfinite(logits).all()
        # CPU stream state is portable between machines; model forwarding may use GPU.
        token = int(torch.multinomial(logits.softmax(-1).cpu(),1,generator=rng))
        result.append(token)
        context.append(token)
        if token == eos:
            break
    return result
