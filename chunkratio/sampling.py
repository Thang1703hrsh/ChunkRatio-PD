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
def generate_batch(model: nn.Module, prefixes: list[list[int]], remaining: list[int], eos: int,
                   rngs: list[torch.Generator], chunk_length: int | None = None) -> list[list[int]]:
    """Batched KV-cached temperature-1 categorical sampling; one generator per row (§5.3).

    Rows are left-padded; each row draws only from its own generator over softmax(raw logits),
    so its tokens do not depend on batch composition. No top-k/top-p, repetition penalty,
    forced EOS or stop strings; a row stops at its first EOS or after min(remaining, L) tokens.
    """
    assert prefixes and len(prefixes) == len(remaining) == len(rngs) and all(prefixes)
    assert all(r >= 0 for r in remaining) and (chunk_length is None or chunk_length > 0)
    limits = [min(r,chunk_length) if chunk_length else r for r in remaining]
    results: list[list[int]] = [[] for _ in prefixes]
    active = [limit > 0 for limit in limits]
    if not any(active):
        return results
    model.eval()
    device = next(model.parameters()).device
    width = max(map(len,prefixes))
    ids = torch.full((len(prefixes),width),eos,dtype=torch.long,device=device)
    mask = torch.zeros_like(ids)
    for i,prefix in enumerate(prefixes):
        ids[i,width-len(prefix):] = torch.tensor(prefix,device=device)
        mask[i,width-len(prefix):] = 1
    positions = (mask.cumsum(-1)-1).clamp(min=0)
    step_ids, step_positions, past = ids, positions, None
    while True:
        out = model(step_ids,attention_mask=mask,position_ids=step_positions,past_key_values=past,use_cache=True)
        logits = out.logits[:,-1].float()
        past = getattr(out,'past_key_values',None)
        tokens = []
        for i in range(len(prefixes)):
            token = eos
            if active[i]:
                assert torch.isfinite(logits[i]).all()
                # CPU stream state is portable between machines; model forwarding may use GPU.
                token = int(torch.multinomial(logits[i].softmax(-1).cpu(),1,generator=rngs[i]))
                results[i].append(token)
                active[i] = token != eos and len(results[i]) < limits[i]
            tokens.append(token)
        if not any(active):
            return results
        new = torch.tensor(tokens,device=device)[:,None]
        mask = torch.cat([mask,torch.ones_like(new)],dim=1)
        if past is None:
            # Models without a cache recompute the full context (correct, slower).
            ids = torch.cat([ids,new],dim=1)
            step_ids, step_positions = ids, (mask.cumsum(-1)-1).clamp(min=0)
        else:
            step_ids, step_positions = new, step_positions[:,-1:]+1


def generate(model: nn.Module, prefix: list[int], remaining: int, eos: int,
             rng: torch.Generator, chunk_length: int | None = None) -> list[int]:
    """Single-row form of generate_batch (§5.3)."""
    return generate_batch(model,[prefix],[remaining],eos,[rng],chunk_length)[0]
