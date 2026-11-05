"""Raw collection sampling and global response stopping (§§5,7)."""
from __future__ import annotations

import hashlib
import inspect

import torch
from torch import nn

from .artifacts import digest


def stream(seed: int, name: str, device: str = 'cpu') -> torch.Generator:
    """Independent deterministic RNG by semantic source/job (§10)."""
    number = int(digest([seed,name])[:15],16)
    return torch.Generator(device=device).manual_seed(number)


def rng_digest(generator: torch.Generator) -> str:
    """sha256 of a generator's state bytes, the shard record of a draw stream (DECISIONS 88).

    The state before a job is that of stream(seed, name) and the state after it follows by replaying the
    job, so a digest identifies both; storing the ~5 KB state as JSON integers made the RNG about 98% of
    every candidate shard (153 of 156 KB at x10)."""
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


def terminal(response: list[int], horizon: int, eos: int) -> bool:
    """Only sampled EOS/global horizon, not a chunk boundary, terminates (§5.2)."""
    return len(response) >= horizon or eos in response


def logits_tail(model: nn.Module, count: int) -> dict[str,int]:
    """Ask HF causal LMs for the last `count` positions' logits only; models without the argument
    (TinyLM) return every position and callers slice. Hidden states are unchanged either way."""
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError,ValueError):
        return {}
    for name in ('logits_to_keep','num_logits_to_keep'):
        if name in parameters:
            return {name:count}
    return {}


def last_logits_only(model: nn.Module) -> dict[str,int]:
    """Last-position logits only: a prefill otherwise builds [rows, width, vocab] FP32 logits
    (57 GiB for 64 Qwen3 states at H=512) of which sampling reads one position."""
    return logits_tail(model,1)


def select_rows(past: object, index: torch.Tensor) -> object:
    """Keep only the batch rows in `index` of a KV cache (HF Cache, legacy tuples) or TinyLM state."""
    if past is None:
        return None
    if hasattr(past,'batch_select_indices'):
        past.batch_select_indices(index)
        return past
    if isinstance(past,torch.Tensor):
        # TinyLM's recurrent state is [layers, batch, hidden].
        return past.index_select(1,index)
    if isinstance(past,tuple):
        return tuple(tuple(t.index_select(0,index) for t in layer) for layer in past)
    raise TypeError(f'cannot select batch rows of a {type(past).__name__} cache')


def sample_rows(logits: torch.Tensor, generators: list[torch.Generator], gpu: bool = False) -> list[int]:
    """One temperature-1 categorical draw per row from softmax(raw logits), each from its own generator.

    Default: the row's CPU generator drives torch.multinomial over CPU probabilities (the original rule).
    gpu=True (opt-in): the row's generator draws one FP64 uniform u on the CPU and the token is found on
    the device by inverse CDF over FP64 probabilities, so only `rows` uniforms and token ids cross the
    bus instead of [rows, vocab] probabilities. Both are exact categorical sampling from the same
    distribution and keep one independent stream per row; their draws differ, so a run must use one rule.
    """
    if not gpu:
        probs = logits.softmax(-1).cpu()
        return [int(torch.multinomial(probs[j],1,generator=g)) for j,g in enumerate(generators)]
    uniforms = torch.tensor([float(torch.rand((),generator=g,dtype=torch.float64)) for g in generators],
                            dtype=torch.float64,device=logits.device)
    cdf = logits.double().softmax(-1).cumsum(-1)
    tokens = torch.searchsorted(cdf,(uniforms*cdf[:,-1])[:,None],right=True).clamp(max=cdf.shape[-1]-1)
    return tokens[:,0].tolist()


@torch.no_grad()
def generate_batch(model: nn.Module, prefixes: list[list[int]], remaining: list[int], eos: int,
                   rngs: list[torch.Generator], chunk_length: int | None = None,
                   gpu_sampling: bool = False) -> list[list[int]]:
    """Batched KV-cached temperature-1 categorical sampling; one generator per row (§5.3).

    Rows are left-padded; each row draws only from its own generator over softmax(raw logits),
    so its tokens do not depend on batch composition. No top-k/top-p, repetition penalty,
    forced EOS or stop strings; a row stops at its first EOS or after min(remaining, L) tokens
    and then leaves the batch, so later steps no longer compute it.
    """
    assert prefixes and len(prefixes) == len(remaining) == len(rngs) and all(prefixes)
    assert all(r >= 0 for r in remaining) and (chunk_length is None or chunk_length > 0)
    limits = [min(r,chunk_length) if chunk_length else r for r in remaining]
    results: list[list[int]] = [[] for _ in prefixes]
    # Original index of each batch row; rows with nothing to generate never enter the batch.
    rows = [i for i,limit in enumerate(limits) if limit > 0]
    if not rows:
        return results
    model.eval()
    device = next(model.parameters()).device
    width = max(len(prefixes[i]) for i in rows)
    ids = torch.full((len(rows),width),eos,dtype=torch.long,device=device)
    mask = torch.zeros_like(ids)
    for j,i in enumerate(rows):
        ids[j,width-len(prefixes[i]):] = torch.tensor(prefixes[i],device=device)
        mask[j,width-len(prefixes[i]):] = 1
    positions = (mask.cumsum(-1)-1).clamp(min=0)
    step_ids, step_positions, past = ids, positions, None
    extra = last_logits_only(model)
    while True:
        out = model(step_ids,attention_mask=mask,position_ids=step_positions,past_key_values=past,use_cache=True,**extra)
        logits = out.logits[:,-1].float()
        past = getattr(out,'past_key_values',None)
        assert torch.isfinite(logits).all()
        # Row streams are CPU generators, so their state stays portable between machines.
        tokens, keep = sample_rows(logits,[rngs[i] for i in rows],gpu_sampling),[]
        for j,i in enumerate(rows):
            token = tokens[j]
            results[i].append(token)
            if token != eos and len(results[i]) < limits[i]:
                keep.append(j)
        if not keep:
            return results
        new = torch.tensor(tokens,device=device)[:,None]
        mask = torch.cat([mask,torch.ones_like(new)],dim=1)
        if len(keep) < len(rows):
            # Finished rows leave the batch together with their cache entries.
            index = torch.tensor(keep,device=device)
            rows = [rows[j] for j in keep]
            new, mask, ids, step_positions = new[index], mask[index], ids[index], step_positions[index]
            past = select_rows(past,index)
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
