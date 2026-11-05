"""Shared causal teacher-forced scoring, masking and detached features (§6)."""
from __future__ import annotations

import contextlib
import hashlib
from typing import Any

import torch
from torch import Tensor, nn

from .artifacts import digest
from .sampling import logits_tail


def freeze(model: nn.Module) -> nn.Module:
    """Frozen models are eval/no parameter gradients (§§2,6)."""
    for parameter in model.parameters():
        parameter.grad = None
    return model.eval().requires_grad_(False)


def fingerprint(model: nn.Module) -> str:
    """Exact architecture, parameters, buffers and precision identity (§4.1)."""
    h = hashlib.sha256(type(model).__qualname__.encode())
    # module structure includes architecture hyperparameters (e.g. attention heads).
    h.update(repr(model).encode())
    if hasattr(model,'config'):
        config = model.config.to_dict() if hasattr(model.config,'to_dict') else vars(model.config)
        config = {k:v for k,v in config.items() if k not in ('_name_or_path','architectures','transformers_version')}
        h.update(digest(config).encode())
    for name,tensor in sorted(model.state_dict().items()):
        h.update(str((name,tuple(tensor.shape),str(tensor.dtype))).encode())
        # Same bytes as hashing the tensor's storage, without a per-byte Python loop
        # (bytes(UntypedStorage) took over an hour for a 1.7B FP32 model).
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()


def preflight(tokenizers: list[Any], models: list[nn.Module], eos_token: str | None = None) -> dict[str,Any]:
    """Require full token mapping, added/special tokens, EOS/template and output dims (§5.1).

    Output dimensions may be padded beyond the tokenizer (Qwen3), but must be identical for all
    four models. With a declared eos_token (for example Qwen3 <|im_end|>), every tokenizer's EOS
    id must be that token.
    """
    assert len(tokenizers) == len(models) == 4
    signatures, dims = [],[]
    for tok,model in zip(tokenizers,models):
        signatures.append({'vocab':tok.get_vocab(),'added':tok.get_added_vocab(),
                           'special':tok.special_tokens_map,'eos':tok.eos_token_id,
                           'pad':tok.pad_token_id,'bos':tok.bos_token_id,'template':tok.chat_template})
        dims.append(model.get_output_embeddings().weight.shape[0])
        if dims[-1] < len(tok):
            raise ValueError('output vocabulary smaller than tokenizer')
        if eos_token is not None and tok.get_vocab().get(eos_token) != tok.eos_token_id:
            raise ValueError(f'tokenizer EOS must be the declared end-of-turn token {eos_token}')
    if len(set(dims)) != 1:
        raise ValueError('padded output vocabulary differs between models')
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError('unsupported tokenizer or template mismatch')
    if signatures[0]['eos'] is None or not signatures[0]['template']:
        raise ValueError('explicit EOS and canonical chat template required')
    return {'tokenizer_hash':digest(signatures[0]),'eos':signatures[0]['eos'],'output_dim':dims[0]}


def score_tokens(logits: Tensor, ids: Tensor, attention: Tensor, prompt_lengths: Tensor,
                 sequence_lengths: Tensor, eos: int) -> tuple[Tensor,Tensor]:
    """logits[k] predicts ids[k+1]; positional masks include first EOS (§6.1)."""
    assert logits.ndim == 3 and ids.shape == logits.shape[:2] == attention.shape
    assert ids.dtype == torch.long and prompt_lengths.shape == sequence_lengths.shape == (ids.shape[0],)
    assert (prompt_lengths >= 1).all() and (sequence_lengths >= prompt_lengths).all()
    assert (sequence_lengths <= ids.shape[1]).all()
    positions = torch.arange(1,ids.shape[1],device=ids.device)[None,:]
    response = (positions >= prompt_lengths[:,None]) & (positions < sequence_lengths[:,None]) & attention[:,1:].bool()
    eos_hits = (ids[:,1:] == eos) & response
    seen_before = eos_hits.cumsum(-1)-eos_hits.long()
    mask = response & (seen_before == 0)
    # Normalise only the scored positions: gathering them first gives the same log-probs without
    # a [rows, width, vocab] FP32 log-softmax over prompt and prefix positions (and the copy
    # autograd keeps of it), which ran an A100-80GB out of memory at H=512.
    counts = mask.sum(-1)
    span = max(int(counts.max()),1)
    offsets = torch.arange(span,device=ids.device)[None,:]
    index = (prompt_lengths[:,None]-1+offsets).clamp(max=ids.shape[1]-2)
    picked = (offsets < counts[:,None]) & mask.gather(1,index)
    raw = logits[:,:-1,:].gather(1,index[:,:,None].expand(-1,-1,logits.shape[-1]))
    raw = raw if raw.dtype == torch.float64 else raw.float()
    assert torch.isfinite(raw).all(), 'nonfinite prediction logits'
    logps = raw.log_softmax(-1).gather(-1,ids[:,1:].gather(1,index)[:,:,None]).squeeze(-1)
    # scatter_add: clamped duplicate indices only carry zeros, so they cannot overwrite a score.
    values = torch.zeros(mask.shape,dtype=logps.dtype,device=logps.device).scatter_add(1,index,logps.masked_fill(~picked,0))
    assert torch.isfinite(values).all()
    return values,mask


def padded(sequences: list[list[int]], device: torch.device, pad: int) -> tuple[Tensor,Tensor]:
    """Right-pad by position; EOS may equal PAD (§6.1)."""
    assert sequences and all(sequences)
    width = max(map(len,sequences))
    ids = torch.full((len(sequences),width),pad,dtype=torch.long,device=device)
    attention = torch.zeros_like(ids)
    for i,seq in enumerate(sequences):
        ids[i,:len(seq)] = torch.tensor(seq,device=device)
        attention[i,:len(seq)] = 1
    return ids,attention


def score_batch(model: nn.Module, prefixes: list[list[int]], chunks: list[list[int]],
                eos: int, frozen: bool = True, score_eos: bool = True) -> Tensor:
    """One authoritative chunk scorer for frozen roles and gradient-carrying student (§6.1).

    The LM head runs only on the positions that predict chunk tokens (tail_logprobs); the student's
    graph therefore holds [rows, longest chunk + 1, vocab] logits instead of the full sequence width.
    """
    assert len(prefixes) == len(chunks) > 0 and all(prefixes)
    model.eval()
    with torch.no_grad() if frozen else contextlib.nullcontext():
        values,_ = tail_logprobs(model,prefixes,chunks,eos,score_eos)
    return values.sum(-1)


@torch.no_grad()
def state_features(model: nn.Module, states: list[list[int]]) -> Tensor:
    """Last valid token's final hidden state from frozen rho, ordinary detached tensor (§6.2)."""
    model.eval()
    device = next(model.parameters()).device
    ids,attention = padded(states,device,0)
    hidden = model(ids,attention_mask=attention,output_hidden_states=True,use_cache=False,
                   **logits_tail(model,1)).hidden_states[-1]
    idx = torch.tensor([len(s)-1 for s in states],device=device)
    return hidden[torch.arange(len(states),device=device),idx].detach().float()


@torch.no_grad()
def trajectory_features(model: nn.Module, prefix: list[int], tokens: list[int]) -> Tensor:
    """Features of every state before tokens[t] from one causal pass, position P+t-1 (§6.2)."""
    assert prefix and tokens
    model.eval()
    ids = torch.tensor([prefix+tokens[:-1]],device=next(model.parameters()).device)
    hidden = model(ids,attention_mask=torch.ones_like(ids),output_hidden_states=True,use_cache=False,
                   **logits_tail(model,1)).hidden_states[-1]
    return hidden[0,len(prefix)-1:].detach().float()


def left_padded(sequences: list[list[int]], device: torch.device, pad: int) -> tuple[Tensor,Tensor,Tensor]:
    """Left-pad so every sequence ends at the last column; positions count only valid tokens."""
    assert sequences and all(sequences)
    width = max(map(len,sequences))
    ids = torch.full((len(sequences),width),pad,dtype=torch.long,device=device)
    attention = torch.zeros_like(ids)
    for i,seq in enumerate(sequences):
        ids[i,width-len(seq):] = torch.tensor(seq,device=device)
        attention[i,width-len(seq):] = 1
    return ids,attention,(attention.cumsum(-1)-1).clamp(min=0)


def length_groups(lengths: list[int], tokens_per_batch: int) -> list[list[int]]:
    """Row indices sorted by length and cut so rows x longest row stays within the budget."""
    assert tokens_per_batch > 0
    groups: list[list[int]] = []
    for i in sorted(range(len(lengths)),key=lambda i:lengths[i]):
        if groups and (len(groups[-1])+1)*lengths[i] <= tokens_per_batch:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def tail_logprobs(model: nn.Module, prefixes: list[list[int]], suffixes: list[list[int]],
                  eos: int, score_eos: bool = True) -> tuple[Tensor,Tensor]:
    """FP32 log-probs of each suffix token after its prefix, [rows, longest suffix], and the scored mask (§6.1).

    Rows are left-padded with position ids, so every suffix ends at the last column and the LM head
    runs only on the last longest-suffix + 1 positions; the backbone still reads each full context.
    As in score_tokens, prompt and padding are never scored, the first EOS is scored once and later
    tokens are not. With score_eos=False (opt-in `score_end_token: false`, DECISIONS) the end-of-turn
    token is not scored either, so ending a response is neutral for every role. Gradients flow unless the caller disables them.
    """
    device = next(model.parameters()).device
    ids,attention,positions = left_padded([p+s for p,s in zip(prefixes,suffixes)],device,eos)
    lengths = torch.tensor([len(s) for s in suffixes],device=device)
    span = max(int(lengths.max()),1)
    logits = model(ids,attention_mask=attention,position_ids=positions,use_cache=False,
                   **logits_tail(model,span+1)).logits[:,-(span+1):]
    # Column j of the tail predicts the token after it; suffix token t of a row with T tokens is
    # predicted at column span-T+t.
    offsets = torch.arange(span,device=device)[None,:]
    valid = offsets < lengths[:,None]
    index = (span-lengths[:,None]+offsets).clamp(max=span-1)
    targets = torch.full((len(suffixes),span),eos,dtype=torch.long,device=device)
    for r,suffix in enumerate(suffixes):
        targets[r,:len(suffix)] = torch.tensor(suffix,device=device,dtype=torch.long)
    eos_hits = (targets == eos) & valid
    mask = valid & (eos_hits.cumsum(-1)-(eos_hits.long() if score_eos else 0) == 0)
    raw = logits.gather(1,index[:,:,None].expand(-1,-1,logits.shape[-1]))
    raw = raw if raw.dtype == torch.float64 else raw.float()
    assert torch.isfinite(raw).all(), 'nonfinite prediction logits'
    logps = raw.log_softmax(-1).gather(-1,targets[:,:,None]).squeeze(-1).masked_fill(~mask,0)
    assert torch.isfinite(logps).all()
    return logps,mask


def sibling_logprobs(model: nn.Module, prefixes: list[list[int]], chunk_lists: list[list[list[int]]],
                     eos: int, score_eos: bool = True) -> list[Tensor]:
    """Summed FP32 log-probs of several chunks after one shared prefix each (§6.1), prefix computed once.

    The prefixes run once (left-padded, positions from the mask) with their cache; the cache rows are
    repeated for each prefix's chunks, which then run on it right-padded, as generation continues a
    prefix. Scores equal score_batch(prefix, chunk) up to floating-point rounding, with the same masks:
    prefix and padding never scored, the first EOS scored once, later tokens not. Gradients flow unless
    the caller disables them. Returns, per prefix, a tensor of its chunks' sums in order.
    """
    from .sampling import select_rows
    assert len(prefixes) == len(chunk_lists) > 0 and all(prefixes)
    chunks = [chunk for group in chunk_lists for chunk in group]
    assert chunks and all(chunks)
    model.eval()
    device = next(model.parameters()).device
    ids,attention,positions = left_padded(prefixes,device,eos)
    out = model(ids,attention_mask=attention,position_ids=positions,use_cache=True,**logits_tail(model,1))
    owner = torch.tensor([i for i,group in enumerate(chunk_lists) for _ in group],device=device)
    past = select_rows(out.past_key_values,owner)
    span = max(map(len,chunks))
    targets = torch.full((len(chunks),span),eos,dtype=torch.long,device=device)
    chunk_mask = torch.zeros_like(targets)
    for r,chunk in enumerate(chunks):
        targets[r,:len(chunk)] = torch.tensor(chunk,device=device,dtype=torch.long)
        chunk_mask[r,:len(chunk)] = 1
    steps = torch.arange(span,device=device)[None,:]
    cont = model(targets,attention_mask=torch.cat([attention[owner],chunk_mask],dim=1),
                 position_ids=positions[owner][:,-1:]+1+steps,past_key_values=past,use_cache=True)
    # The prefix's last position predicts chunk token 0; chunk position t-1 predicts token t.
    logits = torch.cat([out.logits[:,-1:][owner],cont.logits[:,:span-1]],dim=1)
    raw = logits if logits.dtype == torch.float64 else logits.float()
    assert torch.isfinite(raw).all(), 'nonfinite prediction logits'
    lengths = chunk_mask.sum(-1)
    valid = steps < lengths[:,None]
    eos_hits = (targets == eos) & valid
    mask = valid & (eos_hits.cumsum(-1)-(eos_hits.long() if score_eos else 0) == 0)
    sums = raw.log_softmax(-1).gather(-1,targets[:,:,None]).squeeze(-1).masked_fill(~mask,0).sum(-1)
    assert torch.isfinite(sums).all()
    return list(sums.split([len(group) for group in chunk_lists]))


@torch.no_grad()
def continuation_increments_batch(model: nn.Module, prefixes: list[list[int]], suffixes: list[list[int]],
                                  eos: int, tokens_per_batch: int = 8192) -> list[Tensor]:
    """continuation_increments for many trajectories at once (§7.3).

    Rows of similar length share a forward of tail_logprobs (LM head on the suffix positions only).
    """
    assert len(prefixes) == len(suffixes) and all(prefixes)
    assert all(eos not in s[:-1] for s in suffixes), 'suffix continues after its first EOS'
    model.eval()
    device = next(model.parameters()).device
    results = [torch.empty(0,device=device) for _ in suffixes]
    scored = [i for i,s in enumerate(suffixes) if s]
    for group in length_groups([len(prefixes[i])+len(suffixes[i]) for i in scored],tokens_per_batch):
        rows = [scored[g] for g in group]
        logps,_ = tail_logprobs(model,[prefixes[i] for i in rows],[suffixes[i] for i in rows],eos)
        for r,i in enumerate(rows):
            results[i] = logps[r,:len(suffixes[i])]
    return results


@torch.no_grad()
def trajectory_features_batch(model: nn.Module, prefixes: list[list[int]], suffixes: list[list[int]],
                              tokens_per_batch: int = 8192) -> list[Tensor]:
    """trajectory_features for many trajectories at once (§6.2).

    Each row is prefix + suffix[:-1], left-padded, so a row's T states are its last T positions.
    Only one position's logits are requested; the features are the final hidden states.
    """
    assert len(prefixes) == len(suffixes) and all(prefixes) and all(suffixes)
    model.eval()
    device = next(model.parameters()).device
    results: list[Tensor] = [torch.empty(0)]*len(suffixes)
    sequences = [p+s[:-1] for p,s in zip(prefixes,suffixes)]
    for rows in length_groups(list(map(len,sequences)),tokens_per_batch):
        ids,attention,positions = left_padded([sequences[i] for i in rows],device,0)
        hidden = model(ids,attention_mask=attention,position_ids=positions,output_hidden_states=True,
                       use_cache=False,**logits_tail(model,1)).hidden_states[-1]
        for r,i in enumerate(rows):
            results[i] = hidden[r,ids.shape[1]-len(suffixes[i]):].detach().float()
    return results


@torch.no_grad()
def continuation_increments(model: nn.Module, prefix: list[int], tokens: list[int], eos: int) -> Tensor:
    """Per-token frozen likelihoods for reverse suffix labels (§7.3)."""
    if not tokens:
        return torch.empty(0,device=next(model.parameters()).device)
    ids,attention = padded([prefix+tokens],next(model.parameters()).device,eos)
    values,_ = score_tokens(model.eval()(ids,attention_mask=attention,use_cache=False).logits,ids,attention,
                          torch.tensor([len(prefix)],device=ids.device),torch.tensor([ids.shape[1]],device=ids.device),eos)
    return values[0,len(prefix)-1:]
