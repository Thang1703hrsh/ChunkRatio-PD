"""Shared causal teacher-forced scoring, masking and detached features (§6)."""
from __future__ import annotations

import contextlib
import hashlib
from typing import Any

import torch
from torch import Tensor, nn

from .artifacts import digest


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
        data = tensor.detach().cpu().contiguous().clone().untyped_storage()
        for start in range(0,len(data),1024*1024):
            h.update(bytes(data[start:start+1024*1024]))
    return h.hexdigest()


def preflight(tokenizers: list[Any], models: list[nn.Module]) -> dict[str,Any]:
    """Require full token mapping, added/special tokens, EOS/template and output dims (§5.1)."""
    assert len(tokenizers) == len(models) == 4
    signatures = []
    for tok,model in zip(tokenizers,models):
        signatures.append({'vocab':tok.get_vocab(),'added':tok.get_added_vocab(),
                           'special':tok.special_tokens_map,'eos':tok.eos_token_id,
                           'pad':tok.pad_token_id,'bos':tok.bos_token_id,'template':tok.chat_template})
        if model.get_output_embeddings().weight.shape[0] != len(tok):
            raise ValueError('output vocabulary differs from tokenizer')
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError('unsupported tokenizer or template mismatch')
    if signatures[0]['eos'] is None or not signatures[0]['template']:
        raise ValueError('explicit EOS and canonical chat template required')
    return {'tokenizer_hash':digest(signatures[0]),'eos':signatures[0]['eos']}


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
    raw = logits[:,:-1,:]
    raw = raw if raw.dtype == torch.float64 else raw.float()
    assert torch.isfinite(raw).all(), 'nonfinite prediction logits'
    logps = raw.log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
    values = logps.masked_fill(~mask,0)
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
                eos: int, frozen: bool = True) -> Tensor:
    """One authoritative chunk scorer for frozen roles and gradient-carrying student (§6.1)."""
    assert len(prefixes) == len(chunks) > 0 and all(prefixes)
    device = next(model.parameters()).device
    seqs = [p+c for p,c in zip(prefixes,chunks)]
    ids,attention = padded(seqs,device,eos)
    model.eval()
    with torch.no_grad() if frozen else contextlib.nullcontext():
        logits = model(ids,attention_mask=attention,use_cache=False).logits
        values,_ = score_tokens(logits,ids,attention,torch.tensor(list(map(len,prefixes)),device=device),
                              torch.tensor(list(map(len,seqs)),device=device),eos)
    return values.sum(-1)


@torch.no_grad()
def state_features(model: nn.Module, states: list[list[int]]) -> Tensor:
    """Last valid token's final hidden state from frozen rho, ordinary detached tensor (§6.2)."""
    model.eval()
    device = next(model.parameters()).device
    ids,attention = padded(states,device,0)
    hidden = model(ids,attention_mask=attention,output_hidden_states=True,use_cache=False).hidden_states[-1]
    idx = torch.tensor([len(s)-1 for s in states],device=device)
    return hidden[torch.arange(len(states),device=device),idx].detach().float()


@torch.no_grad()
def trajectory_features(model: nn.Module, prefix: list[int], tokens: list[int]) -> Tensor:
    """Features of every state before tokens[t] from one causal pass, position P+t-1 (§6.2)."""
    assert prefix and tokens
    model.eval()
    ids = torch.tensor([prefix+tokens[:-1]],device=next(model.parameters()).device)
    hidden = model(ids,attention_mask=torch.ones_like(ids),output_hidden_states=True,use_cache=False).hidden_states[-1]
    return hidden[0,len(prefix)-1:].detach().float()


@torch.no_grad()
def continuation_increments(model: nn.Module, prefix: list[int], tokens: list[int], eos: int) -> Tensor:
    """Per-token frozen likelihoods for reverse suffix labels (§7.3)."""
    if not tokens:
        return torch.empty(0,device=next(model.parameters()).device)
    ids,attention = padded([prefix+tokens],next(model.parameters()).device,eos)
    values,_ = score_tokens(model.eval()(ids,attention_mask=attention,use_cache=False).logits,ids,attention,
                          torch.tensor([len(prefix)],device=ids.device),torch.tensor([ids.shape[1]],device=ids.device),eos)
    return values[0,len(prefix)-1:]
