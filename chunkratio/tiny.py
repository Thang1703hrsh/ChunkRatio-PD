"""Offline randomly initialized causal LM/tokenizer for synthetic §12 verification."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn


class TinyLM(nn.Module):
    """Causal GRU LM; no pretrained weights, downloads or external tokenizer (§12)."""

    def __init__(self, vocab_size: int = 11, hidden_size: int = 12) -> None:
        """Shared-vocabulary causal architecture for CPU smoke (§12)."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.recurrent = nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.lm_head = nn.Linear(hidden_size,vocab_size)
        self.config = SimpleNamespace(vocab_size=vocab_size,hidden_size=hidden_size,max_position_embeddings=2048)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                output_hidden_states: bool = False, use_cache: bool = False,
                past_key_values: Tensor | None = None, position_ids: Tensor | None = None) -> SimpleNamespace:
        """Masked positions (left or right padding) leave the recurrent state unchanged (§6.2).

        The recurrent state is the cache: past_key_values continues from earlier tokens.
        """
        steps = input_ids.shape[1]
        mask = (torch.ones_like(input_ids) if attention_mask is None else attention_mask[:,-steps:]).bool()
        state = (torch.zeros(1,len(input_ids),self.config.hidden_size,device=input_ids.device)
                 if past_key_values is None else past_key_values)
        embedded, outputs = self.embedding(input_ids), []
        right_padded = bool((mask.long().cummin(-1).values == mask.long()).all())
        if mask.all() or (right_padded and not use_cache):
            # Trailing padding cannot change earlier outputs; one fused recurrent call suffices.
            hidden,state = self.recurrent(embedded,state)
            return SimpleNamespace(logits=self.lm_head(hidden),hidden_states=(hidden,) if output_hidden_states else None,
                                   past_key_values=state if use_cache else None)
        for t in range(steps):
            output,updated = self.recurrent(embedded[:,t:t+1],state)
            state = torch.where(mask[None,:,t,None],updated,state)
            outputs.append(output)
        hidden = torch.cat(outputs,dim=1)
        return SimpleNamespace(logits=self.lm_head(hidden),hidden_states=(hidden,) if output_hidden_states else None,
                               past_key_values=state if use_cache else None)

    def get_output_embeddings(self) -> nn.Linear:
        """Expose output vocabulary for canonical preflight (§5.1)."""
        return self.lm_head


class TinyTokenizer:
    """Explicit synthetic mapping with EOS=PAD for the difficult mask case (§12)."""
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = 1
    chat_template = 'synthetic:user_ids;assistant'
    special_tokens_map = {'eos_token':'<eos>','pad_token':'<eos>','bos_token':'<bos>'}

    def __init__(self, vocab_size: int = 11) -> None:
        """Match the tiny model's vocabulary (§5.1)."""
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        """Preflight output vocabulary size (§5.1)."""
        return self.vocab_size

    def get_vocab(self) -> dict[str,int]:
        """Canonical token-to-ID mapping (§5.1)."""
        return {str(i):i for i in range(self.vocab_size)}

    def get_added_vocab(self) -> dict[str,int]:
        """No added synthetic tokens (§5.1)."""
        return {}
