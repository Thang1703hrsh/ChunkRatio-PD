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
                output_hidden_states: bool = False, use_cache: bool = False) -> SimpleNamespace:
        """Right-padded suffixes cannot affect earlier hidden states (§6.2)."""
        hidden,_ = self.recurrent(self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden),hidden_states=(hidden,) if output_hidden_states else None)

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
