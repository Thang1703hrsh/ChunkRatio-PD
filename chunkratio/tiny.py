"""Offline randomly initialized causal LM/tokenizer for synthetic §12 verification."""
from __future__ import annotations

from pathlib import Path
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


QWEN_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
                 "{% if m['role'] == 'assistant' %}<think>\n\n</think>\n\n{% endif %}{{ m['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n"
                 "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}")


def tiny_qwen3_tokenizer() -> object:
    """Word-level stand-in for the Qwen3 tokenizer: <|im_end|> is EOS, <|endoftext|> is PAD (§12)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    specials = ['<|endoftext|>','<|im_start|>','<|im_end|>','<think>','</think>']
    vocab = {token:i for i,token in enumerate(specials+['user','assistant','system']+[f'w{i}' for i in range(12)])}
    core = Tokenizer(models.WordLevel(vocab,unk_token='<|endoftext|>'))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core,eos_token='<|im_end|>',pad_token='<|endoftext|>',
                                  additional_special_tokens=specials[1:2]+specials[3:],
                                  model_input_names=['input_ids','attention_mask'])
    tok.chat_template = QWEN_TEMPLATE
    return tok


def build_tiny_qwen3(root: Path, seed: int = 0) -> dict[str,Path]:
    """Save T0, a perturbed T+ and a distinct rho as tiny Qwen3 checkpoints; no download (§12).

    The output layer is padded (24 logits for a 20-token vocabulary), as in real Qwen3.
    """
    import copy
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=24,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                         num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=128,
                         eos_token_id=2,pad_token_id=0,bos_token_id=None,tie_word_embeddings=False,
                         use_sliding_window=False,sliding_window=None)
    torch.manual_seed(seed)
    base = Qwen3ForCausalLM(config)
    plus = copy.deepcopy(base)
    with torch.no_grad():
        for p in plus.parameters():
            p.add_(torch.randn_like(p)*.05)
    torch.manual_seed(seed+1)
    rho = Qwen3ForCausalLM(config)
    paths = {}
    for role,model in (('teacher_base',base),('teacher_plus',plus),('student_ref',rho)):
        paths[role] = root/role
        model.save_pretrained(paths[role])
        tiny_qwen3_tokenizer().save_pretrained(paths[role])
    return paths


def synthetic_ultrafeedback(path: Path, rows: int = 10) -> None:
    """Write UltraFeedback-Binarized-format rows over the tiny Qwen3 vocabulary (synthetic, §12)."""
    import json
    lines = []
    for i in range(rows):
        user = {'role':'user','content':f'w{i%12} w{(i+3)%12}'}
        # The last row repeats prompt p0, as UltraFeedback can hold several pairs per prompt.
        prompt_id = 'p0' if i == rows-1 else f'p{i}'
        if i == rows-1:
            user = {'role':'user','content':'w0 w3'}
        lines.append(json.dumps({'prompt_id':prompt_id,'prompt':user['content'],
                                 'chosen':[user,{'role':'assistant','content':f'w{(i+5)%12} w{(i+6)%12}'}],
                                 'rejected':[user,{'role':'assistant','content':f'w{(i+7)%12}'}],
                                 'score_chosen':8.0,'score_rejected':4.0}))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
