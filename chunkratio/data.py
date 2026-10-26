"""Explicit local preference preprocessing to canonical prompt-only inputs (§§5,7,11)."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from .artifacts import atomic_bytes, read_jsonl


def helpsteer_pairs(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Prepare helpfulness comparisons with ties removed before prompt-only use (§11.1)."""
    groups: dict[str,list[dict[str,Any]]] = {}
    for row in rows:
        groups.setdefault(row['prompt'],[]).append(row)
    pairs = []
    for prompt,answers in sorted(groups.items()):
        for a,b in itertools.combinations(answers,2):
            if a['helpfulness'] == b['helpfulness']:
                continue
            chosen,rejected = (a,b) if a['helpfulness'] > b['helpfulness'] else (b,a)
            pairs.append({'messages':[{'role':'user','content':prompt}],
                          'chosen':chosen['response'],'rejected':rejected['response']})
    return pairs


def prepare_data(source: Path, tokenizer_path: str, revision: str, destination: Path, seed: int) -> None:
    """Serialize prompts once; strict template seam; retain human pairs for anchors only (§5.1)."""
    import json
    from transformers import AutoTokenizer
    from .collection import prepare_prompts
    tok = AutoTokenizer.from_pretrained(tokenizer_path,revision=revision)
    rows = read_jsonl(source)
    if rows and 'helpfulness' in rows[0]:
        rows = helpsteer_pairs(rows)
    output = []
    for row in rows:
        messages = row.get('messages')
        if messages is None:
            # DPO-MIX exported chat pairs retain the common conversation prefix.
            if not isinstance(row.get('chosen'),list) or row['chosen'][:-1] != row['rejected'][:-1]:
                raise ValueError('expected messages/chosen/rejected or matching chosen/rejected chat arrays')
            messages = row['chosen'][:-1]
            chosen,rejected = row['chosen'][-1]['content'],row['rejected'][-1]['content']
        else:
            chosen,rejected = row['chosen'],row['rejected']
        prefix = tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        result = {'messages':messages,'prompt_ids':prefix,'tokenizer_revision':revision}
        for label,text in (('chosen',chosen),('rejected',rejected)):
            full = tok.apply_chat_template(messages+[{'role':'assistant','content':text}],tokenize=True,add_generation_prompt=False)
            if full[:len(prefix)] != prefix:
                raise ValueError('template prompt/response seam differs; no silent re-tokenization allowed')
            continuation = full[len(prefix):]
            if tok.eos_token_id in continuation:
                continuation = continuation[:continuation.index(tok.eos_token_id)+1]
            result[label+'_ids'] = continuation
        for key in ('split','group_id'):
            if key in row:
                result[key] = row[key]
        output.append(result)
    prepared = prepare_prompts(output,seed)
    atomic_bytes(destination,(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in prepared)).encode())
