"""Explicit local preference preprocessing to canonical prompt-only inputs (§§5,7,11)."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifacts import atomic_bytes, digest, read_jsonl, write_json


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


def ultrafeedback_pairs(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """UltraFeedback Binarized rows: shared conversation prefix, last assistant turns as the pair (§11, NARVAL)."""
    pairs, seen = [],set()
    for row in rows:
        # Keep the first pair per prompt id in file order; later duplicates are recorded by the caller.
        if row['prompt_id'] in seen:
            continue
        seen.add(row['prompt_id'])
        chosen, rejected = row['chosen'], row['rejected']
        if chosen[:-1] != rejected[:-1] or chosen[-1]['role'] != 'assistant' or rejected[-1]['role'] != 'assistant':
            raise ValueError(f"malformed UltraFeedback pair {row.get('prompt_id')}")
        pairs.append({'messages':chosen[:-1],'chosen':chosen[-1]['content'],'rejected':rejected[-1]['content'],
                      'group_id':row['prompt_id']})
    return pairs


def rank_split(group_ids: list[str], seed: int, head_val_fraction: float, select_fraction: float) -> dict[str,str]:
    """Hold out head_val and select by prompt id before any generation; never touches test (§7.1).

    Groups are ranked by sha256([seed, 'uf-split', id]); the first round(f*n) (at least one when
    f > 0) go to head_val, the next to select, the rest to train.
    """
    unique = sorted(set(group_ids),key=lambda g:digest([seed,'uf-split',g]))
    counts = [max(1,round(f*len(unique))) if f > 0 else 0 for f in (head_val_fraction,select_fraction)]
    if sum(counts) >= len(unique):
        raise ValueError('held-out fractions leave no train prompts')
    split = {g:'head_val' for g in unique[:counts[0]]}
    split.update({g:'select' for g in unique[counts[0]:sum(counts)]})
    split.update({g:'train' for g in unique[sum(counts):]})
    return split


def sharegpt_rows(rows: list[dict[str,Any]]) -> tuple[list[dict[str,Any]],int]:
    """ShareGPT conversations (Deita) -> SFT rows: earlier turns are the prompt, the last
    assistant turn is chosen; rows that do not end with an assistant turn are skipped and counted."""
    roles = {'human':'user','user':'user','gpt':'assistant','assistant':'assistant','system':'system'}
    output, skipped = [],0
    for row in rows:
        turns = [{'role':roles[t['from']],'content':t['value']} for t in row['conversations']]
        if len(turns) < 2 or turns[-1]['role'] != 'assistant':
            skipped += 1
            continue
        output.append({'messages':turns[:-1],'chosen':turns[-1]['content'],'rejected':None})
    return output,skipped


def fit_conversation(row: dict[str,Any], length: Callable[[list[dict[str,str]]],int], limit: int) -> dict[str,Any] | None:
    """Supervise the latest assistant turn whose whole conversation prefix fits `limit` tokens.

    Earlier turns stay as context and no response is cut, so long Deita dialogues still give
    one SFT row instead of being dropped by the length preflight. None if no turn fits.
    """
    turns = row['messages']+[{'role':'assistant','content':row['chosen']}]
    ends = [i for i,t in enumerate(turns) if t['role'] == 'assistant' and i > 0]
    low,high,best = 0,len(ends)-1,None
    while low <= high:
        middle = (low+high)//2
        if length(turns[:ends[middle]+1]) <= limit:
            best,low = ends[middle],middle+1
        else:
            high = middle-1
    if best is None:
        return None
    return {**row,'messages':turns[:best],'chosen':turns[best]['content']}


def prepare_data(source: Path, tokenizer_path: str, revision: str, destination: Path, seed: int,
                 fmt: str = 'auto', assign_split: str | None = None,
                 template_kwargs: dict[str,Any] | None = None,
                 head_val_fraction: float = .05, select_fraction: float = .05,
                 eos_token: str | None = None, fit_max_length: int | None = None) -> None:
    """Serialize prompts once; strict template seam; retain human pairs for anchors only (§5.1)."""
    import json
    from transformers import AutoTokenizer
    from .collection import prepare_prompts
    tok = AutoTokenizer.from_pretrained(tokenizer_path,revision=revision)
    if eos_token is not None:
        # Base checkpoints may declare another EOS; every role must stop at the end-of-turn token.
        if eos_token not in tok.get_vocab():
            raise ValueError(f'{eos_token} is not in the tokenizer vocabulary')
        tok.eos_token = eos_token
    rows = read_jsonl(source)
    template_kwargs = template_kwargs or {}
    ultrafeedback = fmt == 'ultrafeedback' or (fmt == 'auto' and rows and 'prompt_id' in rows[0]
                                               and isinstance(rows[0].get('chosen'),list))
    sharegpt = fmt == 'sharegpt' or (fmt == 'auto' and rows and 'conversations' in rows[0])
    skipped = 0
    if rows and 'helpfulness' in rows[0]:
        rows = helpsteer_pairs(rows)
    elif sharegpt:
        rows,skipped = sharegpt_rows(rows)
        trimmed = unfit = 0
        if fit_max_length is not None:
            def length(turns: list[dict[str,str]]) -> int:
                """Templated token count of a whole conversation prefix."""
                return len(tok.apply_chat_template(turns,tokenize=True,add_generation_prompt=False,**(template_kwargs or {})))
            fitted = []
            for row in rows:
                fit = fit_conversation(row,length,fit_max_length)
                if fit is None:
                    unfit += 1
                    continue
                trimmed += fit['messages'] != row['messages']
                fitted.append(fit)
            rows = fitted
    elif ultrafeedback:
        raw_ids = [r['prompt_id'] for r in rows]
        rows = ultrafeedback_pairs(rows)
        duplicates = sorted({g for g in raw_ids if raw_ids.count(g) > 1})
        if assign_split is None:
            assignment = rank_split([r['group_id'] for r in rows],seed,head_val_fraction,select_fraction)
            rule = "rank by sha256([seed,'uf-split',prompt_id]); first round(f*n) head_val, next select, rest train"
        else:
            assignment = {r['group_id']:assign_split for r in rows}
            rule = f'all rows assigned to {assign_split}'
        for row in rows:
            row['split'] = assignment[row['group_id']]
        write_json(destination.with_name(destination.stem+'.splits.json'),
                   {'source':source.name,'format':'ultrafeedback_binarized','seed':seed,'rule':rule,
                    'fractions':{'head_val':head_val_fraction,'select':select_fraction},
                    'counts':{s:sum(v == s for v in assignment.values()) for s in sorted(set(assignment.values()))},
                    'assignment_sha256':digest(sorted(assignment.items())),
                    'duplicate_prompt_ids_first_pair_kept':duplicates,'rows_in':len(raw_ids),'rows_out':len(rows)})
    if assign_split is not None and not ultrafeedback:
        for row in rows:
            row['split'] = assign_split
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
        prefix = tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,**template_kwargs)
        result = {'messages':messages,'prompt_ids':prefix,'tokenizer_revision':revision}
        for label,text in (('chosen',chosen),('rejected',rejected)):
            if text is None:
                continue
            full = tok.apply_chat_template(messages+[{'role':'assistant','content':text}],tokenize=True,
                                           add_generation_prompt=False,**template_kwargs)
            if full[:len(prefix)] != prefix:
                raise ValueError('template prompt/response seam differs; no silent re-tokenization allowed')
            continuation = full[len(prefix):]
            # The collection sampler stops only at tokenizer EOS; a template whose assistant turn
            # ends with another token would silently run every response to H (§5.1).
            if tok.eos_token_id not in continuation:
                raise ValueError('chat template does not end the assistant turn with tokenizer EOS; '
                                 'set eos_token to the end-of-turn token for every model')
            continuation = continuation[:continuation.index(tok.eos_token_id)+1]
            result[label+'_ids'] = continuation
        for key in ('split','group_id'):
            if key in row:
                result[key] = row[key]
        output.append(result)
    if sharegpt:
        # SFT data: keep the first answer per canonical prompt; record what was dropped.
        seen, unique = set(),[]
        for row in output:
            key = tuple(row['prompt_ids'])
            if key not in seen:
                seen.add(key)
                unique.append(row)
        write_json(destination.with_name(destination.stem+'.sft.json'),
                   {'source':source.name,'format':'sharegpt','rows_in':len(output)+skipped,'rows_out':len(unique),
                    'skipped_not_ending_with_assistant':skipped,'duplicate_prompts_dropped':len(output)-len(unique),
                    'eos_token':tok.eos_token,'fit_max_length':fit_max_length,
                    'turns_trimmed_to_fit':trimmed,'dropped_no_turn_fits':unfit})
        output = unique
    prepared = prepare_prompts(output,seed)
    atomic_bytes(destination,(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in prepared)).encode())
