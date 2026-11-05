"""Opt-in execution optimizations; no changes to CRPD scores or sampling rules."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .scoring import length_groups, score_batch, sibling_logprobs


@torch.no_grad()
def score_candidate_records(records: list[dict[str,Any]], models: dict[str,nn.Module],
                            eos: int, tokens_per_batch: int, score_eos: bool = True) -> None:
    """Score across states within a padded-token budget, preserving row order.

    Generation, deduplication, provenance and RNG streams are untouched. Different
    batch shapes can cause floating point differences on real accelerators.
    """
    rows = [(record['state']['prefix_ids'],candidate)
            for record in records for candidate in record['candidates']]
    groups = length_groups([len(prefix)+len(c['tokens']) for prefix,c in rows],tokens_per_batch)
    for source,model in models.items():
        for group in groups:
            scores = score_batch(model,[rows[i][0] for i in group],
                                 [rows[i][1]['tokens'] for i in group],eos,score_eos=score_eos)
            for i,score in zip(group,scores.tolist()):
                rows[i][1]['logp_'+source] = score


@torch.no_grad()
def score_candidate_siblings(records: list[dict[str,Any]], models: dict[str,nn.Module],
                             eos: int, tokens_per_batch: int, max_states: int = 48, score_eos: bool = True) -> None:
    """Score each state's siblings on one prefix pass per model, many states per forward.

    States are length-sorted and cut so states x longest prefix stays within the token budget, and at most
    `max_states` per forward: the siblings' FP32 logits ([siblings, chunk, vocab], about 19.5 MB per
    sibling for Qwen3) otherwise grow with every short prefix the budget admits (16k tokens of short
    prefixes reached 61 GB on an A100 for T+ alone). Each prefix runs once and its cache is repeated for
    its siblings (sibling_logprobs). Scores equal per-state scoring up to FP32 rounding.
    """
    groups = []
    for group in length_groups([len(r['state']['prefix_ids'])+max(len(c['tokens']) for c in r['candidates'])
                                for r in records],tokens_per_batch):
        groups += [group[i:i+max_states] for i in range(0,len(group),max_states)]
    for source,model in models.items():
        for group in groups:
            sums = sibling_logprobs(model,[records[i]['state']['prefix_ids'] for i in group],
                                    [[c['tokens'] for c in records[i]['candidates']] for i in group],eos,score_eos)
            for i,values in zip(group,sums):
                for candidate,value in zip(records[i]['candidates'],values.tolist()):
                    candidate['logp_'+source] = value
