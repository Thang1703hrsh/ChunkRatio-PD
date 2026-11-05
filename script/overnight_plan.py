"""Pure planning helpers for bounded CRPD smoke experiments; no GPU or cloud imports."""
from __future__ import annotations

import copy
import math
from typing import Any


SMOKE_TRIALS = (
    {'name':'compact_main','variant':'crpd_main','candidates':2,'head_positions':'boundary'},
    {'name':'compact_raw','variant':'crpd_raw','candidates':2,'head_positions':'boundary'},
    {'name':'all_positions_main','variant':'crpd_main','candidates':2,'head_positions':'all'},
    {'name':'k4_main','variant':'crpd_main','candidates':4,'head_positions':'boundary'},
    {'name':'k4_raw','variant':'crpd_raw','candidates':4,'head_positions':'boundary'},
    {'name':'conservative_main','variant':'crpd_main','candidates':2,'head_positions':'boundary',
     'alpha':.01,'learning_rate':5e-7},
)


def reservation_cents(seconds: int, gpu: str) -> int:
    """Round up a conservative reservation including 180 seconds of container overhead.

    Actual requested resource rates are lower: A100 + 4 CPU/48 GiB ~3.08 USD/h,
    L40S + 2 CPU/24 GiB ~2.24 USD/h. The ledger reserves 4.50/3.00 USD/h.
    This is a runtime/resource bound, not a provider-side invoice limit.
    """
    if type(seconds) is not int or seconds <= 0 or gpu not in ('A100-80GB','L40S'):
        raise ValueError('positive duration and known GPU required')
    return math.ceil((seconds+180)*({'A100-80GB':450,'L40S':300}[gpu])/3600)


def reserve(ledger: dict[str,Any], key: str, seconds: int, gpu: str) -> None:
    """Reserve full worst-case duration before launching, without refunding short runs."""
    if key in ledger['jobs']:
        raise ValueError('a job cannot be launched twice')
    amount = reservation_cents(seconds,gpu)
    if ledger['reserved_cents']+amount > ledger['limit_cents']:
        raise ValueError('budget exhausted')
    ledger['reserved_cents'] += amount
    ledger['jobs'][key] = {'reserved_cents':amount,'seconds':seconds,'gpu':gpu,'status':'reserved'}


def configure_trial(base: dict[str,Any], level: str, trial: dict[str,Any], root: str) -> dict[str,Any]:
    """Use tested execution knobs; explicitly declare the smaller data/update budgets."""
    from pathlib import Path
    from chunkratio.fast_run import fast_config
    if level not in ('smoke','medium'):
        raise ValueError('unknown data level')
    cfg = fast_config(base,Path(root)/level,Path(root)/level/'prompts.jsonl',trial['variant'])
    cfg.update({k:v for k,v in trial.items() if k != 'name'})
    cfg.update(output=f"{root}/{level}/{trial['name']}",steps=30 if level == 'smoke' else 60,
               batch_size=16 if level == 'smoke' else 32,selection_interval=15 if level == 'smoke' else 30)
    cfg['pilot_assumptions'].update(
        policy_steps=f"{cfg['steps']} steps x {cfg['batch_size']} pairs; selection every {cfg['selection_interval']}",
        pilot_subset='64/16/16 prompt groups' if level == 'smoke' else '256/64/32 prompt groups',
        fast_x10='Inherited execution optimizations only; this is an explicitly budgeted smoke/medium experiment.',
        compact=f"K={cfg['candidates']}, endpoint-only R=1 correction, head_positions={cfg['head_positions']}",
        overnight='Validation-only screening; fixed compute budget, not a causal K-only ablation; test is final reporting only.')
    return cfg


def validation_delta(result: dict[str,Any]) -> float:
    """Rank completed candidates using select-split rewards only, never test metrics."""
    if result.get('status') != 'success':
        return -math.inf
    policy = result['policy']
    values = (policy.get('selection_score'),policy.get('reference_selection_score'))
    if any(v is None or not math.isfinite(v) for v in values):
        return -math.inf
    return values[0]-values[1]


def promotion_candidates(results: dict[str,dict[str,Any]]) -> list[str]:
    """Promote at most one main and one raw with positive validation gain; exploratory only."""
    selected = []
    for variant in ('crpd_main','crpd_raw'):
        names = [name for name,row in results.items() if row.get('variant') == variant]
        if names:
            best = max(names,key=lambda name:validation_delta(results[name]))
            if validation_delta(results[best]) > 0:
                selected.append(best)
    return selected
