"""Cross-platform external benchmark execution and cost planning (§11)."""
from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import read_json, write_json


def benchmark_plan(model: str, model_id: str, judge: str, adpa_outputs: str,
                   output: str) -> dict[str,Any]:
    """Reuse inherited MT-Bench and official external CLIs with explicit roles (§11.4)."""
    return {'model':model,'model_id':model_id,'primary_judge':judge,'jobs':[
        {'metric':'open_llm','cwd':'.','argv':[sys.executable,'-m','lm_eval','--model','hf',
           '--model_args','pretrained='+model,'--tasks','arc_challenge,hellaswag,truthfulqa_mc2,mmlu,winogrande,gsm8k',
           '--num_fewshot','0','--batch_size','auto','--log_samples','--output_path',output+'/open_llm']},
        {'metric':'mt_bench_answers','cwd':'mtbench/FastChat','argv':[sys.executable,'-m','fastchat.llm_judge.gen_model_answer',
           '--model-path',model,'--model-id',model_id]},
        {'metric':'mt_bench','cwd':'mtbench/FastChat','argv':[sys.executable,'-m','fastchat.llm_judge.gen_judgment',
           '--model-list',model_id,'--judge-model',judge,'--mode','single']},
        {'metric':'ae_vs_adpa','cwd':'.','argv':[sys.executable,'-m','alpaca_eval','evaluate','--model_outputs',output+'/alpaca_outputs.json',
           '--reference_outputs',adpa_outputs,'--annotators_config',output+'/primary_annotator.yaml','--output_path',output+'/ae_vs_adpa']},
        {'metric':'ae2','cwd':'.','argv':[sys.executable,'-m','alpaca_eval','evaluate','--model_outputs',output+'/alpaca_outputs.json',
           '--output_path',output+'/ae2']},
        # Arena-Hard has its own repository/config lifecycle; make that dependency explicit.
        {'metric':'arena_hard','cwd':'external/arena-hard-auto','argv':[sys.executable,'gen_judgment.py']},
        {'metric':'diversity_optional','cwd':'.','argv':[sys.executable,'diversity_metrics/compute_diversity.py',
           '--infile',output+'/diversity_generations.jsonl','--out',output+'/diversity.json']}],
        'requires':{'lm_eval':'0.4.9.2','fastchat':'vendored','alpaca_eval':'PIN_REQUIRED','arena_hard':'PIN_REQUIRED'},
        'notes':['Generate benchmark-specific output files using pinned official generation configurations.',
                 'Provide primary_annotator.yaml with primary_judge; official AE2/Arena judges may differ.',
                 'All six Open LLM tasks use declared zero-shot settings here; report them rather than silently claiming a different preset.',
                 'RM held-out scoring uses the pinned selection RM on test prompts, never select prompts.']}


def execute_plan(plan: dict[str,Any], output: Path) -> None:
    """Run configured argv arrays without shell interpolation; record wall-time/status (§11)."""
    for package,version in plan['requires'].items():
        if version == 'PIN_REQUIRED':
            raise ValueError(f'pin external benchmark version: {package}')
        if version != 'vendored' and importlib.metadata.version(package.replace('_','-')) != version:
            raise ValueError(f'benchmark dependency version mismatch: {package}')
    outcomes = []
    for job in plan['jobs']:
        start = time.perf_counter()
        completed = subprocess.run(job['argv'],cwd=Path(job['cwd']),check=False)
        outcomes.append({'metric':job['metric'],'wall_seconds':time.perf_counter()-start,'exit_code':completed.returncode})
        write_json(output,{'jobs':outcomes,'gpu_hours':None,'note':'wall time is measured; GPU utilization requires external hardware profiling'})
        if completed.returncode:
            raise RuntimeError(f'benchmark job failed: {job["metric"]}')


def estimate_cost(states: int, trajectories: int, average_suffix: float,
                  generated_tokens_per_gpu_second: float | None = None,
                  scored_tokens_per_gpu_second: float | None = None) -> dict[str,Any]:
    """Spec method equation 17 plus two-teacher scoring; no invented hardware rate (§11.4)."""
    generated = states*4*32+trajectories*average_suffix
    scored = states*4*32*3+trajectories*average_suffix*2
    estimate = None
    if generated_tokens_per_gpu_second and scored_tokens_per_gpu_second:
        estimate = (generated/generated_tokens_per_gpu_second+scored/scored_tokens_per_gpu_second)/3600
    return {'generated_tokens':generated,'scored_tokens':scored,'estimated_gpu_hours':estimate,
            'assumptions':'L=32 K=4; aggregate single-GPU rates; excludes prefix prefill, head fit, policy, preparation and tuning',
            'status':'conditional estimate' if estimate is not None else 'unavailable: provide measured target-hardware throughput'}
