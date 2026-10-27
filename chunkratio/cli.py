"""Cross-platform stage CLI, lazy optional dependencies and explicit failure (§10)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_bytes, read_json, read_jsonl, write_json
from .config import STAGES, load_config


def main(argv: list[str] | None = None) -> int:
    """Dispatch pipeline, reference, protocol and reporting commands (§10)."""
    parser = argparse.ArgumentParser(prog='python -m chunkratio',description='ChunkRatio-PD v1.1, spec revision 2')
    commands = parser.add_subparsers(dest='command',required=True)
    for name in STAGES:
        sub = commands.add_parser(name)
        sub.add_argument('--config',required=True)
    sub = commands.add_parser('smoke')
    sub.add_argument('--config',default='config/crpd/smoke.json')
    sub.add_argument('--output',default='output/smoke')
    sub = commands.add_parser('toy-check')
    sub.add_argument('--output',default='output/reference-checks.json')
    sub = commands.add_parser('learned-tree')
    sub.add_argument('--output',default='output/learned-tree.json')
    sub.add_argument('--steps',type=int,default=200)
    sub = commands.add_parser('diagnose-displacement')
    sub.add_argument('--config',required=True)
    sub.add_argument('--prompts',type=int,required=True)
    sub.add_argument('--replicates',type=int,default=16)
    sub.add_argument('--output',required=True)
    sub = commands.add_parser('report')
    sub.add_argument('--input',required=True)
    sub.add_argument('--output',required=True)
    sub.add_argument('--allow-incomplete',action='store_true')
    sub = commands.add_parser('benchmark-plan')
    for name in ('model','model-id','judge','adpa-outputs','output'):
        sub.add_argument('--'+name,required=True)
    sub = commands.add_parser('benchmark-run')
    sub.add_argument('--plan',required=True)
    sub.add_argument('--output',required=True)
    sub = commands.add_parser('estimate-cost')
    sub.add_argument('--states',type=int,required=True)
    sub.add_argument('--trajectories',type=int,required=True)
    sub.add_argument('--average-suffix',type=float,required=True)
    sub.add_argument('--generation-rate',type=float)
    sub.add_argument('--scoring-rate',type=float)
    sub = commands.add_parser('rm-evaluate')
    sub.add_argument('--config',required=True)
    sub.add_argument('--model',required=True)
    sub.add_argument('--output',required=True)
    sub = commands.add_parser('prepare-data')
    sub.add_argument('--input',required=True)
    sub.add_argument('--tokenizer',required=True)
    sub.add_argument('--revision',required=True)
    sub.add_argument('--output',required=True)
    sub.add_argument('--seed',type=int,default=0)
    sub.add_argument('--format',choices=('auto','ultrafeedback'),default='auto')
    sub.add_argument('--assign-split',choices=('train','head_val','select','test'))
    sub.add_argument('--chat-template-kwargs',default='{}')
    sub.add_argument('--head-val-fraction',type=float,default=.05)
    sub.add_argument('--select-fraction',type=float,default=.05)
    for name in ('train-sft','train-teacher-dpo'):
        sub = commands.add_parser(name)
        sub.add_argument('--config',required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'toy-check':
            output = Path(args.output)
            output.parent.mkdir(parents=True,exist_ok=True)
            return subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'reference/ChunkRatio_Math_Checks.py'),str(output)]).returncode
        if args.command == 'validate-config':
            cfg = load_config(args.config)
            print(json.dumps({'status':'config_valid','variant':cfg['variant'],
                              'note':'checkpoint/tokenizer/identity preflight runs when models are loaded'},indent=2))
            return 0
        if args.command == 'diagnose-displacement':
            from .diagnostics import diagnose
            print(json.dumps(diagnose(load_config(args.config),args.prompts,args.replicates,Path(args.output)),indent=2))
            return 0
        if args.command == 'report':
            from .evaluation import report_tables
            atomic_bytes(Path(args.output),report_tables(read_jsonl(Path(args.input)),args.allow_incomplete).encode())
            return 0
        if args.command == 'benchmark-plan':
            from .experiment import benchmark_plan
            write_json(Path(args.output),benchmark_plan(args.model,args.model_id,args.judge,args.adpa_outputs,str(Path(args.output).parent)))
            return 0
        if args.command == 'benchmark-run':
            from .experiment import execute_plan
            execute_plan(read_json(Path(args.plan)),Path(args.output))
            return 0
        if args.command == 'estimate-cost':
            from .experiment import estimate_cost
            print(json.dumps(estimate_cost(args.states,args.trajectories,args.average_suffix,
                                          args.generation_rate,args.scoring_rate),indent=2))
            return 0
        if args.command == 'rm-evaluate':
            from .evaluation import evaluate_reward_model
            print(json.dumps(evaluate_reward_model(load_config(args.config),args.model,Path(args.output)),indent=2))
            return 0
        if args.command == 'prepare-data':
            from .data import prepare_data
            prepare_data(Path(args.input),args.tokenizer,args.revision,Path(args.output),args.seed,args.format,
                         args.assign_split,json.loads(args.chat_template_kwargs),args.head_val_fraction,args.select_fraction)
            return 0
        if args.command in ('train-sft','train-teacher-dpo'):
            from .preparation import prepare_checkpoint
            prepare_checkpoint(read_json(Path(args.config)),args.command)
            return 0
        if args.command == 'smoke':
            from .smoke import smoke
            result = smoke(Path(args.output),load_config(args.config))
            print(result['summary'])
            return 0
        if args.command == 'learned-tree':
            from .learned_tree import run
            result = run(args.steps)
            write_json(Path(args.output),result)
            print('LEARNED TREE COMPLETED: synthetic finite-data experiment')
            return 0
        from .pipeline import pipeline
        result = pipeline(load_config(args.config),args.command)
        print(json.dumps(result,indent=2))
        return 0
    except ModuleNotFoundError as exc:
        message = f'DEPENDENCY BLOCKED: {exc.name} is unavailable in active Python {sys.version.split()[0]}'
        print(message,file=sys.stderr)
        if args.command == 'smoke':
            write_json(Path(args.output)/'STATUS.json',{'status':'blocked','reason':message,'synthetic':True})
        return 2
    except (ValueError,RuntimeError,AssertionError,FileNotFoundError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1
