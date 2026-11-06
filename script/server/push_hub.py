"""Push a finished CRPD student to the Hugging Face Hub (private by default) and record the pinned revision.

One commit holds the exported student (weights, config, tokenizer), a model card and crpd_provenance.json
(resolved config, selection, held-out and benchmark results). After the commit the safetensors sha256 on the Hub
are compared with the local files, and <run>/hub.json records the repo, the commit and the check, so a later
evaluation of the pushed model can pin exactly the evaluated weights (benchmarks.py --from-hub).

Refuses a run whose checkpoint selection kept the reference (the exported student is then rho itself) and a run
without heldout.json (the card would carry no measured behaviour). Rerunning is a no-op unless the local files
changed or --update-card is given (for example after the benchmarks finished).

  python script/server/push_hub.py --run output/server/<pair>/full/runs/<name>/seed0 --repo <user>/<name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda: f.read(1 << 24),b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def check_run(run: Path) -> dict[str,Any]:
    """The run's records needed for a push; raises SystemExit when the run must not be pushed."""
    student = run/'student'
    if not (student/'config.json').exists() or not list(student.glob('*.safetensors')):
        raise SystemExit(f'{student} holds no exported model (config.json and *.safetensors)')
    diagnostics = read(run/'diagnostics.json') or {}
    policy = diagnostics.get('policy',{})
    if policy.get('selected') == 'reference':
        raise SystemExit(f'{run}: checkpoint selection kept the reference, so the student is rho; nothing to push')
    heldout = read(run/'heldout.json')
    if heldout is None:
        raise SystemExit(f'{run}: no heldout.json; run the heldout stage before pushing')
    return {'config':read(run/'resolved_config.json') or {},'diagnostics':diagnostics,'policy':policy,
            'heldout':heldout,'policy_index':read(run/'policy_index.json') or {}}


def card(run: Path, records: dict[str,Any], benchmarks: Path | None) -> str:
    """Model card: what was trained from what, how the checkpoint was chosen and what was measured."""
    cfg,policy,h = records['config'],records['policy'],records['heldout']
    models = cfg.get('models',{})
    rho,plus,base = (models.get(r,{}) for r in ('student_ref','teacher_plus','teacher_base'))
    d = h['rm_student_minus_rho']
    lines = ['---',f"base_model: {rho.get('path','unknown')}",'tags:','- crpd','- distillation','- chunkratio-pd','---','',
             f'# {run.parent.name}','',
             'CRPD (ChunkRatio-PD v1.1) student. Research checkpoint, not a product model.','',
             '| | |','|---|---|',
             f"| Student init (rho) | `{rho.get('path')}` @ `{rho.get('revision')}` |",
             f"| T+ (DPO teacher) | `{plus.get('path')}` @ `{plus.get('revision')}` |",
             f"| T0 (SFT teacher) | `{base.get('path')}` @ `{base.get('revision')}` |",
             f"| Variant | `{cfg.get('variant')}`, alpha {cfg.get('alpha')}, gamma {cfg.get('teacher_gap_scale',1)}, "
             f"pair sources `{cfg.get('pair_sources','all')}` |",
             f"| Training | {policy.get('optimizer_steps')} steps, batch {cfg.get('batch_size')}, "
             f"{records['diagnostics'].get('counts',{}).get('pairs')} pairs |",
             f"| Code revision | `{records['policy_index'].get('code_revision')}` |",'',
             '## Checkpoint selection','',
             f"Mean reward on the 64 select prompts ({policy.get('selection_rule')}): {policy.get('selection_score')} "
             f"(rho {policy.get('reference_selection_score')}). No test or benchmark result was used to choose it.",'',
             f"## Held-out behaviour ({h['prompts']} test prompts, greedy)",'',
             '| | rho | student |','|---|---:|---:|',
             f"| RM mean | {h['rm_mean']['rho']:.3f} | {h['rm_mean']['student']:.3f} |",
             f"| EOS rate | {h['eos_rate']['rho']:.3f} | {h['eos_rate']['student']:.3f} |",
             f"| Mean length (tokens) | {h['length_mean']['rho']:.0f} | {h['length_mean']['student']:.0f} |",
             f"| 4-gram repetition | {h['repetition_4gram']['rho']:.3f} | {h['repetition_4gram']['student']:.3f} |",'',
             f"RM student - rho: {d['mean']:+.3f}, bootstrap 95% [{d['bootstrap_95'][0]:+.3f}, {d['bootstrap_95'][1]:+.3f}], "
             f"wins/losses {d['wins']}/{d['losses']}.",'']
    summary = benchmarks/'summary.md' if benchmarks else None
    if summary and summary.exists():
        lines += ['## Benchmarks (lm-eval 0.4.9.2, hf fp32)','',summary.read_text(encoding='utf-8').replace('## ','### '),'']
    else:
        lines += ['Benchmarks: not run at the time of this push.','']
    return '\n'.join(lines)


def push(run: Path, repo: str, *, private: bool = True, benchmarks: Path | None = None, update_card: bool = False,
         api: Any = None) -> dict[str,Any]:
    """Commit student + card + provenance in one commit, verify the weights on the Hub, write <run>/hub.json."""
    from huggingface_hub import CommitOperationAdd
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    records = check_run(run)
    student = run/'student'
    local = {p.name:sha256(p) for p in sorted(student.glob('*.safetensors'))}
    earlier = read(run/'hub.json')
    if earlier and earlier.get('repo') == repo and earlier.get('safetensors_sha256') == local and earlier.get('verified') \
            and not update_card:
        print(f'{repo}@{earlier["revision"]}: already pushed and verified; nothing to do',flush=True)
        return earlier
    api.create_repo(repo,repo_type='model',private=private,exist_ok=True)
    provenance = {'run':run.as_posix(),'resolved_config':records['config'],'policy':records['policy'],
                  'counts':records['diagnostics'].get('counts'),'heldout':{k:v for k,v in records['heldout'].items()
                                                                            if k != 'per_prompt'},
                  'code_revision':records['policy_index'].get('code_revision'),'safetensors_sha256':local}
    if benchmarks and (benchmarks/'summary.json').exists():
        provenance['benchmarks'] = read(benchmarks/'summary.json')
    operations = [CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=str(p))
                  for p in sorted(student.iterdir()) if p.is_file() and p.name != 'README.md']
    operations += [CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card(run,records,benchmarks).encode('utf-8')),
                   CommitOperationAdd(path_in_repo='crpd_provenance.json',
                                      path_or_fileobj=json.dumps(provenance,indent=1,default=str).encode('utf-8'))]
    commit = api.create_commit(repo_id=repo,repo_type='model',operations=operations,
                               commit_message=f'CRPD student {run.parent.name}')
    revision = getattr(commit,'oid',None) or str(commit)
    info = api.model_info(repo,revision=revision,files_metadata=True)
    remote = {s.rfilename:(s.lfs.sha256 if getattr(s,'lfs',None) else None) for s in info.siblings}
    mismatch = {name:remote.get(name) for name,value in local.items() if remote.get(name) != value}
    record = {'repo':repo,'revision':revision,'private':private,'pushed_at':time.strftime('%Y-%m-%dT%H:%M:%S'),
              'safetensors_sha256':local,'verified':not mismatch,'mismatch':mismatch,
              'benchmarks_in_card':bool(benchmarks and (benchmarks/'summary.md').exists())}
    from chunkratio.artifacts import write_json
    write_json(run/'hub.json',record)
    if mismatch:
        raise SystemExit(f'{repo}@{revision}: Hub sha256 differs from the local weights for {sorted(mismatch)}')
    print(f'{repo}@{revision}: pushed ({"private" if private else "public"}), weights verified',flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run',required=True,help='Finished run folder (holds student/, diagnostics.json, heldout.json)')
    parser.add_argument('--repo',required=True,help='<user or org>/<name>')
    parser.add_argument('--public',action='store_true',help='Create the repo public (default private)')
    parser.add_argument('--benchmarks',default=None,help='benchmarks.py output folder to include in the card')
    parser.add_argument('--update-card',action='store_true',help='Commit again even if the weights are unchanged')
    args = parser.parse_args()
    push(Path(args.run),args.repo,private=not args.public,benchmarks=Path(args.benchmarks) if args.benchmarks else None,
         update_card=args.update_card)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
