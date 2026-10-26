"""Resolve both preference tracks / three seeds without launching unless --execute (§11)."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Generate fully pinned run configs from a user-resolved protocol (§11.1)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True,help='Protocol JSON with actual checkpoint revisions and verified DPO reference')
    parser.add_argument('--dpo-prompts',required=True)
    parser.add_argument('--helpsteer-prompts',required=True)
    parser.add_argument('--output',default='output/protocol')
    parser.add_argument('--variants',nargs='+',default=['crpd_main','crpd_raw'])
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Import from the repository even when launched as python script/file.py.
    sys.path.insert(0,str(root))
    from chunkratio.config import validate
    from chunkratio.artifacts import write_json
    base = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    for dataset,prompts in [('DPO-MIX-7K',args.dpo_prompts),('HelpSteer2',args.helpsteer_prompts)]:
        for seed in (0,1,2):
            for variant in args.variants:
                cfg = copy.deepcopy(base)
                folder = Path(args.output)/dataset/variant/str(seed)
                cfg.update(variant=variant,seed=seed,prompts=prompts,output=str(folder),dataset=dataset)
                validate(cfg)
                path = folder/'input_config.json'
                write_json(path,cfg)
                command = [sys.executable,'-m','chunkratio','evaluate','--config',str(path)]
                print(json.dumps(command))
                if args.execute:
                    subprocess.run(command,cwd=root,check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
