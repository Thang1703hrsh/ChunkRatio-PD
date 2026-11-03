"""Modal CPU check: does the fitted value head beat constant and budget-only predictors on head_val?

Reads the x10 pilot's correction shards (labels, budgets, detached rho features) and its frozen head from
the pilot volume, read-only. No GPU and no new generation. Reported on head_val trajectories:

- the fit-head objective (mean over trajectories of SSE / H) for zero, train-mean constant, a budget-only
  least-squares fit (1, b, b^2 on the remaining-budget fraction, fitted on train) and the head;
- the same at each trajectory's first position only, which is the endpoint value build-targets uses;
- within-state: endpoints that share a parent state, demeaned per state. Pair targets only use differences
  between siblings, so this is the part of mu_hat that can change a pair's label.

Usage (repository root), on the workspace that owns crpd-qwen3-pilot-hub-x10, or locally on a downloaded
copy of the two cache folders (`modal volume get` of correction-<key> and head-<key>):

  modal run script/modal/head_check.py
  python script/modal/head_check.py --cache <folder holding both> --output output/head-check-x10.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SOURCE = Path('/source')
CACHE = SOURCE/'output/narval/qwen3_1p7b_to_0p6b_hub/pilot/cache'
CORRECTION = 'correction-aae9d2b2ef3672653e70'
HEAD = 'head-7a3e249e6a6f4abe31a9'
HORIZON = 512


def objective(predictions: list, labels: list) -> float:
    """fit-head objective: mean over trajectories of SSE / H."""
    return sum(float(((p-y)**2).sum())/HORIZON for p,y in zip(predictions,labels))/len(labels)


def correlation(x: list[float], y: list[float]) -> float:
    """Pearson correlation; nan when either side is constant."""
    n = len(x)
    mx,my = sum(x)/n,sum(y)/n
    sxy = sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx,syy = sum((a-mx)**2 for a in x),sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx > 0 and syy > 0 else float('nan')


def r2(predictions: list[float], labels: list[float]) -> float:
    """1 - SSE/SST around the labels' own mean."""
    mean = sum(labels)/len(labels)
    sst = sum((y-mean)**2 for y in labels)
    return 1-sum((p-y)**2 for p,y in zip(predictions,labels))/sst if sst > 0 else float('nan')


def parent(endpoint: dict, prefixes: list[dict]) -> str | None:
    """The prefix seed an endpoint extends: the longest prefix seed whose prefix_ids start its prefix_ids."""
    best = None
    for state in prefixes:
        ids = state['prefix_ids']
        if len(ids) < len(endpoint['prefix_ids']) and endpoint['prefix_ids'][:len(ids)] == ids:
            if best is None or len(ids) > len(best['prefix_ids']):
                best = state
    return best['id'] if best else None


def run_check(cache: Path, correction: str, head_dir: str) -> dict:
    """Score zero, constant, budget-only and the fitted head on head_val correction trajectories."""
    import torch
    from torch import nn
    folder = cache/correction
    # Each completed job is one <digest>.jsonl record with a <digest>.features.pt attachment.
    shards = {}
    for path in sorted(folder.glob('*.jsonl')):
        shards[path.name.removesuffix('.jsonl')] = json.loads(path.read_text(encoding='utf-8'))['payload']
    train = [r for r in shards.values() if r['seed']['split'] == 'train']
    val = [(name,r) for name,r in shards.items() if r['seed']['split'] == 'head_val']
    saved = torch.load(cache/head_dir/'head.pt',map_location='cpu',weights_only=True)
    head = nn.Sequential(nn.Linear(saved['metadata']['hidden_size']+1,256),nn.GELU(),nn.Linear(256,1))
    head.load_state_dict({k.removeprefix('net.'):v for k,v in saved['state_dict'].items()})
    head.eval()
    # Constant minimizing the summed-SSE objective on train: pooled mean over all train positions.
    pooled = torch.cat([torch.tensor(r['labels'],dtype=torch.float64) for r in train])
    constant = float(pooled.mean())
    # Budget-only least squares on train positions: label ~ 1 + b + b^2.
    b = torch.cat([torch.tensor(r['budgets'],dtype=torch.float64) for r in train])
    coef = torch.linalg.lstsq(torch.stack([torch.ones_like(b),b,b*b],dim=1),pooled[:,None]).solution[:,0]
    labels,zero,const,budget,learned = [],[],[],[],[]
    with torch.no_grad():
        for name,row in val:
            y = torch.tensor(row['labels'],dtype=torch.float64)
            bb = torch.tensor(row['budgets'],dtype=torch.float64)
            features = torch.load(folder/f'{name}.features.pt',map_location='cpu',weights_only=True).float()
            labels.append(y)
            zero.append(torch.zeros_like(y))
            const.append(torch.full_like(y,constant))
            budget.append(coef[0]+coef[1]*bb+coef[2]*bb*bb)
            learned.append(head(torch.cat([features,bb.float()[:,None]],dim=-1)).squeeze(-1).double())
    predictors = {'zero':zero,'train_mean':const,'budget_only':budget,'head':learned}
    result = {'trajectories':{'train':len(train),'head_val':len(val)},'positions_head_val':int(sum(len(y) for y in labels)),
              'label_mean_train':constant,'label_std_train':float(pooled.std()),'budget_coefficients':coef.tolist(),
              'stored_head_val':saved['metadata']['head_val'],'objective':{},'first_position':{},'within_state':{}}
    first = [float(y[0]) for y in labels]
    for name,pred in predictors.items():
        result['objective'][name] = objective(pred,labels)
        p0 = [float(p[0]) for p in pred]
        result['first_position'][name] = {'mse':sum((p-y)**2 for p,y in zip(p0,first))/len(first),'r2':r2(p0,first),
                                          'corr':correlation(p0,first)}
    # Within-state: endpoints of the same parent state, both sides demeaned per state.
    seeds = [r['seed'] for _,r in val]
    prefixes = [s for s in seeds if s.get('seed_source') == 'prefix']
    groups: dict[str,list[int]] = {}
    for i,seed in enumerate(seeds):
        if seed.get('seed_source') != 'prefix':
            key = parent(seed,prefixes)
            if key is not None:
                groups.setdefault(key,[]).append(i)
    groups = {k:v for k,v in groups.items() if len(v) >= 2}
    result['within_state'] = {'states':len(groups),'endpoints':sum(len(v) for v in groups.values())}
    for name,pred in predictors.items():
        dy,dp = [],[]
        for members in groups.values():
            ys = [first[i] for i in members]
            ps = [float(pred[i][0]) for i in members]
            dy += [y-sum(ys)/len(ys) for y in ys]
            dp += [p-sum(ps)/len(ps) for p in ps]
        result['within_state'][name] = {'corr':correlation(dp,dy),'r2':r2(dp,dy)} if dy else None
    result['seed_keys'] = sorted(seeds[0])
    return result


def save(result: dict, output: str) -> None:
    """Write the result JSON and print it."""
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache',required=True,help='folder holding the correction and head folders')
    parser.add_argument('--correction',default=CORRECTION)
    parser.add_argument('--head',default=HEAD)
    parser.add_argument('--output',default='output/head-check-x10.json')
    args = parser.parse_args()
    save(run_check(Path(args.cache),args.correction,args.head),args.output)
else:
    import modal

    app = modal.App('crpd-head-check')
    image = modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.6.0',
                                                                      index_url='https://download.pytorch.org/whl/cpu')

    @app.function(image=image,cpu=4.0,memory=16384,timeout=3600,
                  volumes={str(SOURCE):modal.Volume.from_name('crpd-qwen3-pilot-hub-x10').read_only()})
    def check(correction: str, head_dir: str) -> dict:
        """The check on the pilot volume."""
        return run_check(CACHE,correction,head_dir)

    @app.local_entrypoint()
    def main(correction: str = CORRECTION, head: str = HEAD, output: str = 'output/head-check-x10.json') -> None:
        """Run the check on Modal and save its JSON locally."""
        save(check.remote(correction,head),output)
