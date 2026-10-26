"""Finite-rollout learned head + actual soft pair policy fitting (spec §12)."""
from __future__ import annotations

import itertools
import math
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .math import head_objective, pair_loss, suffix_returns
from .sampling import stream
from .tree import TensorWorld
from .value_head import MeanHead


def run(steps: int = 200) -> dict[str,Any]:
    """Separate approximation/prediction/fitting errors in both specified regimes (§12)."""
    import importlib.util
    source = Path(__file__).resolve().parents[1]/'reference/ChunkRatio_Math_Checks.py'
    spec = importlib.util.spec_from_file_location('tree_reference_input',source)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    result = {'synthetic':True,'finite_rollouts_per_state':16,'policy_steps':steps,'regimes':{}}
    for name,a,b in [('small_displacement_large_mismatch',.08,1.),('large_displacement_small_mismatch',1.,.08)]:
        world = TensorWorld(*ref.make_world(random.Random(7),a,b))
        states = list(ref.PREFIXES)
        index = {s:i for i,s in enumerate(states)}
        features = torch.eye(len(states))
        head = MeanHead(len(states))
        optimizer = torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=0)
        datasets: dict[str,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]] = {'train':[],'head_val':[]}
        for split in datasets:
            for state in states:
                rng = stream(42,'learned-tree/'+name+'/'+split+'/'+str(state))
                for _ in range(16):
                    prefix = state
                    visited,inc = [],[]
                    while not world.terminal(prefix):
                        visited.append(index[prefix])
                        token = int(torch.multinomial(torch.tensor(world.rho[prefix]),1,generator=rng))
                        inc.append(world.d(prefix,(token,)))
                        prefix += (token,)
                    ids = torch.tensor(visited)
                    budgets = torch.tensor([(world.horizon-len(states[i]))/world.horizon for i in visited])
                    datasets[split].append((features[ids],budgets,suffix_returns(torch.tensor(inc))))
        best,best_state = float('inf'),None
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            predictions = [head(f,budget) for f,budget,_ in datasets['train']]
            loss = head_objective(predictions,[y for _,_,y in datasets['train']],world.horizon)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                validation = head_objective([head(f,budget) for f,budget,_ in datasets['head_val']],
                                            [y for _,_,y in datasets['head_val']],world.horizon).item()
            if validation < best:
                best = validation
                best_state = {k:v.detach().clone() for k,v in head.state_dict().items()}
        head.load_state_dict(best_state)
        with torch.no_grad():
            predicted = head(features,torch.tensor([(world.horizon-len(s))/world.horizon for s in states])).tolist()
        def mu(state: tuple[int,...]) -> float:
            """Terminal values are zero regardless of fitted predictor (§6.2)."""
            return 0. if world.terminal(state) else predicted[index[state]]
        reference_logps = torch.tensor([world.rho[s] for s in states],dtype=torch.float64).log()
        logits = torch.nn.Parameter(reference_logps.clone())
        policy_optimizer = torch.optim.Adam([logits],lr=.03)
        targets = torch.tensor([[world.d(s,(token,))+mu(s+(token,)) for token in world.vocab] for s in states],dtype=torch.float64)
        pairs = list(itertools.combinations(world.vocab,2))
        for _ in range(steps):
            policy_optimizer.zero_grad(set_to_none=True)
            logps = logits.log_softmax(-1)
            losses = [pair_loss(logps[:,c],logps[:,d],reference_logps[:,c],reference_logps[:,d],targets[:,c],targets[:,d],.3)[0]
                      for c,d in pairs]
            loss = torch.stack(losses).mean()
            loss.backward()
            policy_optimizer.step()
        target_distribution = ref.walk(world,ref.fixed(1),mu)
        with torch.no_grad():
            logps = logits.log_softmax(-1)
            fitted = {y:math.exp(sum(logps[index[y[:i]],token].item() for i,token in enumerate(y))) for y in world.completions(())}
            entropy = torch.stack([F.binary_cross_entropy_with_logits(.3*(targets[:,c]-targets[:,d]),
                                   (.3*(targets[:,c]-targets[:,d])).sigmoid(),reduction='none') for c,d in pairs]).mean()
            actual_loss = torch.stack([pair_loss(logps[:,c],logps[:,d],reference_logps[:,c],reference_logps[:,d],targets[:,c],targets[:,d],.3)[0]
                                      for c,d in pairs]).mean()
        exact = world.target()
        details = [{'state':list(s),'phi':world.phi(s),'mu_rho':world.stats(s)['mu_rho'],'mu_hat':mu(s)} for s in states]
        sibling_errors = []
        for state in states:
            for c,d in pairs:
                sibling_errors.append((mu(state+(c,))-mu(state+(d,)))-
                                      (world.stats(state+(c,))['mu_rho']-world.stats(state+(d,))['mu_rho']))
        result['regimes'][name] = {'head_val':best,'states':details,'max_sibling_mean_error':max(map(abs,sibling_errors)),
            'target_reverse_kl':ref.kl(target_distribution,exact),'fitted_reverse_kl':ref.kl(fitted,exact),
            'fitted_to_target_kl':ref.kl(fitted,target_distribution),'excess_pair_loss':float(actual_loss-entropy),
            'restricted_candidate_graph':{'observed_tokens':[0,1],
                'unidentified_reference_outside_mass_at_root':world.rho[()][2],
                'note':'pair odds on tokens 0/1 alone cannot identify EOS mass'},
            'disclosure':'independent head_val trajectories share enumerable state support; not an unseen-prompt generalization study'}
    return result
