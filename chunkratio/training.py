"""Offline CRPD student trainer: hierarchical batches, gradient routing and resume (§8)."""
from __future__ import annotations

import contextlib
import copy
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .artifacts import digest
from .math import diagnostics, pair_loss
from .scoring import fingerprint, score_batch
from .value_head import save_tensor


def lr_multiplier(cfg: dict[str,Any]) -> Callable[[int],float]:
    """Constant by default; opt-in linear warmup, then constant or cosine decay over cfg['steps']."""
    schedule = cfg.get('lr_schedule','constant')
    warmup = math.ceil(cfg.get('warmup_ratio',0.)*cfg['steps'])
    def factor(step: int) -> float:
        """LambdaLR multiplier for the optimizer step about to run."""
        if step < warmup:
            return (step+1)/warmup
        if schedule == 'constant':
            return 1.
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1,cfg['steps']-warmup)))
    return factor


def student_precision(cfg: dict[str,Any]) -> contextlib.AbstractContextManager:
    """Opt-in bf16 autocast for student forwards only; weights, optimizer and log-softmax stay FP32."""
    if cfg.get('policy_autocast_dtype') == 'bfloat16':
        return torch.autocast(torch.device(cfg['device']).type,dtype=torch.bfloat16)
    return contextlib.nullcontext()


class ChunkRatioTrainer:
    """Dedicated offline batches/loop selected by train.py --crpd (§3.2)."""

    def __init__(self, student: nn.Module, rho: nn.Module, targets: dict[str,Any],
                 prompts: list[dict[str,Any]], cfg: dict[str,Any], eos: int,
                 selector: Callable[[nn.Module],float] | None = None) -> None:
        """Require exact initialization and explicit policy-selection measure (§§2,11)."""
        if fingerprint(student) != fingerprint(rho):
            raise ValueError('student must be exact reference initialization')
        self.student,self.rho,self.targets,self.cfg,self.eos = student,rho,targets,cfg,eos
        self.selector = selector
        self.prompts = [p for p in prompts if p['split'] == 'train']
        self.groups: dict[str,dict[str,list[dict[str,Any]]]] = defaultdict(lambda:defaultdict(list))
        for pair in targets['pairs']:
            self.groups[pair['state']['prompt_id']][pair['state']['id']].append(pair)
        self.rng = random.Random(cfg['seed']+701)
        self.history: list[dict[str,Any]] = []

    def batch_loss(self, batch: list[dict[str,Any]]) -> tuple[torch.Tensor,dict[str,float]]:
        """Hierarchically sampled batch mean implements prescribed measure (§7.2)."""
        prefixes = [p['state']['prefix_ids'] for p in batch]
        with student_precision(self.cfg):
            c = score_batch(self.student,prefixes,[p['c']['tokens'] for p in batch],self.eos,frozen=False)
            d = score_batch(self.student,prefixes,[p['d']['tokens'] for p in batch],self.eos,frozen=False)
        def column(side: str, key: str) -> torch.Tensor:
            """Move frozen cached scalar scores to the student device (§8)."""
            return c.new_tensor([p[side][key] for p in batch])
        losses,m,u = pair_loss(c,d,column('c','logp_student_ref'),column('d','logp_student_ref'),
                             column('c','target_score'),column('d','target_score'),self.cfg['alpha'])
        diag = {k:v.mean().item() for k,v in diagnostics(losses,m,u).items()}
        return losses.mean(),diag

    def anchor_loss(self) -> torch.Tensor:
        """Only crpd_anchor adds fixed-lambda human-pair DPO (§9)."""
        rows = [p for p in self.prompts if 'chosen_ids' in p and 'rejected_ids' in p]
        if not rows:
            raise ValueError('crpd_anchor requires human preference pairs in train split')
        row = self.rng.choice(rows)
        prefix = [row['prompt_ids']]
        c,d = [row['chosen_ids']],[row['rejected_ids']]
        with student_precision(self.cfg):
            pi_delta = score_batch(self.student,prefix,c,self.eos,False)-score_batch(self.student,prefix,d,self.eos,False)
        ref_delta = score_batch(self.rho,prefix,c,self.eos)-score_batch(self.rho,prefix,d,self.eos)
        return -F.logsigmoid(self.cfg['anchor_beta']*(pi_delta-ref_delta)).mean()

    def sample(self) -> list[dict[str,Any]]:
        """Prompt uniformly, state uniformly, then unordered pair uniformly (§7.2)."""
        batch = []
        for _ in range(self.cfg['batch_size']):
            states = self.groups[self.rng.choice(list(self.groups))]
            batch.append(self.rng.choice(states[self.rng.choice(list(states))]))
        return batch

    def train(self, path: Path) -> dict[str,Any]:
        """Resume complete step transactions; no optimizer for exact null/audit (§§4.1,10)."""
        cfg = self.cfg
        if self.targets['route'] == 'no_alignment_update' or cfg['variant'] == 'crpd_audit':
            return {'optimizer_steps':0,'reason':self.targets['route'] if cfg['variant'] != 'crpd_audit' else 'diagnostic_only'}
        if not self.groups:
            raise ValueError('no eligible training sibling pairs')
        if not cfg.get('synthetic',False) and self.selector is None:
            raise ValueError('real policy selection requires pinned reward model on select split')
        optimizer = getattr(torch.optim,cfg['optimizer'])(self.student.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier(cfg))
        identity = digest({'targets':self.targets,'config':{k:v for k,v in cfg.items() if k not in ('steps','output')}})
        start,best,best_state = 0,-float('inf'),None
        if path.exists():
            saved = torch.load(path,map_location=cfg['device'],weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('incompatible policy checkpoint')
            self.student.load_state_dict(saved['student'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            self.rng.setstate(saved['sampler_state'])
            torch.set_rng_state(saved['torch_rng'].cpu())
            if torch.cuda.is_available() and saved['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            start,best,best_state = saved['step'],saved['best'],saved['best_state']
            self.history = saved['history']
        frozen_hash = fingerprint(self.rho)
        interval = cfg.get('selection_interval',1)
        for step in range(start,cfg['steps']):
            self.student.eval()  # no dropout, gradients intentionally enabled
            optimizer.zero_grad(set_to_none=True)
            batch = self.sample()
            loss_total = 0.
            diag = {'excess_bce':0.,'logit_error':0.,'saturation':0.}
            microbatch_size = cfg.get('policy_microbatch_size',1)
            assert microbatch_size > 0
            for offset in range(0,len(batch),microbatch_size):
                micro = batch[offset:offset+microbatch_size]
                loss,metrics = self.batch_loss(micro)
                weight = len(micro)/len(batch)
                (loss*weight).backward()
                loss_total += loss.item()*weight
                for key in diag:
                    diag[key] += metrics[key]*weight
            if cfg['variant'] == 'crpd_anchor':
                anchor = cfg['anchor_lambda']*self.anchor_loss()
                assert torch.isfinite(anchor)
                anchor.backward()
                loss_total += anchor.item()
            optimizer.step()
            # Streamed so a long remote run can be monitored between checkpoints.
            print(json.dumps({'variant':cfg['variant'],'step':step+1,'loss':loss_total,
                              'lr':optimizer.param_groups[0]['lr'],**diag}),flush=True)
            scheduler.step()
            # Selection and resume checkpoints share one cadence and always include the last step;
            # an interrupted run replays from the last checkpoint with its saved sampler/RNG state.
            if (step+1) % interval and step+1 != cfg['steps']:
                self.history.append({'step':step+1,'loss':loss_total,'selection_score':None,**diag})
                continue
            if self.selector is not None:
                score = float(self.selector(self.student))
            else:
                # Synthetic convergence diagnostic only; never a real checkpoint selector.
                with torch.no_grad():
                    score = -sum(self.batch_loss([p])[0].item()*p['weight'] for p in self.targets['pairs'])
            if score > best:
                best = score
                best_state = {k:v.detach().cpu().clone() for k,v in self.student.state_dict().items()}
            self.history.append({'step':step+1,'loss':loss_total,'selection_score':score,**diag})
            save_tensor(path,{'identity':identity,'student':self.student.state_dict(),
                              'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                              'step':step+1,'sampler_state':self.rng.getstate(),'torch_rng':torch.get_rng_state(),
                              'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                              'best':best,'best_state':best_state,'history':self.history})
        assert fingerprint(self.rho) == frozen_hash
        assert all(p.grad is None for p in self.rho.parameters())
        assert best_state is not None
        self.student.load_state_dict(best_state)
        return {'optimizer_steps':cfg['steps'],'selection_score':best,'history':self.history,
                'selection_rule':'synthetic_negative_pair_bce' if self.selector is None else 'select_mean_reward'}
