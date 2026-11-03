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
    """Opt-in bf16 autocast for policy forwards (student and the rho it is compared with);
    weights, optimizer and log-softmax stay FP32."""
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
        losses,metrics = self.pair_terms(batch)
        return losses.mean(),{k:v.mean().item() for k,v in metrics.items()}

    def pair_terms(self, batch: list[dict[str,Any]]) -> tuple[torch.Tensor,dict[str,torch.Tensor]]:
        """Unreduced per-pair losses and diagnostics of one batch of sibling pairs (§8)."""
        prefixes = [p['state']['prefix_ids'] for p in batch]
        chunks = {side:[p[side]['tokens'] for p in batch] for side in ('c','d')}
        with student_precision(self.cfg):
            c = score_batch(self.student,prefixes,chunks['c'],self.eos,frozen=False)
            d = score_batch(self.student,prefixes,chunks['d'],self.eos,frozen=False)
        def column(side: str, key: str) -> torch.Tensor:
            """Move frozen cached scalar scores to the student device (§8)."""
            return c.new_tensor([p[side][key] for p in batch])
        def reference(side: str) -> torch.Tensor:
            """log rho in the student's precision, so a = log pi - log rho starts at exactly zero.
            The cached scores are FP32; bf16 student forwards would differ from them at student = rho."""
            if self.cfg.get('policy_autocast_dtype') is None:
                return column(side,'logp_student_ref')
            with student_precision(self.cfg):
                return score_batch(self.rho,prefixes,chunks[side],self.eos)
        losses,m,u = pair_loss(c,d,reference('c'),reference('d'),
                             column('c','target_score'),column('d','target_score'),self.cfg['alpha'])
        return losses,diagnostics(losses,m,u)

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
        # The step count may grow on resume only when the learning-rate schedule ignores it: warmup and
        # cosine are functions of cfg['steps'], so for them the budget is part of the identity.
        payload = {'targets':self.targets,'config':{k:v for k,v in cfg.items() if k not in ('steps','output')}}
        if cfg.get('lr_schedule','constant') != 'constant' or cfg.get('warmup_ratio',0.):
            payload['schedule_steps'] = cfg['steps']
        identity = digest(payload)
        start,best,best_state = 0,-float('inf'),None
        if path.exists():
            saved = torch.load(path,map_location=cfg['device'],weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('incompatible policy checkpoint (targets, config or, with warmup/cosine, steps differ)')
            if saved['step'] > cfg['steps']:
                raise ValueError(f"policy checkpoint is at step {saved['step']}, beyond steps={cfg['steps']}")
            self.student.load_state_dict(saved['student'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            self.rng.setstate(saved['sampler_state'])
            torch.set_rng_state(saved['torch_rng'].cpu())
            if torch.cuda.is_available() and saved['cuda_rng'] is not None:
                # map_location put the saved generator states on the GPU; set_rng_state needs CPU bytes.
                torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
            start,best,best_state = saved['step'],saved['best'],saved['best_state']
            self.history = saved['history']
        frozen_hash = fingerprint(self.rho)
        interval = cfg.get('selection_interval',1)
        if cfg.get('selection_fallback_to_reference') and not cfg.get('selection_baseline'):
            raise ValueError('selection_fallback_to_reference needs selection_baseline (rho must be scored)')
        if cfg.get('selection_baseline') and start == 0 and self.selector is not None:
            # Opt-in: rho's own score on the select prompts, so checkpoints can be compared with no
            # training. Recorded only: it never competes for best_state.
            score,prompts = self.select()
            self.history.append({'step':0,'loss':None,'selection_score':score,'selection_prompt_scores':prompts,
                                 'excess_bce':None,'logit_error':None,'saturation':None})
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
            prompts = None
            if self.selector is not None:
                score,prompts = self.select()
            else:
                # Synthetic convergence diagnostic only; never a real checkpoint selector.
                with torch.no_grad():
                    score = -sum(self.batch_loss([p])[0].item()*p['weight'] for p in self.targets['pairs'])
            if score > best:
                best = score
                best_state = {k:v.detach().cpu().clone() for k,v in self.student.state_dict().items()}
            self.history.append({'step':step+1,'loss':loss_total,'selection_score':score,
                                 'selection_prompt_scores':prompts,**diag})
            save_tensor(path,{'identity':identity,'student':self.student.state_dict(),
                              'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                              'step':step+1,'sampler_state':self.rng.getstate(),'torch_rng':torch.get_rng_state(),
                              'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                              'best':best,'best_state':best_state,'history':self.history})
        assert fingerprint(self.rho) == frozen_hash
        assert all(p.grad is None for p in self.rho.parameters())
        assert best_state is not None
        self.student.load_state_dict(best_state)
        result = {'optimizer_steps':cfg['steps'],'selection_score':best,'history':self.history,
                  'selection_rule':'synthetic_negative_pair_bce' if self.selector is None else 'select_mean_reward',
                  **self.against_reference(best)}
        if cfg.get('selection_fallback_to_reference') and result['improved_over_reference'] is False:
            # Opt-in rule: no checkpoint beat rho on select, so rho itself is exported and reported as such.
            self.student.load_state_dict(self.rho.state_dict())
            result['selected'] = 'reference'
        return result

    def select(self) -> tuple[float,list[float] | None]:
        """Mean select reward of the current student and, when the selector records them, per-prompt rewards."""
        score = float(self.selector(self.student))
        prompts = getattr(self.selector,'last_scores',None)
        return score,(list(prompts) if prompts is not None else None)

    def against_reference(self, best: float) -> dict[str,Any]:
        """Best checkpoint against the recorded rho baseline: whether it improved and, with per-prompt
        rewards, the paired mean difference with a normal 95% interval and win/loss counts."""
        baseline = next((h for h in self.history if h['step'] == 0 and h['selection_score'] is not None),None)
        if baseline is None:
            return {'improved_over_reference':None,'selected':'checkpoint'}
        report = {'improved_over_reference':best > baseline['selection_score'],'selected':'checkpoint',
                  'reference_selection_score':baseline['selection_score']}
        chosen = next(h for h in self.history if h['step'] > 0 and h['selection_score'] == best)
        if baseline.get('selection_prompt_scores') and chosen.get('selection_prompt_scores'):
            diffs = [a-b for a,b in zip(chosen['selection_prompt_scores'],baseline['selection_prompt_scores'])]
            mean = sum(diffs)/len(diffs)
            sd = (sum((d-mean)**2 for d in diffs)/(len(diffs)-1))**.5 if len(diffs) > 1 else 0.
            half = 1.96*sd/len(diffs)**.5
            report['paired_vs_reference'] = {'step':chosen['step'],'mean_difference':mean,'low':mean-half,'high':mean+half,
                                             'prompts':len(diffs),'wins':sum(d > 0 for d in diffs),
                                             'losses':sum(d < 0 for d in diffs)}
        return report
