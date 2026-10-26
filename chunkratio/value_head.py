"""One scalar detached-reference mean head with fixed-H regression (§§6.2,7.4)."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .artifacts import atomic_bytes, digest
from .math import head_objective
from .scoring import fingerprint, freeze


def save_tensor(path: Path, payload: Any) -> None:
    """Atomically save model/optimizer/tensor shards (§10)."""
    buffer = io.BytesIO()
    torch.save(payload,buffer)
    atomic_bytes(path,buffer.getvalue())


class MeanHead(nn.Module):
    """Linear(d+1,256)->GELU->Linear(256,1), no dropout (§6.2)."""

    def __init__(self, hidden_size: int) -> None:
        """Add one remaining-response-budget fraction feature (§6.2)."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size+1,256),nn.GELU(),nn.Linear(256,1))

    def forward(self, features: Tensor, budgets: Tensor) -> Tensor:
        """Detached rho features; ordinary tensors permit head gradients (§6.2)."""
        assert features.ndim == 2 and budgets.shape == (len(features),)
        assert ((budgets >= 0) & (budgets <= 1)).all()
        return self.net(torch.cat([features.detach().float(),budgets[:,None].float()],dim=-1)).squeeze(-1)


def trajectory_loss(head: MeanHead, rows: list[dict[str,Any]], horizon: int) -> Tensor:
    """Trajectory-uniform SSE/H including all nonterminal visited states (§7.4)."""
    device = next(head.parameters()).device
    predictions, labels = [],[]
    for row in rows:
        features = torch.tensor(row['features'],dtype=torch.float32,device=device)
        budgets = torch.tensor(row['budgets'],device=device)
        predictions.append(head(features,budgets))
        labels.append(torch.tensor(row['labels'],dtype=torch.float32,device=device))
    return head_objective(predictions,labels,horizon)


def fit_head(rows: list[dict[str,Any]], cfg: dict[str,Any], path: Path) -> tuple[MeanHead,dict[str,Any]]:
    """Fit, head_val-select, resume epoch/optimizer/RNG and freeze (§§7.4,10)."""
    train = [r for r in rows if r['seed']['split'] == 'train']
    validation = [r for r in rows if r['seed']['split'] == 'head_val']
    if not train or not validation:
        raise ValueError('head fitting requires disjoint train and head_val prompts')
    identity = digest({'rows':rows,'horizon':cfg['horizon'],'lr':cfg['head_learning_rate'],
                       'seed':cfg['seed'],'architecture':'Linear(d+1,256)-GELU-Linear(256,1)'})
    torch.manual_seed(cfg['seed']+503)
    head = MeanHead(len(train[0]['features'][0])).to(cfg['device'])
    optimizer = torch.optim.AdamW(head.parameters(),lr=cfg['head_learning_rate'],weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    epoch, best, best_state, history = 0,float('inf'),None,[]
    checkpoint = path.with_suffix('.resume.pt')
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location=cfg['device'],weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('incompatible head checkpoint')
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng'].cpu())
        epoch,best,best_state,history = saved['epoch'],saved['best'],saved['best_state'],saved['history']
    for i in range(epoch,cfg['head_epochs']):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        # Accumulate one trajectory at a time; normalize by total trajectories * H.
        total = 0.
        for row in train:
            loss = trajectory_loss(head,[row],cfg['horizon'])/len(train)
            assert torch.isfinite(loss)
            loss.backward()
            total += loss.item()
        optimizer.step()
        scheduler.step()
        head.eval()
        with torch.no_grad():
            val = trajectory_loss(head,validation,cfg['horizon']).item()
        if val < best:
            best = val
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append({'epoch':i+1,'train':total,'head_val':val})
        save_tensor(checkpoint,{'identity':identity,'head':head.state_dict(),'optimizer':optimizer.state_dict(),
                                'scheduler':scheduler.state_dict(),'epoch':i+1,'best':best,
                                'best_state':best_state,'history':history,'rng':torch.get_rng_state()})
    assert best_state is not None
    head.load_state_dict(best_state)
    freeze(head)
    metadata = {'identity':identity,'hash':fingerprint(head),'head_val':best,'history':history,
                'feature_layer':'final','feature_dtype':'float32','budget_encoding':'(H-n)/H',
                'hidden_size':len(train[0]['features'][0]),'train_prompt_reuse':True}
    save_tensor(path,{'metadata':metadata,'state_dict':head.state_dict()})
    return head,metadata
