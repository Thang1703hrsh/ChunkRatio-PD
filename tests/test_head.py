"""Frozen features and head-only training verification (§12)."""
from pathlib import Path

import pytest

torch = pytest.importorskip('torch',reason='head fitting requires PyTorch')
from chunkratio.scoring import fingerprint, freeze, state_features
from chunkratio.tiny import TinyLM
from chunkratio.value_head import MeanHead, fit_head, trajectory_loss


def test_head_gradient_and_resume(tmp_path: Path) -> None:
    """Only scalar head changes, validation chooses checkpoint and resume is deterministic (§7.4)."""
    torch.manual_seed(4)
    rho = freeze(TinyLM())
    before = fingerprint(rho)
    features = state_features(rho,[[1],[1,2]]).tolist()
    rows = [{'seed':{'split':split},'features':features,'budgets':[1.,.5],'labels':[.5,.2]} for split in ('train','head_val')]
    cfg = {'horizon':4,'seed':3,'head_learning_rate':.01,'device':'cpu','head_epochs':3}
    head,meta = fit_head(rows,cfg,tmp_path/'head.pt')
    assert fingerprint(rho) == before and all(p.grad is None for p in rho.parameters())
    assert not any(p.requires_grad for p in head.parameters())
    restored,other = fit_head(rows,cfg,tmp_path/'head.pt')
    assert meta['hash'] == other['hash'] == fingerprint(restored)
    assert len(meta['history']) == 3
    fit_head(rows,{**cfg,'head_epochs':1},tmp_path/'partial.pt')
    extended,extended_meta = fit_head(rows,cfg,tmp_path/'partial.pt')
    assert extended_meta['hash'] == meta['hash']
    assert extended_meta['history'] == meta['history']
