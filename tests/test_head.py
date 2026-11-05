"""Frozen features and head-only training verification (§12)."""
from pathlib import Path

import pytest

torch = pytest.importorskip('torch',reason='head fitting requires PyTorch')
from chunkratio.artifacts import ShardStore
from chunkratio.collection import correction_features
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


def test_boundary_head_positions_keep_only_chunk_grid_states(tmp_path: Path) -> None:
    """head_positions='boundary' scores rollout position t only when (n + t) % L == 0, enters the head
    identity, and leaves the default all-position objective unchanged (compact configuration)."""
    from chunkratio.math import head_objective
    torch.manual_seed(5)
    head = MeanHead(3)
    features = torch.randn(10,3)
    budgets = [1-t/16 for t in range(10)]
    labels = [float(t) for t in range(10)]
    row = {'seed':{'split':'train','n':3},'features':features.tolist(),'budgets':budgets,'labels':labels}
    keep = [1,5,9]
    expected = head_objective([head(features[keep],torch.tensor(budgets)[keep])],[torch.tensor(labels)[keep]],16)
    torch.testing.assert_close(trajectory_loss(head,[row],16,boundary=4),expected)
    everything = head_objective([head(features,torch.tensor(budgets))],[torch.tensor(labels)],16)
    torch.testing.assert_close(trajectory_loss(head,[row],16),everything)
    rows = [row,{**row,'seed':{'split':'head_val','n':3}}]
    cfg = {'horizon':16,'seed':3,'head_learning_rate':.01,'device':'cpu','head_epochs':2,'chunk_length':4}
    _,default = fit_head(rows,cfg,tmp_path/'all.pt')
    _,boundary = fit_head(rows,{**cfg,'head_positions':'boundary'},tmp_path/'boundary.pt')
    assert default['identity'] != boundary['identity'] and boundary['positions'] == 'chunk boundaries (L=4)'


def test_feature_shards_and_minibatch_head(tmp_path: Path) -> None:
    """Binary feature shards verify content; minibatches take ceil(N/B) steps and resume exactly (§§7.4,10)."""
    torch.manual_seed(6)
    rho = freeze(TinyLM())
    store = ShardStore(tmp_path/'shards',{'test':'features'})
    rows = []
    for i,split in enumerate(['train']*3+['head_val']):
        job = f'trajectory-{i}'
        features = state_features(rho,[[1],[1,2+i]])
        rows.append({'id':job,'seed':{'split':split},'budgets':[1.,.5],'labels':[.5-i/10,.2],
                     'features_sha256':store.put_tensor(job,'features',features)})
        torch.testing.assert_close(correction_features(store)(rows[-1]),features,atol=0,rtol=0)
    cfg = {'horizon':4,'seed':3,'head_learning_rate':.01,'device':'cpu','head_epochs':3,'head_batch_size':2}
    with pytest.raises(ValueError,match='loader'):
        fit_head(rows,cfg,tmp_path/'missing.pt')
    load = correction_features(store)
    _,meta = fit_head(rows,cfg,tmp_path/'head.pt',load)
    saved = torch.load(tmp_path/'head.resume.pt',weights_only=True)
    assert saved['optimizer']['state'][0]['step'].item() == 3*2
    assert meta['batch_trajectories'] == 2
    fit_head(rows,{**cfg,'head_epochs':1},tmp_path/'partial.pt',load)
    _,extended = fit_head(rows,cfg,tmp_path/'partial.pt',load)
    assert extended['hash'] == meta['hash'] and extended['history'] == meta['history']
    _,full_batch = fit_head(rows,{k:v for k,v in cfg.items() if k != 'head_batch_size'},tmp_path/'full.pt',load)
    assert full_batch['batch_trajectories'] == 3 and full_batch['hash'] != meta['hash']
    corrupt = state_features(rho,[[1],[1,9]])
    store.put_tensor('trajectory-0','features',corrupt)
    with pytest.raises(ValueError,match='corrupt tensor shard'):
        load(rows[0])
