"""Sampling, collection and exact-case training controls (§12)."""
import copy
from pathlib import Path

import pytest

torch = pytest.importorskip('torch',reason='pipeline runtime requires PyTorch')
from chunkratio.artifacts import ShardStore
from chunkratio.collection import (collect_candidates, collect_correction, collect_prefixes, correction_features,
                                  prepare_prompts)
from chunkratio.config import load_config
from chunkratio.models import load_models
from chunkratio.sampling import generate, stream, terminal
from chunkratio.scoring import fingerprint, preflight, score_batch
from chunkratio.targets import build_targets
from chunkratio.training import ChunkRatioTrainer


def test_global_horizon_raw_sampling() -> None:
    """Chunk boundaries retain budget; raw categorical frequencies match probabilities (§5)."""
    from types import SimpleNamespace
    class Constant(torch.nn.Module):
        """Controlled categorical model (§12)."""
        def __init__(self) -> None:
            """Declare independent token probabilities (§5.3)."""
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([.2,.3,.5]).log())
        def forward(self, ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            """Broadcast raw logits over positions (§5.3)."""
            return SimpleNamespace(logits=self.logits.expand(*ids.shape,3))
    model = Constant()
    rng = stream(3,'raw')
    draws = [generate(model,[1],1,2,rng)[0] for _ in range(3000)]
    observed = torch.bincount(torch.tensor(draws),minlength=3)/len(draws)
    torch.testing.assert_close(observed,torch.tensor([.2,.3,.5]),atol=.035,rtol=0)
    assert len(generate(model,[1,0,1],2,2,rng,32)) <= 2
    assert not terminal([1,1],6,2)
    assert terminal([1,2],6,2)
    assert terminal([1]*6,6,2)


def test_duplicate_split_preflight() -> None:
    """Token mapping mismatch and duplicate cross-split leakage fail before generation (§§5,7)."""
    with pytest.raises(ValueError,match='crosses splits'):
        prepare_prompts([{'prompt_ids':[1],'split':'train'},{'prompt_ids':[1],'split':'test'}],0)
    cfg = load_config('config/crpd/smoke.json')
    models,student,toks,_ = load_models(cfg)
    toks[-1].chat_template = 'different'
    with pytest.raises(ValueError,match='mismatch'):
        preflight(toks,[models['teacher_plus'],models['teacher_base'],models['student_ref'],student])


def test_collection_resume_and_gradient_routing(tmp_path: Path) -> None:
    """Whole pipeline fixtures verify caching, both sources and only student gradients (§12)."""
    from chunkratio.artifacts import read_jsonl
    from chunkratio.value_head import fit_head
    cfg = load_config('config/crpd/smoke.json')
    models,student,_,info = load_models(cfg)
    prompts = prepare_prompts(read_jsonl(Path(cfg['prompts'])),cfg['seed'])
    a,b,c = [ShardStore(tmp_path/name,{'test':name}) for name in ('prefix','candidate','correction')]
    states = collect_prefixes(prompts,models['student_ref'],cfg,info['eos'],a)
    records = collect_candidates(states,models,cfg,info['eos'],b)
    rows = collect_correction(records,models,cfg,info['eos'],c)
    assert rows == collect_correction(records,models,cfg,info['eos'],c)
    head,_ = fit_head(rows,cfg,tmp_path/'head.pt',correction_features(c))
    targets = build_targets(records,head,models,cfg,info['eos'],'reference_mean')
    assert sum(p['weight'] for p in targets['pairs']) == pytest.approx(1.)
    frozen_before = {k:fingerprint(v) for k,v in models.items()}
    student_before = fingerprint(student)
    trainer = ChunkRatioTrainer(student,models['student_ref'],targets,prompts,cfg,info['eos'])
    result = trainer.train(tmp_path/'policy.pt')
    assert result['optimizer_steps'] == cfg['steps']
    assert fingerprint(student) != student_before
    assert {k:fingerprint(v) for k,v in models.items()} == frozen_before
    assert all(p.grad is None for model in models.values() for p in model.parameters())
    assert all(p.grad is None for p in head.parameters())
    fresh = copy.deepcopy(models['student_ref']).requires_grad_(True)
    resumed = ChunkRatioTrainer(fresh,models['student_ref'],targets,prompts,cfg,info['eos'])
    restored = resumed.train(tmp_path/'policy.pt')
    assert restored['history'] == result['history']
    assert fingerprint(fresh) == fingerprint(student)
    # Resume from an actual partial run, not only an already-completed checkpoint.
    partial_student = copy.deepcopy(models['student_ref']).requires_grad_(True)
    short_cfg = {**cfg,'steps':1}
    partial = ChunkRatioTrainer(partial_student,models['student_ref'],targets,prompts,short_cfg,info['eos'])
    partial.train(tmp_path/'partial.pt')
    continued_student = copy.deepcopy(models['student_ref']).requires_grad_(True)
    continued = ChunkRatioTrainer(continued_student,models['student_ref'],targets,prompts,cfg,info['eos'])
    continued_result = continued.train(tmp_path/'partial.pt')
    assert continued_result['history'] == result['history']
    assert fingerprint(continued_student) == fingerprint(student)


def test_no_alignment_update_and_exact_zero(tmp_path: Path) -> None:
    """No alignment never steps; shared reference hard-bypasses endpoint head (§4.1)."""
    cfg = load_config('config/crpd/smoke.json')
    models,student,_,info = load_models(cfg)
    before = fingerprint(student)
    trainer = ChunkRatioTrainer(student,models['student_ref'],{'pairs':[],'route':'no_alignment_update'},[],cfg,info['eos'])
    assert trainer.train(tmp_path/'none.pt')['optimizer_steps'] == 0
    assert fingerprint(student) == before and not (tmp_path/'none.pt').exists()
    state = {'id':'state','prefix_ids':[1],'prompt_id':'p','split':'train','n':0}
    candidates = [{'id':str(i),'tokens':[i+2],'terminal':False,
                   'logp_teacher_plus':-.3,'logp_teacher_base':-.4,'logp_student_ref':-.5} for i in range(2)]
    result = build_targets([{'state':state,'candidates':candidates,'skipped':False}],None,models,cfg,info['eos'],'exact_zero_residual')
    assert all(c['endpoint_mu'] == 0 for c in result['records'][0]['candidates'])


def test_selection_and_checkpoint_cadence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RM selection and resume checkpoints run every selection_interval steps and at the end (§§10,11.1)."""
    import chunkratio.training as training
    cfg = {**load_config('config/crpd/smoke.json'),'steps':5,'selection_interval':2}
    models,student,_,info = load_models(cfg)
    state = {'id':'state','prefix_ids':[1],'prompt_id':'p','split':'train','n':0}
    candidates = [{'id':str(i),'tokens':[i+2],'terminal':False,'logp_teacher_plus':-.3*i,
                   'logp_teacher_base':-.4,'logp_student_ref':-.5} for i in range(2)]
    targets = build_targets([{'state':state,'candidates':candidates,'skipped':False}],None,models,cfg,
                            info['eos'],'exact_zero_residual')
    for pair in targets['pairs']:
        for side in ('c','d'):
            pair[side]['logp_student_ref'] = score_batch(models['student_ref'],[[1]],[pair[side]['tokens']],info['eos']).item()
    calls, saves = [],[]
    original = training.save_tensor
    monkeypatch.setattr(training,'save_tensor',lambda path,payload:(saves.append(payload['step']),original(path,payload)))
    trainer = ChunkRatioTrainer(student,models['student_ref'],targets,[],cfg,info['eos'],
                                selector=lambda model:calls.append(1) or -len(calls))
    result = trainer.train(tmp_path/'policy.pt')
    assert len(calls) == 3 and saves == [2,4,5]
    assert [h['selection_score'] is not None for h in result['history']] == [False,True,False,True,True]
    assert result['optimizer_steps'] == 5
    from chunkratio.config import validate
    with pytest.raises(ValueError,match='selection_interval'):
        validate({**cfg,'selection_interval':0})
    unresolved = {k:v for k,v in cfg.items() if k != 'selection_interval'}
    with pytest.raises(ValueError,match='explicit selection_interval'):
        validate({**unresolved,'synthetic':False})
