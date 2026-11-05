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
    # Prompt X carries groups A and B, so B's other prompt Y is X's duplicate group too, in either order.
    linked = [{'prompt_ids':[1,2],'group_id':'A','split':'train'},{'prompt_ids':[1,2],'group_id':'B','split':'train'},
              {'prompt_ids':[3],'group_id':'B','split':'test'}]
    for rows in (linked,linked[::-1]):
        with pytest.raises(ValueError,match='crosses splits'):
            prepare_prompts(rows,0)
    kept = prepare_prompts([{**row,'split':'train'} for row in linked],0)
    assert sorted(r['prompt_ids'] for r in kept) == [[1,2],[3]]
    # Unsplit rows of one duplicate group share one hashed split.
    unsplit = prepare_prompts([{'prompt_ids':[4],'group_id':'C'},{'prompt_ids':[5],'group_id':'C'}],0)
    assert len({r['split'] for r in unsplit}) == 1
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
    # A checkpoint past the requested budget is never reported as that budget.
    with pytest.raises(ValueError,match='beyond steps=1'):
        ChunkRatioTrainer(copy.deepcopy(models['student_ref']).requires_grad_(True),models['student_ref'],targets,
                          prompts,short_cfg,info['eos']).train(tmp_path/'policy.pt')
    # Warmup/cosine depend on the step budget, so a cosine run cannot be extended on resume.
    cosine = {**cfg,'lr_schedule':'cosine','warmup_ratio':.5}
    ChunkRatioTrainer(copy.deepcopy(models['student_ref']).requires_grad_(True),models['student_ref'],targets,prompts,
                      {**cosine,'steps':1},info['eos']).train(tmp_path/'cosine.pt')
    with pytest.raises(ValueError,match='incompatible policy checkpoint'):
        ChunkRatioTrainer(copy.deepcopy(models['student_ref']).requires_grad_(True),models['student_ref'],targets,
                          prompts,cosine,info['eos']).train(tmp_path/'cosine.pt')


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


def test_sampler_choice_is_part_of_every_collection_key(tmp_path: Path) -> None:
    """Switching gpu_sampling on a shared cache must collect afresh, never reuse the other sampler's
    rollouts; the default sampler keeps its existing keys, and batch grouping does not change keys."""
    from chunkratio.pipeline import pipeline
    cfg = {**load_config('config/crpd/smoke.json'),'output':str(tmp_path/'run')}
    pipeline(cfg,'collect-prefixes')
    pipeline({**cfg,'generation_batch_size':1},'collect-prefixes')
    caches = lambda: sorted(p.name for p in (tmp_path/'run'/'cache').iterdir() if p.name.startswith('prefixes-'))
    default = caches()
    assert len(default) == 1
    pipeline({**cfg,'gpu_sampling':True},'collect-prefixes')
    both = caches()
    assert len(both) == 2 and default[0] in both
    from chunkratio.artifacts import read_json
    responses = {name:sorted(tuple(read_json(p)['payload']['response'])
                             for p in (tmp_path/'run'/'cache'/name).glob('*.jsonl')) for name in both}
    assert responses[both[0]] != responses[both[1]]


@pytest.mark.parametrize('autocast',[None,'bfloat16'])
def test_shared_prefix_scoring_matches_default_end_to_end(tmp_path: Path, autocast: object) -> None:
    """shared_prefix_scoring (candidate scores, student and rho pair scores) reproduces the default run:
    candidate log-probs, every step's loss and the pair diagnostics agree up to rounding."""
    from chunkratio.artifacts import read_json
    from chunkratio.pipeline import pipeline
    base = {**load_config('config/crpd/smoke.json'),'variant':'crpd_raw','steps':3,'policy_autocast_dtype':autocast}
    runs = {}
    for name,extra in (('default',{}),('shared',{'shared_prefix_scoring':True,'score_tokens_per_batch':64})):
        result = pipeline({**base,**extra,'output':str(tmp_path/name)},'evaluate')
        records = {r['state']['id']:r for r in read_json(Path(read_json(tmp_path/name/'target_index.json')['path']))['records']}
        runs[name] = (result,records)
    (a,ra),(b,rb) = runs['default'],runs['shared']
    assert ra.keys() == rb.keys()
    for state_id,record in ra.items():
        for c,d in zip(record['candidates'],rb[state_id]['candidates']):
            assert c['tokens'] == d['tokens']
            for key in ('logp_student_ref','logp_teacher_base','logp_teacher_plus'):
                assert c[key] == pytest.approx(d[key],abs=1e-4)
    losses = lambda r: [h['loss'] for h in r['policy']['history'] if h['loss'] is not None]
    assert losses(a) == pytest.approx(losses(b),abs=2e-4)
    for key,value in a['pair_diagnostics'].items():
        assert b['pair_diagnostics'][key] == pytest.approx(value,abs=1e-3)


def test_unscored_end_token_has_its_own_candidate_cache(tmp_path: Path) -> None:
    """score_end_token=False rescored candidates under a separate key: same tokens, and every chunk that
    ends with EOS loses exactly that token's score; chunks without EOS keep theirs."""
    from chunkratio.artifacts import read_json
    from chunkratio.pipeline import pipeline
    cfg = {**load_config('config/crpd/smoke.json'),'variant':'crpd_raw','output':str(tmp_path/'run')}
    pipeline(cfg,'collect-candidates')
    pipeline({**cfg,'score_end_token':False},'collect-candidates')
    stores = sorted(p for p in (tmp_path/'run'/'cache').iterdir() if p.name.startswith('candidates-'))
    assert len(stores) == 2
    records = [{read_json(p)['payload']['state']['id']:read_json(p)['payload'] for p in store.glob('*.jsonl')} for store in stores]
    assert records[0].keys() == records[1].keys()
    eos = load_models(cfg)[3]['eos']
    changed = 0
    for state_id,a in records[0].items():
        for c,d in zip(a['candidates'],records[1][state_id]['candidates']):
            assert c['tokens'] == d['tokens']
            if eos in c['tokens']:
                changed += any(abs(c[k]-d[k]) > 1e-6 for k in ('logp_teacher_plus','logp_teacher_base','logp_student_ref'))
            else:
                assert all(abs(c[k]-d[k]) < 1e-5 for k in ('logp_teacher_plus','logp_teacher_base','logp_student_ref'))
    assert changed > 0


def test_gamma_sweep_shares_collection_and_keys_targets(tmp_path: Path) -> None:
    """gamma runs on one cache reuse the gamma = 1 collection (one prefix and one candidate store); an
    explicit gamma = 1 keeps the original target store, gamma < 1 gets its own, and its targets are
    gamma times the gamma = 1 targets with unchanged pairs and weights (DECISIONS)."""
    from chunkratio.artifacts import read_json
    from chunkratio.pipeline import pipeline
    cfg = {**load_config('config/crpd/smoke.json'),'variant':'crpd_raw','output':str(tmp_path/'run')}
    index = lambda: read_json(tmp_path/'run'/'target_index.json')['path']
    pipeline(cfg,'build-targets')
    base_path = index()
    base = read_json(Path(base_path))
    pipeline({**cfg,'teacher_gap_scale':1.},'build-targets')
    assert index() == base_path
    pipeline({**cfg,'teacher_gap_scale':.3},'build-targets')
    assert index() != base_path
    scaled = read_json(Path(index()))
    stores = lambda stage: [p for p in (tmp_path/'run'/'cache').iterdir() if p.name.startswith(stage+'-')]
    assert len(stores('prefixes')) == len(stores('candidates')) == 1 and len(stores('targets')) == 2
    for a,b in zip(base['records'],scaled['records']):
        for c,d in zip(a['candidates'],b['candidates']):
            assert c['tokens'] == d['tokens'] and d['target_score'] == pytest.approx(.3*c['target_score'],abs=1e-12)
    assert [(p['id'],p['weight']) for p in base['pairs']] == [(p['id'],p['weight']) for p in scaled['pairs']]


def test_candidate_coverage_scores_end_token_for_student_and_rho(tmp_path: Path) -> None:
    """Coverage measures actual chunk probabilities: with score_end_token=false rho is rescored with the
    end token like the student, so its mass equals the default run's rho mass on the same tokens."""
    from chunkratio.pipeline import pipeline
    base = {**load_config('config/crpd/smoke.json'),'variant':'crpd_raw','steps':1}
    a = pipeline({**base,'output':str(tmp_path/'scored')},'evaluate')
    b = pipeline({**base,'score_end_token':False,'output':str(tmp_path/'unscored')},'evaluate')
    ra = {r['state_id']:r for r in a['candidate_coverage']}
    assert ra.keys() == {r['state_id'] for r in b['candidate_coverage']}
    for row in b['candidate_coverage']:
        assert row['end_token_scored']
        assert row['reference_candidate_mass'] == pytest.approx(ra[row['state_id']]['reference_candidate_mass'],rel=1e-5)


def test_overlong_prompts_dropped_before_generation(tmp_path: Path) -> None:
    """Prompt + H beyond any model context is excluded by rule and recorded, not fatal (§5.2)."""
    import json
    from chunkratio.artifacts import read_json
    from chunkratio.pipeline import pipeline
    prompts = tmp_path/'prompts.jsonl'
    rows = [{'prompt_ids':[1,2],'split':'train'},{'prompt_ids':[1]*2043,'split':'train'},
            {'prompt_ids':[2]*2042,'split':'train'}]
    prompts.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    cfg = {**load_config('config/crpd/smoke.json'),'prompts':str(prompts),'output':str(tmp_path/'run')}
    states = pipeline(cfg,'collect-prefixes')['states']
    manifest = read_json(tmp_path/'run'/'manifest.json')
    dropped = manifest['ineligible_prompts']['ids']
    assert dropped == [prepare_prompts([rows[1]],0)[0]['id']]
    assert states > 0
    with pytest.raises(ValueError,match='no prompt fits'):
        pipeline({**cfg,'horizon':4096,'output':str(tmp_path/'none')},'validate-config')
    capped = pipeline({**cfg,'max_prompt_tokens':100,'output':str(tmp_path/'capped')},'validate-config')['ineligible_prompts']
    assert capped['max_prompt_tokens'] == 100 and set(capped['ids']) == {r['id'] for r in prepare_prompts(rows[1:],0)}


def test_code_revision_follows_uncommitted_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing any package source changes the revision; line endings alone do not (§10)."""
    from chunkratio.pipeline import code_revision, source_digest
    package = tmp_path/'pkg'
    (package/'sub').mkdir(parents=True)
    (package/'a.py').write_bytes(b'x = 1\n')
    (package/'sub'/'b.py').write_bytes(b'y = 2\n')
    before = source_digest(package)
    (package/'a.py').write_bytes(b'x = 1\r\n')
    assert source_digest(package) == before
    (package/'sub'/'b.py').write_bytes(b'y = 3\n')
    assert source_digest(package) != before
    monkeypatch.setenv('CRPD_CODE_REVISION','abc-dirty')
    assert code_revision() == f'abc-dirty+src.{source_digest()[:16]}'


def test_train_only_collection_keeps_raw_pairs_and_student(tmp_path: Path) -> None:
    """crpd_raw with collection_splits ['train'] skips the head_val prompts, yet gives the same train
    states, candidates, pairs, weights, targets and trained student as the default collection (DECISIONS 87)."""
    import torch
    from chunkratio.artifacts import read_json
    from chunkratio.pipeline import pipeline
    base = {**load_config('config/crpd/smoke.json'),'variant':'crpd_raw','steps':2}
    a = pipeline({**base,'output':str(tmp_path/'all')},'evaluate')
    b = pipeline({**base,'collection_splits':['train'],'output':str(tmp_path/'train')},'evaluate')
    ta = read_json(Path(read_json(tmp_path/'all'/'target_index.json')['path']))
    tb = read_json(Path(read_json(tmp_path/'train'/'target_index.json')['path']))
    assert {r['state']['split'] for r in ta['records']} == {'train','head_val'}
    assert {r['state']['split'] for r in tb['records']} == {'train'}
    assert tb['pairs'] == ta['pairs']
    assert b['counts']['pairs'] == a['counts']['pairs'] and b['counts']['states'] < a['counts']['states']
    sa = torch.load(tmp_path/'all'/'student.pt',weights_only=True)['state_dict']
    sb = torch.load(tmp_path/'train'/'student.pt',weights_only=True)['state_dict']
    assert all(torch.equal(sa[k],sb[k]) for k in sa)


def test_shared_store_gives_only_configured_splits(tmp_path: Path) -> None:
    """A train-only run on a cache that already holds head_val shards returns the train states only, so its
    pairs and counts do not depend on which runs filled the cache."""
    from chunkratio.pipeline import pipeline
    base = {**load_config('config/crpd/smoke.json'),'variant':'crpd_raw','output':str(tmp_path/'run')}
    full = pipeline(base,'collect-prefixes')['states']
    fresh = pipeline({**base,'collection_splits':['train'],'output':str(tmp_path/'fresh')},'collect-prefixes')['states']
    shared = pipeline({**base,'collection_splits':['train']},'collect-prefixes')['states']
    assert shared == fresh < full
    # Candidates too: the shared store holds head_val shards after a full collection.
    pipeline(base,'collect-candidates')
    fresh_c = pipeline({**base,'collection_splits':['train'],'output':str(tmp_path/'fresh')},'collect-candidates')
    shared_c = pipeline({**base,'collection_splits':['train']},'collect-candidates')
    assert shared_c['states'] == fresh_c['states'] < pipeline(base,'collect-candidates')['states']
    # Full evaluate on the shared cache equals the fresh train-only run, cost counts included.
    a = pipeline({**base,'steps':1,'collection_splits':['train'],'output':str(tmp_path/'fresh')},'evaluate')
    b = pipeline({**base,'steps':1,'collection_splits':['train']},'evaluate')
    assert a['counts'] == b['counts']
    assert a['cost']['collection_generated_tokens'] == b['cost']['collection_generated_tokens']
    assert len(a['candidate_coverage']) == len(b['candidate_coverage'])


def test_store_records_detect_corruption(tmp_path: Path) -> None:
    """records() reads each shard once and still rejects a tampered payload or a renamed shard."""
    import json
    store = ShardStore(tmp_path/'s',{'stage':'t'})
    store.put('a',{'x':1})
    store.put('b',{'x':2})
    assert sorted(r['x'] for r in store.records()) == [1,2]
    path = next((tmp_path/'s').glob('*.jsonl'))
    record = json.loads(path.read_text())
    record['payload']['x'] = 99
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='corrupt shard'):
        store.records()


def test_collection_splits_validation() -> None:
    """Train-only collection is for head-free variants; malformed split lists fail closed."""
    from chunkratio.config import validate
    base = load_config('config/crpd/smoke.json')
    validate({**base,'variant':'crpd_raw','collection_splits':['train']})
    for bad in (['head_val'],['train','select'],'train',['train','train']):
        with pytest.raises(ValueError):
            validate({**base,'variant':'crpd_raw','collection_splits':bad})
    with pytest.raises(ValueError,match='crpd_raw and crpd_suffix only'):
        validate({**base,'variant':'crpd_main','collection_splits':['train']})


def test_cache_keys_ignore_commit_but_follow_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Collection keys carry the source/library identity, not the git commit: a new commit reuses the
    collection, while the manifest still records the full code revision (DECISIONS 87)."""
    from chunkratio.artifacts import cache_revision, read_json, source_digest
    from chunkratio.pipeline import pipeline
    cfg = {**load_config('config/crpd/smoke.json'),'variant':'crpd_raw','output':str(tmp_path/'run')}
    assert cache_revision().startswith(f'src.{source_digest()[:16]}+py.')
    stores = lambda: sorted(p.name for p in (tmp_path/'run'/'cache').iterdir())
    monkeypatch.setenv('CRPD_CODE_REVISION','aaa')
    pipeline(cfg,'collect-candidates')
    first = stores()
    monkeypatch.setenv('CRPD_CODE_REVISION','bbb')
    pipeline(cfg,'collect-candidates')
    assert stores() == first
    assert read_json(tmp_path/'run'/'manifest.json')['code_revision'].startswith('bbb+src.')


@pytest.mark.parametrize('options',[{},{'shared_prefix_scoring':True},{'score_end_token':False},
                                    {'policy_tokens_per_microbatch':64}])
def test_pair_diagnostics_score_each_candidate_once(tmp_path: Path, options: dict) -> None:
    """Post-training diagnostics from one score per candidate equal the former per-pair pass (pair_terms
    over every pair) for the default rule, shared-prefix scoring, the unscored end token and a token budget,
    while scoring fewer chunks than two per pair (DECISIONS 88)."""
    import torch
    from chunkratio.artifacts import read_jsonl
    from chunkratio.pipeline import pair_diagnostics
    cfg = {**load_config('config/crpd/smoke.json'),'variant':'crpd_raw',**options}
    models,student,_,info = load_models(cfg)
    prompts = prepare_prompts(read_jsonl(Path(cfg['prompts'])),cfg['seed'])
    a,b = [ShardStore(tmp_path/name,{'test':name}) for name in ('prefix','candidate')]
    states = collect_prefixes(prompts,models['student_ref'],cfg,info['eos'],a)
    records = collect_candidates(states,models,cfg,info['eos'],b)
    targets = build_targets(records,None,models,cfg,info['eos'],'reference_mean')
    trainer = ChunkRatioTrainer(student,models['student_ref'],targets,prompts,{**cfg,'steps':2},info['eos'])
    trainer.train(tmp_path/'policy.pt')
    pairs = targets['pairs']
    expected = {'excess_bce':0.,'logit_error':0.,'saturation':0.}
    with torch.no_grad():
        _,metrics = trainer.pair_terms(pairs)
        weights = torch.tensor([p['weight'] for p in pairs],dtype=torch.float64)
        for key in expected:
            expected[key] = float((metrics[key].double().cpu()*weights).sum())
    calls = []
    original = trainer.candidate_logps
    trainer.candidate_logps = lambda chunk: calls.append(sum(len(c) for _,c in chunk)) or original(chunk)
    got = pair_diagnostics(trainer,pairs,cfg)
    assert expected['logit_error'] > 0
    for key in expected:
        assert got[key] == pytest.approx(expected[key],rel=1e-5,abs=1e-7)
    unique = len({(p['state']['id'],p[s]['id']) for p in pairs for s in ('c','d')})
    assert sum(calls) == unique < 2*len(pairs)


def test_shards_record_rng_digests_that_replay(tmp_path: Path) -> None:
    """Shards keep sha256 digests of the draw streams instead of ~5 KB integer lists: the 'before' digest is
    the named stream's initial state, and replaying a candidate draw reproduces its tokens and 'after' digest
    (DECISIONS 88)."""
    import json
    from chunkratio.artifacts import read_jsonl
    from chunkratio.config import collection_seed
    from chunkratio.sampling import generate_batch, rng_digest
    cfg = load_config('config/crpd/smoke.json')
    models,_,_,info = load_models(cfg)
    prompts = prepare_prompts(read_jsonl(Path(cfg['prompts'])),cfg['seed'])
    a,b = [ShardStore(tmp_path/name,{'test':name}) for name in ('prefix','candidate')]
    states = collect_prefixes(prompts,models['student_ref'],cfg,info['eos'],a)
    records = collect_candidates(states,models,cfg,info['eos'],b)
    seed = collection_seed(cfg)
    prefix = a.records()[0]
    assert prefix['rng_before'] == rng_digest(stream(seed,'prefix/'+prefix['prompt']['id']))
    record = records[0]
    state,draw = record['state'],record['rng']['teacher_plus'][0]
    rng = stream(seed,f"candidate/{state['id']}/teacher_plus/0")
    assert draw['before'] == rng_digest(rng)
    tokens = generate_batch(models['teacher_plus'],[state['prefix_ids']],[cfg['horizon']-state['n']],info['eos'],
                            [rng],cfg['chunk_length'])[0]
    assert draw['after'] == rng_digest(rng)
    assert any(c['tokens'] == tokens and 'teacher_plus' in c['sources'] for c in record['candidates'])
    sizes = [len(json.dumps(r['rng'])) for r in records]
    assert max(sizes) < 2000
