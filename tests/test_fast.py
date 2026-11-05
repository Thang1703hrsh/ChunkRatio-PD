"""Optimized collection preserves samples, scores, frozen roles and resumability."""
import copy
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from chunkratio.artifacts import ShardStore, read_jsonl
from chunkratio.collection import collect_candidates, collect_prefixes, correction_features, prepare_prompts
from chunkratio.config import load_config
from chunkratio.models import load_models
from chunkratio.scoring import fingerprint


@pytest.mark.parametrize('k',[2,4])
def test_cross_state_collection_scores_and_resume(tmp_path: Path, k: int) -> None:
    cfg = {**load_config('config/crpd/smoke.json'),'candidates':k,'generation_batch_size':8}
    models,_,_,info = load_models(cfg)
    before = {name:fingerprint(model) for name,model in models.items()}
    states = collect_prefixes(prepare_prompts(read_jsonl(Path(cfg['prompts'])),0),models['student_ref'],cfg,info['eos'],
                             ShardStore(tmp_path/'p',{'test':'prefix'}))
    slow = collect_candidates(states,models,cfg,info['eos'],ShardStore(tmp_path/'slow',{}))
    fast_cfg = {**cfg,'batch_candidate_scoring':True,'score_tokens_per_batch':128}
    store = ShardStore(tmp_path/'fast',{})
    fast = collect_candidates(states,models,fast_cfg,info['eos'],store)
    assert len(fast) == len(slow)
    for actual,expected in zip(fast,slow):
        actual,expected = copy.deepcopy(actual),copy.deepcopy(expected)
        for a,b in zip(actual['candidates'],expected['candidates']):
            for name in models:
                assert a.pop('logp_'+name) == pytest.approx(b.pop('logp_'+name),abs=1e-5)
        assert actual == expected
    assert collect_candidates(states,models,fast_cfg,info['eos'],store) == fast
    assert before == {name:fingerprint(model) for name,model in models.items()}
    assert all(p.grad is None for m in models.values() for p in m.parameters())


def test_feature_cache_is_bounded_and_checks_new_attachments(tmp_path: Path, monkeypatch) -> None:
    store = ShardStore(tmp_path,{})
    rows = [{'id':str(i),'features_sha256':store.put_tensor(str(i),'features',torch.full((2,3),float(i)))}
            for i in range(3)]
    original = store.get_tensor
    reads = []
    def counted(*args):
        reads.append(args[0])
        return original(*args)
    monkeypatch.setattr(store,'get_tensor',counted)
    load = correction_features(store,max_bytes=48)  # two 24-byte tensors
    for i in [0,1,0,2,1]:
        torch.testing.assert_close(load(rows[i]),torch.full((2,3),float(i)))
    assert reads == ['0','1','2','1']
    new = {'id':'0','features_sha256':store.put_tensor('0','features',torch.full((2,3),9.))}
    torch.testing.assert_close(load(new),torch.full((2,3),9.))


def test_hf_cross_state_scores_match_individual_rows() -> None:
    transformers = pytest.importorskip('transformers')
    from chunkratio.fast import score_candidate_records
    from chunkratio.scoring import freeze, score_batch
    torch.manual_seed(20)
    model = freeze(transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)))
    records = [{'state':{'prefix_ids':[1,2]},'candidates':[{'tokens':[3,0]},{'tokens':[4]}]},
               {'state':{'prefix_ids':[1,3,4,5,6]},'candidates':[{'tokens':[7,8,0]}]}]
    score_candidate_records(records,{'rho':model},0,32)
    for record in records:
        for row in record['candidates']:
            expected = score_batch(model,[record['state']['prefix_ids']],[row['tokens']],0).item()
            assert row['logp_rho'] == pytest.approx(expected,abs=1e-5)


def test_completion_marker_rejects_stale_and_modified_exports(tmp_path: Path, monkeypatch) -> None:
    from chunkratio.artifacts import digest, write_json
    from chunkratio.fast_run import completed, output_fingerprints
    import chunkratio.fast_run as fast_run
    monkeypatch.setattr(fast_run,'code_revision',lambda:'fixed-source')
    output = tmp_path/'run'
    cfg = {'output':str(output),'steps':100}
    assert not completed(cfg)
    for name in ('diagnostics.json','resolved_config.json','manifest.json','student/config.json'):
        write_json(output/name,{})
    (output/'student/model.safetensors').write_bytes(b'test-weights')
    marker = {'config_hash':digest(cfg),'code_revision':'fixed-source','artifacts':output_fingerprints(output)}
    write_json(output/'fast_completed.json',marker)
    assert completed(cfg)
    with pytest.raises(ValueError,match='another config'):
        completed({**cfg,'steps':200})
    (output/'student/model.safetensors').write_bytes(b'changed-weights')
    with pytest.raises(ValueError,match='changed or corrupted'):
        completed(cfg)


def test_fast_config_preserves_models_and_declares_compact_changes() -> None:
    from chunkratio.fast_run import fast_config
    base = load_config('config/crpd/smoke.json')
    before = copy.deepcopy(base)
    cfg = fast_config(base,Path('output/isolated'),Path('output/isolated/prompts.jsonl'),'crpd_main')
    assert base == before and cfg['models'] == base['models']
    assert cfg['candidates'] == 2 and cfg['steps'] == 100
    assert cfg['selection_baseline'] and not cfg['selection_fallback_to_reference']
    assert cfg['collection_decoding'] == base['collection_decoding']
    assert cfg.get('policy_autocast_dtype') == base.get('policy_autocast_dtype')
