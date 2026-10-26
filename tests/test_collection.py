"""Correction rollouts, suffix labels, causal features and global budget (§§5.2,6.2,7.3)."""
from pathlib import Path

import pytest

torch = pytest.importorskip('torch',reason='collection requires PyTorch')
from chunkratio.artifacts import ShardStore, read_jsonl
from chunkratio.collection import (collect_candidates, collect_correction, collect_prefixes, correction_features,
                                  prepare_prompts)
from chunkratio.config import load_config
from chunkratio.models import load_models
from chunkratio.sampling import generate, stream, terminal
from chunkratio.scoring import state_features


def token_logps(model: torch.nn.Module, prefix: list[int], tokens: list[int]) -> list[float]:
    """Independent unbatched per-token recomputation, one forward per position (§6.1)."""
    values = []
    with torch.no_grad():
        for t,token in enumerate(tokens):
            logits = model(torch.tensor([prefix+tokens[:t]])).logits[0,-1].double()
            values.append(logits.log_softmax(-1)[token].item())
    return values


@pytest.mark.parametrize('variant,source,numerator',[('crpd_main','student_ref','teacher_plus'),
                                                     ('crpd_mu_teacher','teacher_plus','student_ref')])
def test_correction_rollouts_labels_and_features(tmp_path: Path, variant: str, source: str, numerator: str) -> None:
    """Rollouts from rho (T+ for the v1.0 ablation) to EOS/H; reverse-cumsum labels per state (§7.3)."""
    cfg = {**load_config('config/crpd/smoke.json'),'variant':variant}
    models,_,_,info = load_models(cfg)
    eos, horizon = info['eos'], cfg['horizon']
    prompts = prepare_prompts(read_jsonl(Path(cfg['prompts'])),cfg['seed'])
    states = collect_prefixes(prompts,models['student_ref'],cfg,eos,ShardStore(tmp_path/'p',{'t':'p'}))
    records = collect_candidates(states[:3],models,cfg,eos,ShardStore(tmp_path/'c',{'t':'c'}))
    store = ShardStore(tmp_path/'r',{'t':'r'})
    rows = collect_correction(records,models,cfg,eos,store)
    assert rows and all('features' not in row for row in rows)
    load = correction_features(store)
    for row in rows:
        seed, tokens = row['seed'], row['tokens']
        assert set(row['generated_tokens']) == {source}
        assert set(row['scored_tokens']) == {numerator,'teacher_base'}
        assert tokens == generate(models[source],seed['prefix_ids'],horizon-seed['n'],eos,
                                  stream(cfg['seed'],'correction/'+row['id']))
        assert not terminal(seed['response_ids'],horizon,eos)
        assert terminal(seed['response_ids']+tokens,horizon,eos)
        increments = [a-b for a,b in zip(token_logps(models[numerator],seed['prefix_ids'],tokens),
                                          token_logps(models['teacher_base'],seed['prefix_ids'],tokens))]
        assert row['labels'] == pytest.approx([sum(increments[t:]) for t in range(len(tokens))],abs=1e-5)
        assert row['budgets'] == pytest.approx([(horizon-seed['n']-t)/horizon for t in range(len(tokens))])
        features = load(row)
        assert features.shape == (len(tokens),12) and features.dtype == torch.float32
        for t in range(len(tokens)):
            alone = state_features(models['student_ref'],[seed['prefix_ids']+tokens[:t]])[0]
            torch.testing.assert_close(features[t],alone,atol=1e-6,rtol=0)


def test_candidate_budget_is_global_and_chunk_cutoff_nonterminal(tmp_path: Path) -> None:
    """Candidates stop at EOS or min(L,H-n); only EOS/H is terminal (§5.2)."""
    cfg = {**load_config('config/crpd/smoke.json'),'chunk_length':4}
    models,_,_,info = load_models(cfg)
    eos, horizon = info['eos'], cfg['horizon']
    late = {'id':'late','prompt_id':'p','split':'train','prefix_ids':[1,2]+[3]*5,'response_ids':[3]*5,'n':5}
    early = {'id':'early','prompt_id':'p','split':'train','prefix_ids':[1,2],'response_ids':[],'n':0}
    records = {r['state']['id']:r for r in collect_candidates([late,early],models,cfg,eos,
                                                              ShardStore(tmp_path/'c',{'t':'c'}))}
    assert all(len(c['tokens']) == 1 and c['terminal'] for c in records['late']['candidates'])
    for c in records['early']['candidates']:
        assert 1 <= len(c['tokens']) <= 4 and eos not in c['tokens'][:-1]
        assert c['terminal'] == (c['tokens'][-1] == eos)
