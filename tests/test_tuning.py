"""Opt-in tuning knobs: SFT turn fitting, policy LR schedule, config checks and student autocast."""
import math

import pytest

from chunkratio.data import fit_conversation


def test_fit_conversation_keeps_latest_fitting_assistant_turn() -> None:
    """Earlier turns stay as context; the supervised answer is never cut; None if nothing fits."""
    row = {'messages':[{'role':'user','content':'u1'},{'role':'assistant','content':'a1'},
                       {'role':'user','content':'u2'},{'role':'assistant','content':'a2'},
                       {'role':'user','content':'u3'}],'chosen':'a3','rejected':None,'split':'train'}
    length = lambda turns: 10*len(turns)
    full = fit_conversation(row,length,60)
    assert full['messages'] == row['messages'] and full['chosen'] == 'a3' and full['split'] == 'train'
    middle = fit_conversation(row,length,45)
    assert [t['content'] for t in middle['messages']] == ['u1','a1','u2'] and middle['chosen'] == 'a2'
    first = fit_conversation(row,length,20)
    assert [t['content'] for t in first['messages']] == ['u1'] and first['chosen'] == 'a1'
    assert fit_conversation(row,length,19) is None


def test_lr_schedule_default_constant_and_opt_in_warmup_cosine() -> None:
    """Default stays the constant primary schedule; cosine warms up linearly then decays toward 0."""
    pytest.importorskip('torch')
    from chunkratio.training import lr_multiplier
    constant = lr_multiplier({'steps':10})
    assert [constant(s) for s in range(10)] == [1.]*10
    cosine = lr_multiplier({'steps':100,'lr_schedule':'cosine','warmup_ratio':.1})
    assert cosine(0) == pytest.approx(.1) and cosine(9) == pytest.approx(1.)
    assert cosine(10) == pytest.approx(1.) and cosine(55) == pytest.approx(.5)
    assert 0 < cosine(99) < .01
    values = [cosine(s) for s in range(10,100)]
    assert all(a >= b for a,b in zip(values,values[1:])) and not any(math.isnan(v) for v in values)
    warm_constant = lr_multiplier({'steps':20,'warmup_ratio':.25})
    assert warm_constant(0) == pytest.approx(.2) and warm_constant(19) == 1.


@pytest.mark.parametrize('key,value',[('lr_schedule','linear'),('warmup_ratio',1.),('warmup_ratio',-.1),
                                      ('policy_autocast_dtype','float16')])
def test_config_rejects_invalid_tuning_values(key: str, value: object) -> None:
    """Unknown schedules, out-of-range warmup and non-bf16 autocast fail closed."""
    from chunkratio.config import load_config, validate
    cfg = load_config('config/crpd/smoke.json')
    validate(cfg)
    cfg[key] = value
    with pytest.raises(ValueError):
        validate(cfg)


def test_student_autocast_keeps_fp32_weights_scores_and_grads() -> None:
    """bf16 autocast changes only the forward compute: FP32 chunk scores, finite FP32 gradients."""
    torch = pytest.importorskip('torch')
    from chunkratio.scoring import score_batch
    from chunkratio.tiny import TinyLM
    from chunkratio.training import student_precision
    model = TinyLM(vocab_size=7,hidden_size=8)
    with student_precision({'policy_autocast_dtype':'bfloat16','device':'cpu'}):
        scores = score_batch(model,[[1,2],[3]],[[4,5,6],[2,6]],6,frozen=False)
    assert scores.dtype == torch.float32 and torch.isfinite(scores).all()
    scores.sum().backward()
    assert all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    with student_precision({'device':'cpu'}):
        exact = score_batch(model,[[1,2],[3]],[[4,5,6],[2,6]],6,frozen=False)
    assert torch.allclose(scores.detach(),exact.detach(),atol=.1)


def margin_trainer(cfg: dict) -> tuple:
    """Student = exact rho copy, one pair whose cached rho scores are FP32 as collection writes them."""
    import copy
    torch = pytest.importorskip('torch')
    from chunkratio.scoring import score_batch
    from chunkratio.tiny import TinyLM
    from chunkratio.training import ChunkRatioTrainer
    torch.manual_seed(3)
    rho = TinyLM(vocab_size=7,hidden_size=8)
    prefix,chunks = [1,2,3],{'c':[4,5,1,2],'d':[2,6]}
    pair = {'state':{'prefix_ids':prefix,'prompt_id':'p','id':'s'},'weight':1.}
    for side,tokens in chunks.items():
        pair[side] = {'tokens':tokens,'target_score':0.,'logp_student_ref':score_batch(rho,[prefix],[tokens],6).item()}
    prompts = [{'split':'train','prompt_ids':prefix,'chosen_ids':chunks['c'],'rejected_ids':chunks['d']}]
    trainer = ChunkRatioTrainer(copy.deepcopy(rho),rho,{'route':'reference_mean','pairs':[pair]},prompts,
                                {'seed':0,'alpha':.3,'device':'cpu','anchor_beta':.05,**cfg},6)
    return trainer,pair,rho


def test_token_budget_microbatches_keep_the_step_gradient() -> None:
    """policy_tokens_per_microbatch regroups one step's pairs by length; the accumulated gradient equals
    the fixed-size split's up to summation order, and every pair is used exactly once."""
    import copy
    torch = pytest.importorskip('torch')
    from chunkratio.scoring import score_batch
    from chunkratio.tiny import TinyLM
    from chunkratio.training import ChunkRatioTrainer
    torch.manual_seed(5)
    rho = TinyLM(vocab_size=7,hidden_size=8)
    pairs = []
    for i in range(12):
        prefix = [1]+[2+(i+j)%4 for j in range(1+3*(i%5))]
        pair = {'state':{'prefix_ids':prefix,'prompt_id':f'p{i%3}','id':f's{i}'},'weight':1/12,'id':str(i)}
        for side,tokens,target in (('c',[4,5,1,2][:1+i%4],.3*i),('d',[2,6],-.2*i)):
            pair[side] = {'tokens':tokens,'target_score':target,'logp_student_ref':score_batch(rho,[prefix],[tokens],6).item()}
        pairs.append(pair)
    width = lambda p: len(p['state']['prefix_ids'])+max(len(p['c']['tokens']),len(p['d']['tokens']))
    grads = {}
    for name,cfg in (('fixed',{'policy_microbatch_size':5}),('budget',{'policy_tokens_per_microbatch':24})):
        trainer = ChunkRatioTrainer(copy.deepcopy(rho),rho,{'route':'reference_mean','pairs':pairs},[],
                                    {'seed':0,'alpha':.3,'device':'cpu',**cfg},6)
        groups = trainer.microbatches(pairs)
        assert sorted(p['id'] for g in groups for p in g) == sorted(p['id'] for p in pairs)
        if name == 'budget':
            assert len(groups) > 1 and all(len(g) == 1 or len(g)*max(map(width,g)) <= 24 for g in groups)
        for micro in groups:
            loss,_ = trainer.batch_loss(micro)
            (loss*len(micro)/len(pairs)).backward()
        grads[name] = [p.grad.clone() for p in trainer.student.parameters()]
    for a,b in zip(grads['fixed'],grads['budget']):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def recorded_margin(trainer: object, pair: dict, monkeypatch: pytest.MonkeyPatch) -> object:
    """The student margin batch_loss hands to the loss."""
    import chunkratio.training as training
    margins = []
    original = training.pair_loss
    def spy(*args: object) -> tuple:
        """Record the margin of the real pair loss."""
        result = original(*args)
        margins.append(result[1])
        return result
    monkeypatch.setattr(training,'pair_loss',spy)
    trainer.batch_loss([pair])
    return margins[0]


def test_margin_is_exactly_zero_at_initialization_under_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    """a_theta = 0 at student = rho: autocast scores rho the same way instead of comparing with FP32."""
    trainer,pair,_ = margin_trainer({'policy_autocast_dtype':'bfloat16'})
    assert (recorded_margin(trainer,pair,monkeypatch) == 0).all()


@pytest.mark.parametrize('autocast',[None,'bfloat16'])
def test_shared_prefix_margin_is_exactly_zero_at_initialization(monkeypatch: pytest.MonkeyPatch, autocast: object) -> None:
    """With shared_prefix_scoring the student and rho scores come from the same computation, so the
    margin a = log pi - log rho is exactly zero at student = rho, with or without autocast."""
    trainer,pair,_ = margin_trainer({'policy_autocast_dtype':autocast,'shared_prefix_scoring':True})
    assert (recorded_margin(trainer,pair,monkeypatch) == 0).all()


def test_autocast_anchor_is_log2_at_initialization() -> None:
    """The anchor's rho delta shares the student's precision, so its margin also starts at zero."""
    import math
    trainer,_,_ = margin_trainer({'policy_autocast_dtype':'bfloat16','anchor_lambda':.1})
    assert trainer.anchor_loss().item() == pytest.approx(math.log(2),abs=1e-7)


def test_selection_baseline_records_rho_without_competing(tmp_path: object) -> None:
    """Step 0 holds rho's score; a higher rho score still cannot become the exported checkpoint."""
    settings = {'steps':2,'optimizer':'AdamW','learning_rate':1e-3,'variant':'crpd_main','batch_size':1,'selection_interval':1}
    scores = iter([5.,1.,2.])
    trainer,_,_ = margin_trainer({**settings,'selection_baseline':True})
    trainer.selector = lambda student: next(scores)
    result = trainer.train(tmp_path/'with.pt')
    assert [h['step'] for h in result['history']] == [0,1,2] and result['history'][0]['selection_score'] == 5.
    assert result['selection_score'] == 2.
    scores = iter([1.,2.])
    trainer,_,_ = margin_trainer(settings)
    trainer.selector = lambda student: next(scores)
    assert [h['step'] for h in trainer.train(tmp_path/'without.pt')['history']] == [1,2]


def test_fp32_policy_keeps_cached_rho_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without autocast the primary path stays as it was: no extra rho forward, cached scores used."""
    from chunkratio.cost import ForwardMeter
    trainer,pair,rho = margin_trainer({})
    meter = ForwardMeter({'rho':rho})
    margin = recorded_margin(trainer,pair,monkeypatch)
    meter.close()
    assert meter.counts['rho']['forward_calls'] == 0
    assert margin.abs().max() < 1e-5


class PromptSelector:
    """Selector returning preset per-prompt rewards in order, recording them like reward_selector does."""

    def __init__(self, rounds: list[list[float]]) -> None:
        """Rewards per call."""
        self.rounds = iter(rounds)
        self.last_scores: list[float] | None = None

    def __call__(self, student: object) -> float:
        """Mean of this call's rewards."""
        self.last_scores = next(self.rounds)
        return sum(self.last_scores)/len(self.last_scores)


def test_selection_reports_paired_gain_and_optional_fallback(tmp_path: object) -> None:
    """Per-prompt rewards give a paired comparison with rho; the opt-in fallback exports rho when nothing beats it."""
    from chunkratio.scoring import fingerprint
    settings = {'steps':2,'optimizer':'AdamW','learning_rate':1e-3,'variant':'crpd_main','batch_size':1,
                'selection_interval':1,'selection_baseline':True}
    trainer,_,rho = margin_trainer(settings)
    trainer.selector = PromptSelector([[1.,1.,1.],[2.,0.,1.],[3.,2.,1.]])
    result = trainer.train(tmp_path/'gain.pt')
    assert result['improved_over_reference'] and result['selected'] == 'checkpoint'
    paired = result['paired_vs_reference']
    assert paired['step'] == 2 and paired['mean_difference'] == 1. and (paired['wins'],paired['losses']) == (2,0)
    assert result['history'][0]['selection_prompt_scores'] == [1.,1.,1.]
    trainer,_,rho = margin_trainer({**settings,'selection_fallback_to_reference':True})
    trainer.selector = PromptSelector([[5.,5.],[1.,2.],[2.,1.]])
    result = trainer.train(tmp_path/'fallback.pt')
    assert result['improved_over_reference'] is False and result['selected'] == 'reference'
    assert fingerprint(trainer.student) == fingerprint(rho)
    # The reported score is the exported model's (rho, 5.0), not the best trained checkpoint's (1.5).
    assert result['selection_score'] == 5. and result['best_checkpoint_score'] == 1.5
    assert result['selection_rule'] == 'select_mean_reward_with_reference_fallback'
    trainer,_,_ = margin_trainer({**settings,'selection_baseline':False,'selection_fallback_to_reference':True})
    trainer.selector = PromptSelector([[1.]])
    with pytest.raises(ValueError,match='needs selection_baseline'):
        trainer.train(tmp_path/'invalid.pt')
