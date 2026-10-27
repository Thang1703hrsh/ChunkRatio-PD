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
