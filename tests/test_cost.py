"""Measured token costs include prefill and do not confuse PAD with EOS (§11.4)."""
import pytest

torch = pytest.importorskip('torch',reason='forward instrumentation requires PyTorch')
from chunkratio.cost import ForwardMeter
from chunkratio.scoring import score_batch
from chunkratio.tiny import TinyLM


def test_forward_measurement_and_detach() -> None:
    """Frozen vs student forward counts use attention positions (§11.4)."""
    model = TinyLM()
    meter = ForwardMeter({'student':model})
    score_batch(model,[[1,2],[1]],[[3,0],[0]],0,frozen=False)
    assert meter.counts['student'] == {'forward_calls':1,'input_tokens':6,'with_grad_input_tokens':6}
    score_batch(model,[[1]],[[0]],0)
    assert meter.counts['student']['input_tokens'] == 8
    assert meter.counts['student']['with_grad_input_tokens'] == 6
    meter.close()
    score_batch(model,[[1]],[[0]],0)
    assert meter.counts['student']['forward_calls'] == 2


def test_kv_cached_steps_count_only_new_tokens() -> None:
    """Cached history in the attention mask is not re-counted; left padding of the prefill still is not."""
    from chunkratio.sampling import generate_batch, stream
    model = TinyLM()
    meter = ForwardMeter({'student':model})
    state,mask = None,torch.tensor([[1,1,1]])
    ids = torch.tensor([[1,2,3]])
    for _ in range(4):
        out = model(ids,attention_mask=mask,use_cache=True,past_key_values=state)
        state,ids,mask = out.past_key_values,torch.tensor([[4]]),torch.cat([mask,torch.ones(1,1,dtype=torch.long)],dim=1)
    assert meter.counts['student'] == {'forward_calls':4,'input_tokens':6,'with_grad_input_tokens':6}
    meter.close()
    meter = ForwardMeter({'student':model})
    prefixes = [[1,2,3],[4]]
    generate_batch(model,prefixes,[3,3],0,[stream(0,'a'),stream(0,'b')])
    calls = meter.counts['student']['forward_calls']
    assert meter.counts['student']['input_tokens'] == 4+2*(calls-1)
    meter.close()


def test_finished_rows_leave_the_batch() -> None:
    """After a row ends, later steps forward only the rows still generating."""
    from chunkratio.sampling import generate_batch, stream
    torch.manual_seed(0)
    model = TinyLM()
    meter = ForwardMeter({'student':model})
    lengths = list(map(len,generate_batch(model,[[1,2,3],[4]],[4,1],0,[stream(0,'a'),stream(0,'b')])))
    assert lengths[1] == 1 and lengths[0] > 1
    # Prefill counts both prompts (4 tokens); step k forwards only the rows longer than k tokens.
    assert meter.counts['student']['forward_calls'] == max(lengths)
    assert meter.counts['student']['input_tokens'] == 4+sum(sum(n > k for n in lengths) for k in range(1,max(lengths)))
    meter.close()
