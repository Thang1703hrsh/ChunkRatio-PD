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
