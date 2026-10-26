"""Causal scoring/masks/gradient fixtures (§§5,6,12)."""
import pytest

torch = pytest.importorskip('torch',reason='PyTorch unavailable for active interpreter')
from chunkratio.scoring import score_batch, score_tokens, state_features
from chunkratio.tiny import TinyLM


def test_shift_mask_prompt_padding_eos() -> None:
    """Hand fixture: prompt/PAD/post-EOS excluded, EOS=PAD included once (§6.1)."""
    logits = torch.zeros((1,6,4),dtype=torch.float64,requires_grad=True)
    ids = torch.tensor([[1,1,2,0,3,0]])
    values, mask = score_tokens(logits,ids,torch.ones_like(ids),torch.tensor([2]),torch.tensor([6]),0)
    assert mask.tolist() == [[False,True,True,False,False]]
    assert values.sum().item() == pytest.approx(-2*__import__('math').log(4))
    values.sum().backward()
    assert logits.grad[0,0].abs().sum() == 0
    assert logits.grad[0,3:].abs().sum() == 0


def test_batching_features_causality_and_frozen_gradients() -> None:
    """Right padding/EOS=PAD batching, unseen suffixes and no_grad (§6)."""
    torch.manual_seed(3)
    model = TinyLM(vocab_size=7,hidden_size=8)
    prefixes, chunks = [[1,2],[1]], [[3,0],[4]]
    result = score_batch(model,prefixes,chunks,0,frozen=True)
    assert not result.requires_grad
    for i in range(2):
        single = score_batch(model,[prefixes[i]],[chunks[i]],0,frozen=True)
        torch.testing.assert_close(single,result[i:i+1])
    f = state_features(model,[[1,2],[1]])
    torch.testing.assert_close(f[0],state_features(model,[[1,2]])[0])
    extended = model(torch.tensor([[1,2,3,4]]),output_hidden_states=True).hidden_states[-1][0,1].detach()
    torch.testing.assert_close(f[0],extended)
    policy_scores = score_batch(model,prefixes,chunks,0,frozen=False)
    assert policy_scores.requires_grad
    policy_scores.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
