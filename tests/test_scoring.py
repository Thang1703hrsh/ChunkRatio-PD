"""Causal scoring/masks/gradient fixtures (§§5,6,12)."""
import pytest

torch = pytest.importorskip('torch',reason='PyTorch unavailable for active interpreter')
from chunkratio.sampling import generate, stream
from chunkratio.scoring import (continuation_increments, continuation_increments_batch, fingerprint, score_batch,
                                score_tokens, state_features, trajectory_features, trajectory_features_batch)
from chunkratio.tiny import TinyLM


def test_scored_positions_only_match_full_log_softmax() -> None:
    """Gathering response positions before the log-softmax equals normalising every position."""
    gen = torch.Generator().manual_seed(5)
    rows,width,vocab,eos = 5,12,9,0
    logits = torch.randn(rows,width,vocab,generator=gen,dtype=torch.float64,requires_grad=True)
    ids = torch.randint(1,vocab,(rows,width),generator=gen)
    prompt_lengths = torch.tensor([1,3,2,5,11])
    sequence_lengths = torch.tensor([12,9,12,7,12])
    ids[2,6] = eos
    attention = (torch.arange(width)[None,:] < sequence_lengths[:,None]).long()
    values,mask = score_tokens(logits,ids,attention,prompt_lengths,sequence_lengths,eos)
    full = logits[:,:-1].log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1).masked_fill(~mask,0)
    torch.testing.assert_close(values,full,atol=0,rtol=0)
    assert mask[2].tolist().index(True) == 1 and not mask[2,6:].any()
    grad, = torch.autograd.grad(values.sum(),logits)
    expected, = torch.autograd.grad(full.sum(),logits)
    torch.testing.assert_close(grad,expected,atol=1e-12,rtol=0)


def test_fingerprint_bytes_unchanged_and_fast() -> None:
    """Same digest as hashing each tensor's storage bytes; a 64 MB layer hashes in well under a second (§4.1)."""
    import hashlib
    import time
    def storage_digest(model: torch.nn.Module) -> str:
        h = hashlib.sha256(type(model).__qualname__.encode())
        h.update(repr(model).encode())
        for name,tensor in sorted(model.state_dict().items()):
            h.update(str((name,tuple(tensor.shape),str(tensor.dtype))).encode())
            h.update(bytes(tensor.detach().cpu().contiguous().clone().untyped_storage()))
        return h.hexdigest()
    model = torch.nn.Sequential(torch.nn.Linear(7,5),torch.nn.Linear(5,3).to(torch.bfloat16))
    model.register_buffer('ints',torch.arange(11))
    model.register_buffer('mask',torch.tensor([True,False,True]))
    model.register_buffer('scalar16',torch.tensor(2.5,dtype=torch.float16))
    model.register_buffer('view',torch.randn(4,6)[:,1:4])
    assert fingerprint(model) == storage_digest(model)
    start = time.perf_counter()
    fingerprint(torch.nn.Linear(4096,4096))
    assert time.perf_counter()-start < 2.


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
    # One causal pass gives every state's feature; each equals the prefix run alone.
    trajectory = trajectory_features(model,[1,2],[3,4,0])
    assert trajectory.shape == (3,8)
    for t,state in enumerate([[1,2],[1,2,3],[1,2,3,4]]):
        torch.testing.assert_close(trajectory[t],state_features(model,[state])[0])
    policy_scores = score_batch(model,prefixes,chunks,0,frozen=False)
    assert policy_scores.requires_grad
    policy_scores.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


class GradProbe(TinyLM):
    """Record autograd mode inside every forward, independent of requires_grad flags (§6.2)."""

    def forward(self, *args: object, **kwargs: object) -> object:
        """Delegate after recording grad mode (§6.2)."""
        self.modes.append(torch.is_grad_enabled())
        return super().forward(*args,**kwargs)


def test_frozen_roles_always_run_under_no_grad() -> None:
    """Sampling, increments, features and frozen scoring never build graphs (§§2,6.2)."""
    torch.manual_seed(5)
    probe = GradProbe(vocab_size=7,hidden_size=8)
    probe.modes = []
    assert all(p.requires_grad for p in probe.parameters())
    tokens = generate(probe,[1,2],3,0,stream(1,'probe'))
    assert not continuation_increments(probe,[1,2],[3,4],0).requires_grad
    assert not state_features(probe,[[1,2]]).requires_grad
    assert not trajectory_features(probe,[1,2],[3,4]).requires_grad
    assert not score_batch(probe,[[1]],[[2]],0,frozen=True).requires_grad
    assert len(probe.modes) == len(tokens)+4 and not any(probe.modes)
    assert not continuation_increments_batch(probe,[[1,2]],[[3,4]],0)[0].requires_grad
    assert not trajectory_features_batch(probe,[[1,2]],[[3,4]])[0].requires_grad
    assert len(probe.modes) == len(tokens)+6 and not any(probe.modes)
    score_batch(probe,[[1]],[[2]],0,frozen=False)
    assert probe.modes[-1]


def test_batched_correction_scoring_matches_single_rows() -> None:
    """Length-grouped, left-padded batches give each trajectory its own increments and features (§§6.2,7.3)."""
    torch.manual_seed(9)
    model = TinyLM(vocab_size=7,hidden_size=8)
    prefixes = [[1,2],[3],[1,4,5,6],[2,2,2],[5]]
    suffixes = [[3,0],[4,4,6,1],[2],[6,5,0],[]]
    # A budget of 4 tokens puts every row in its own forward; 8192 puts them all in one.
    for budget in (4,8192):
        increments = continuation_increments_batch(model,prefixes,suffixes,0,budget)
        for prefix,suffix,got in zip(prefixes,suffixes,increments):
            torch.testing.assert_close(got,continuation_increments(model,prefix,suffix,0))
        rows = [i for i,suffix in enumerate(suffixes) if suffix]
        features = trajectory_features_batch(model,[prefixes[i] for i in rows],[suffixes[i] for i in rows],budget)
        for i,got in zip(rows,features):
            torch.testing.assert_close(got,trajectory_features(model,prefixes[i],suffixes[i]))
    # Generated suffixes end at their first EOS; anything after it is a caller error.
    with pytest.raises(AssertionError):
        continuation_increments_batch(model,[[1]],[[0,3]],0)


def test_tail_scoring_counts_first_eos_once_and_ignores_later_tokens() -> None:
    """score_batch scores the chunk only: the first EOS once, nothing after it (§6.1)."""
    torch.manual_seed(13)
    model = TinyLM(vocab_size=7,hidden_size=8)
    with_tail = score_batch(model,[[1,2]],[[3,0,4,5]],0)
    torch.testing.assert_close(with_tail,score_batch(model,[[1,2]],[[3,0]],0))
    assert not torch.allclose(score_batch(model,[[1,2]],[[3,0]],0),score_batch(model,[[1,2]],[[3]],0))
    torch.testing.assert_close(score_batch(model,[[1,2]],[[]],0),torch.zeros(1))
