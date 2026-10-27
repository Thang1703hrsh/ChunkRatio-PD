"""Batched KV-cached collection sampling: batch invariance and per-row generators (§§5.3,10)."""
import pytest

torch = pytest.importorskip('torch',reason='sampling requires PyTorch')
from chunkratio.sampling import generate, generate_batch, stream
from chunkratio.tiny import TinyLM

PREFIXES = [[1,2,3],[1],[4,5,6,7,8],[2,2],[3,9,1,4]]
REMAINING = [6,3,6,1,0]


def check_batch_invariance(model: torch.nn.Module, eos: int, chunk_length: int | None) -> list[list[int]]:
    """Each row equals its own single-row run with the same semantic seed (§10)."""
    names = [f'row/{i}' for i in range(len(PREFIXES))]
    batched = generate_batch(model,PREFIXES,REMAINING,eos,[stream(7,n) for n in names],chunk_length)
    for prefix,remaining,name,tokens in zip(PREFIXES,REMAINING,names,batched):
        assert tokens == generate(model,prefix,remaining,eos,stream(7,name),chunk_length)
        limit = min(remaining,chunk_length) if chunk_length else remaining
        assert len(tokens) <= limit and eos not in tokens[:-1]
    # A different batch composition (reversed, partial) gives the same per-row tokens.
    order = [3,0,2]
    subset = generate_batch(model,[PREFIXES[i] for i in order],[REMAINING[i] for i in order],eos,
                            [stream(7,names[i]) for i in order],chunk_length)
    assert subset == [batched[i] for i in order]
    return batched


@pytest.mark.parametrize('chunk_length',[None,2])
def test_tiny_batch_invariance_and_budget(chunk_length: int | None) -> None:
    """Left-padded TinyLM rows reproduce unbatched sampling; budgets and EOS respected (§5.2)."""
    torch.manual_seed(11)
    model = TinyLM()
    batched = check_batch_invariance(model,0,chunk_length)
    assert batched[4] == [] and any(batched)


def test_per_row_generators_are_independent() -> None:
    """Advancing one row's generator never changes another row's draws (§10)."""
    torch.manual_seed(12)
    model = TinyLM()
    first = generate_batch(model,PREFIXES[:2],[5,5],0,[stream(1,'a'),stream(1,'b')])
    advanced = stream(1,'a')
    torch.rand(5,generator=advanced)
    second = generate_batch(model,PREFIXES[:2],[5,5],0,[advanced,stream(1,'b')])
    assert second[1] == first[1]
