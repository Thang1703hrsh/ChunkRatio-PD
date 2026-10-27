"""Dependency-free artifact, split and configuration verification (§10)."""
from pathlib import Path

import pytest

from chunkratio.artifacts import ShardStore, digest, read_jsonl, stage_key


def test_read_jsonl_keeps_unicode_line_separators(tmp_path: Path) -> None:
    """U+2028/U+2029/U+0085 inside strings stay in one record (the Deita prepare-data crash)."""
    import json
    rows = [{'text':'a b c\x85d\x0ce'},{'text':'second'}]
    path = tmp_path/'rows.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    assert read_jsonl(path) == rows


def test_resume_integrity_and_incompatible_metadata(tmp_path: Path) -> None:
    """Resume no duplicates; incompatible or corrupted cache is rejected (§12)."""
    store = ShardStore(tmp_path,{'model':'first','horizon':512})
    payload = {'tokens':[1,2],'rng_after':[5,7]}
    store.put('job',payload)
    store.put('job',payload)
    assert ShardStore(tmp_path,store.metadata).records() == [payload]
    with pytest.raises(ValueError,match='incompatible'):
        ShardStore(tmp_path,{'model':'second'})
    with pytest.raises(ValueError,match='conflicting'):
        store.put('job',{'tokens':[3]})
    file = tmp_path/(digest('job')+'.jsonl')
    file.write_text(file.read_text().replace('first','other').replace('[1, 2]','[1, 3]'))
    with pytest.raises(ValueError,match='corrupt'):
        store.get('job')


def test_tensor_shards_hash_elements_fast_and_store_only_the_view(tmp_path: Path) -> None:
    """Digest covers shape, dtype and the view's own bytes; a 16 MB shard round-trips quickly (§10)."""
    import hashlib
    import time
    torch = pytest.importorskip('torch')
    from chunkratio.artifacts import tensor_digest
    hidden = torch.randn(1,40,8)
    view = hidden[0,10:]
    expected = hashlib.sha256(str((tuple(view.shape),str(view.dtype))).encode())
    expected.update(bytes(view.contiguous().clone().untyped_storage()))
    assert tensor_digest(view) == expected.hexdigest() == tensor_digest(view.clone())
    store = ShardStore(tmp_path,{'kind':'features'})
    sha = store.put_tensor('job','features',view)
    loaded = store.get_tensor('job','features',sha)
    assert torch.equal(loaded,view) and loaded.untyped_storage().nbytes() == view.numel()*4
    big = torch.randn(4096,1024)
    start = time.perf_counter()
    store.get_tensor('big','features',store.put_tensor('big','features',big))
    assert time.perf_counter()-start < 5.


def test_stage_identity() -> None:
    """Dependency changes yield isolated cache namespaces (§10)."""
    assert stage_key('targets',{'head':'a'}) != stage_key('targets',{'head':'b'})
    assert stage_key('pairs',{'alpha':.3}) != stage_key('pairs',{'alpha':1})


def test_atomic_replace_retries_transient_windows_locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A briefly locked destination (OneDrive/antivirus) is retried; a persistent lock still fails (§10)."""
    import os
    import chunkratio.artifacts as artifacts
    real, calls = os.replace, []
    def flaky(source: str, destination: object) -> None:
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError('locked')
        real(source,destination)
    monkeypatch.setattr(artifacts.os,'replace',flaky)
    monkeypatch.setattr(artifacts.time,'sleep',lambda seconds: None)
    artifacts.write_json(tmp_path/'a.json',{'x':1})
    assert len(calls) == 3 and (tmp_path/'a.json').read_text().strip().startswith('{')
    def locked(source: str, destination: object) -> None:
        raise PermissionError('locked')
    monkeypatch.setattr(artifacts.os,'replace',locked)
    with pytest.raises(PermissionError):
        artifacts.write_json(tmp_path/'b.json',{'x':1})
    assert not list(tmp_path.glob('b.json*'))
