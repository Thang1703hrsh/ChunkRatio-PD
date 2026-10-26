"""Dependency-free artifact, split and configuration verification (§10)."""
from pathlib import Path

import pytest

from chunkratio.artifacts import ShardStore, digest, stage_key


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


def test_stage_identity() -> None:
    """Dependency changes yield isolated cache namespaces (§10)."""
    assert stage_key('targets',{'head':'a'}) != stage_key('targets',{'head':'b'})
    assert stage_key('pairs',{'alpha':.3}) != stage_key('pairs',{'alpha':1})
