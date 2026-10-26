"""Atomic versioned JSON artifacts and resumable deterministic shards (§10)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = 1


def digest(value: Any) -> str:
    """Canonical metadata/content identity (§10)."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    """Commit a whole file using same-directory replace on Windows/Linux (§10)."""
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    """Reject nonfinite values and persist readable JSON (§10)."""
    atomic_bytes(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def read_json(path: Path) -> Any:
    """Read a UTF-8 artifact, including legacy BOM headers (§10)."""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    """Read canonical prompt or evaluation records (§10)."""
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


class ShardStore:
    """Single-writer immutable job shards; completed jobs are never duplicated (§10)."""

    def __init__(self, path: Path, metadata: dict[str,Any]) -> None:
        """Reject incompatible provenance before reusing any record (§10)."""
        self.path = path
        self.metadata = {'schema':SCHEMA,**metadata}
        self.path.mkdir(parents=True,exist_ok=True)
        header = path/'metadata.json'
        if header.exists() and read_json(header) != self.metadata:
            raise ValueError(f'incompatible cache metadata: {path}')
        if not header.exists():
            write_json(header,self.metadata)

    def get(self, job_id: str) -> dict[str,Any] | None:
        """Load an atomically completed job with integrity verification (§10)."""
        path = self.path/(digest(job_id)+'.jsonl')
        if not path.exists():
            return None
        record = read_json(path)
        if record['job_id'] != job_id or digest(record['payload']) != record['sha256']:
            raise ValueError(f'corrupt shard {path}')
        return record['payload']

    def put(self, job_id: str, payload: dict[str,Any]) -> None:
        """Persist record plus RNG states as a single transaction (§10)."""
        previous = self.get(job_id)
        if previous is not None:
            if previous != payload:
                raise ValueError(f'conflicting completed shard {job_id}')
            return
        record = {'job_id':job_id,'payload':payload,'sha256':digest(payload)}
        atomic_bytes(self.path/(digest(job_id)+'.jsonl'),(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode())

    def records(self) -> list[dict[str,Any]]:
        """Read every completed shard deterministically (§10)."""
        result = []
        for path in sorted(self.path.glob('*.jsonl')):
            record = read_json(path)
            payload = self.get(record['job_id'])
            assert payload is not None
            result.append(payload)
        return result


def stage_key(stage: str, dependencies: dict[str,Any]) -> str:
    """Namespace only by relevant dependencies; alpha never changes candidate scores (§10)."""
    return stage+'-'+digest(dependencies)[:20]
