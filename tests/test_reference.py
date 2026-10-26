"""Executable supplied reference and provenance; no tensor dependency (§§1,12)."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_reference():
    """Load the exact supplied executable fixture (§1)."""
    spec = importlib.util.spec_from_file_location('crpd_reference', ROOT/'reference/ChunkRatio_Math_Checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_nested(actual: object, expected: object, tol: float = 1e-10) -> None:
    """Compare every numeric fixture within FP64 tolerance (§12)."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            compare_nested(actual[key],expected[key],tol)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected):
            compare_nested(a,b,tol)
    elif isinstance(expected,(int,float)):
        assert abs(float(actual)-expected) <= tol
    else:
        assert actual == expected


def test_reference_all_checks(tmp_path: Path) -> None:
    """Run original before scaling; compare its JSON without overwriting source (§12)."""
    path = tmp_path/'reference.json'
    result = subprocess.run([sys.executable,str(ROOT/'reference/ChunkRatio_Math_Checks.py'),str(path)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert 'ALL CHECKS PASSED' in result.stdout
    compare_nested(json.loads(path.read_text()),json.loads((ROOT/'reference/ChunkRatio_Math_Checks.json').read_text()))


def test_source_hashes() -> None:
    """Persist immutable document and reference identities (§1)."""
    for row in (ROOT/'reference/HASHES.txt').read_text().splitlines():
        digest,path = row.split('  ',1)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest


def test_agent_instructions_identical() -> None:
    """Keep both requested reviewer entry points consistent (§3)."""
    assert (ROOT/'AGENTS.md').read_bytes() == (ROOT/'CLAUDE.md').read_bytes()
