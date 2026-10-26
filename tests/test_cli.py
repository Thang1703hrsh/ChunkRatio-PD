"""CLI remains inspectable without optional training dependencies (§10)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chunkratio.config import VARIANTS, load_config, validate
from chunkratio.data import helpsteer_pairs


def test_config_variants_and_strict_protocol() -> None:
    """Every variant is selectable, unresolved real model identities fail closed (§9)."""
    cfg = load_config('config/crpd/smoke.json')
    for variant in VARIANTS:
        validate({**cfg,'variant':variant})
    with pytest.raises(ValueError,match='path and revision'):
        load_config('config/crpd/protocol.json')
    with pytest.raises(ValueError,match='raw categorical'):
        validate({**cfg,'collection_decoding':{'temperature':.7}})


def test_module_and_additive_training_cli() -> None:
    """CRPD opt-in dispatch works without importing legacy Unix-only modules (§3)."""
    for command in ([sys.executable,'-m','chunkratio','--help'],[sys.executable,'train.py','--crpd','--help']):
        result = subprocess.run(command,capture_output=True,text=True)
        assert result.returncode == 0 and 'collect-correction' in result.stdout


def test_helpsteer_ties_removed() -> None:
    """Helpfulness pairs prepared once; main student then consumes prompts only (§11)."""
    rows = [{'prompt':'p','response':str(i),'helpfulness':score} for i,score in enumerate([2,2,4])]
    pairs = helpsteer_pairs(rows)
    assert len(pairs) == 2 and all(p['chosen'] == '2' for p in pairs)
