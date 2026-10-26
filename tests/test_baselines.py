"""Unchanged TBPO sources and opt-in-only dispatch (§§3,12)."""
import ast
import subprocess
from pathlib import Path

from test_reference import ROOT


def baseline(path: str) -> str:
    """Read immutable starting revision without modifying refs (§3)."""
    return subprocess.check_output(['git','show','tbpo-baseline:'+path],cwd=ROOT).decode().replace('\r\n','\n')


def test_baseline_sources_unchanged() -> None:
    """Exact preservation of TBPO trainers, heads, scores, data and loss configs (§3)."""
    paths = ['trainers.py','utils.py','preference_datasets.py','baseline_head.py',
             'loss/loss.py','loss/loss_utils.py','loss/h_function.py',
             'config/config.yaml','config/loss/Q_tbpo.yaml','config/loss/A_tbpo.yaml']
    for path in paths:
        assert (ROOT/path).read_text(encoding='utf-8') == baseline(path)


def test_training_dispatch_is_additive() -> None:
    """Default entry point has identical legacy suffix (§3)."""
    current = (ROOT/'train.py').read_text(encoding='utf-8')
    assert current[current.index('import torch'):] == baseline('train.py')
    first = ast.parse(current).body[0]
    assert isinstance(first,ast.If)
    assert '"--crpd"' in current[:current.index('import torch')]
