"""Dependency-independent delivery checks, avoiding silent skipped tensor coverage."""
import ast
import json
from pathlib import Path

from test_reference import ROOT


def test_production_function_contracts() -> None:
    """Every production function cites its role and has type annotations (§quality bar)."""
    for path in (ROOT/'chunkratio').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                assert ast.get_docstring(node),(path,node.name,'docstring')
                assert node.returns is not None,(path,node.name,'return annotation')
                for arg in (*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs):
                    if arg.arg not in ('self','cls'):
                        assert arg.annotation is not None,(path,node.name,arg.arg)


def test_protocol_constants_and_launchers() -> None:
    """Required hyperparameters and portable launchers are delivered (§§5,11)."""
    cfg = json.loads((ROOT/'config/crpd/protocol.json').read_text())
    assert (cfg['horizon'],cfg['chunk_length'],cfg['candidates']) == (512,32,4)
    preparation = cfg['preparation']
    assert (preparation['sft_epochs'],preparation['sft_learning_rate'],preparation['sft_batch_size']) == (3,2e-5,128)
    assert (preparation['teacher_dpo_beta'],preparation['teacher_dpo_learning_rate'],preparation['teacher_dpo_batch_size'],preparation['teacher_dpo_epochs']) == (.05,5e-7,128,2)
    for name in ('smoke_test','train_crpd','run_crpd_protocol'):
        for suffix in ('.ps1','.sh'):
            assert (ROOT/'script'/(name+suffix)).exists()
