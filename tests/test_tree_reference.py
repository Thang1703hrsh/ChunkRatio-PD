"""FP64 port driven by the same exhaustive reference identities (§12)."""
import json
from pathlib import Path

import pytest

pytest.importorskip('torch',reason='FP64 port requires PyTorch')
from chunkratio.tree import TensorWorld
from test_reference import ROOT, compare_nested, load_reference


def test_fp64_tree_identities_regimes_and_routing() -> None:
    """Both phi, partition/Q/Jensen/bounds/regimes match all supplied fixtures (§12)."""
    reference = load_reference()
    reference.World = TensorWorld
    results = {}
    reference.check_identities(results)
    reference.check_routing(results)
    reference.check_regimes(results)
    fixtures = json.loads((ROOT/'reference/ChunkRatio_Math_Checks.json').read_text())
    for key in results:
        compare_nested(results[key],fixtures[key],1e-10)
    assert max(results['identity_max_errors'].values()) <= 1e-10


def test_variable_eos_action_mass() -> None:
    """Early EOS and horizon action probabilities sum to 1 for every rule (§12)."""
    import random
    reference = load_reference()
    world = TensorWorld(*reference.make_world(random.Random(4),.3,.8))
    assert (2,) in world.completions(())
    for state in reference.PREFIXES:
        for rule in reference.RULES.values():
            chunks = world.chunks(state,rule)
            mass = sum(world.logp(world.rho,state,c).exp().item() for c in chunks)
            assert abs(mass-1) < 1e-12
