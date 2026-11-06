"""Server launcher for the three riskKD (NARVAL) pairs (DECISIONS 93)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chunkratio.config import validate
from chunkratio.pilot import PAIRS
from chunkratio.riskkd import RISKKD, SCALES, server_config, steps_for

REVISIONS = {'reward_model':'r','preference_dataset':'p','sft_dataset':'d','teacher_base_model':'t','student_base_model':'s'}
EVIDENCE = {'relative_l2_distance':{'reference':1e-3,'base':1e-2},'rule':'nearest'}


@pytest.mark.parametrize('pair',sorted(RISKKD))
@pytest.mark.parametrize('variant',['crpd_raw','crpd_main'])
def test_server_configs_validate_for_every_pair(pair: str, variant: str) -> None:
    """Every pair gives a valid config with its pinned trio, end-of-turn EOS, the assumption with evidence, the
    speed settings and the run's choices; raw may use gamma, pair_sources and train-only collection."""
    spec = RISKKD[pair]
    protocol = json.loads(Path(f"config/crpd/narval/{spec['protocol']}.json").read_text(encoding='utf-8-sig'))
    raw = variant == 'crpd_raw'
    cfg = server_config(pair,variant,REVISIONS,'h','data/train.jsonl','out/run','out/cache',alpha=.03,
                        gamma=.05 if raw else 1.,pair_sources='within' if raw else 'all',steps=150,protocol=protocol,
                        collection_splits=['train'] if raw else None,evidence=EVIDENCE)
    validate(cfg)
    assert {r:(m['path'],m['revision']) for r,m in cfg['models'].items()} == \
           {r:(s['repo'],s['revision']) for r,s in spec['checkpoints'].items()}
    assert cfg['eos_token'] == spec['eos'] and cfg['teacher_dpo_reference_verified'] is False
    assert cfg['teacher_dpo_reference_assumption']['evidence']['relative_l2_distance'] == EVIDENCE['relative_l2_distance']
    assert cfg['shared_prefix_scoring'] and cfg['gpu_sampling'] and cfg['states_per_prompt'] == 2
    assert PAIRS[spec['run_pair']]['tokenizer'] == spec['checkpoints']['teacher_base']['repo']
    if raw:
        assert cfg['teacher_gap_scale'] == .05 and cfg['pair_sources'] == 'within' and cfg['collection_splits'] == ['train']


def test_steps_and_scales() -> None:
    """Steps cover the requested passes; x10/x30 share the 20,480 UltraFeedback rows of the Modal pilots."""
    assert steps_for(7119,128,2.7) == 151 and steps_for(10,128,1.) == 1
    assert SCALES['x10']['uf_train_rows'] == SCALES['x30']['uf_train_rows'] == 20480
    assert SCALES['full']['subset']['train'] is None and SCALES['full']['subset']['select'] == 64


@pytest.mark.parametrize('pair',sorted(RISKKD))
def test_run_pair_dry_run_plans_every_stage(pair: str) -> None:
    """The dry run prints one collection worker per GPU and one training run per gamma, without side effects."""
    out = subprocess.run([sys.executable,'script/server/run_pair.py','--pair',pair,'--scale','x10','--gpus','0,1,2',
                          '--gammas','0.1,0.05','--stages','collect,train,heldout','--dry-run','--root','output/dry-run-test'],
                         capture_output=True,text=True,check=True).stdout
    assert out.count('collect-candidates') == 3 and out.count(' evaluate ') == 2 and out.count('heldout_eval.py') == 2
    assert 'GPU 2:' in out and not Path('output/dry-run-test').exists()
