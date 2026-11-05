"""Budget reservations and validation-only promotion must fail closed."""
import copy

import pytest

from script.overnight_plan import SMOKE_TRIALS, configure_trial, promotion_candidates, reservation_cents, reserve
from chunkratio.config import load_config, validate


def test_budget_reservation_precedes_launch_and_never_overruns():
    ledger={'limit_cents':173,'reserved_cents':0,'jobs':{}}
    reserve(ledger,'first',1200,'A100-80GB')
    assert ledger['reserved_cents']==173
    before=copy.deepcopy(ledger)
    with pytest.raises(ValueError,match='budget'):
        reserve(ledger,'second',1,'L40S')
    assert ledger==before
    with pytest.raises(ValueError,match='twice'):
        reserve(ledger,'first',1200,'A100-80GB')
    assert reservation_cents(1200,'L40S')==115


def test_planned_maximum_has_room_under_thirty_dollars():
    new=sum(reservation_cents(1200,'A100-80GB') for _ in SMOKE_TRIALS)
    new+=2*reservation_cents(2700,'A100-80GB')
    new+=reservation_cents(1800,'A100-80GB')
    new+=3*reservation_cents(1200,'L40S')
    assert new==2351 and new<=2400
    assert 2400+400+75+125==3000  # prior work + controller + contingency
    small=4*reservation_cents(1200,'A100-80GB')+reservation_cents(2700,'A100-80GB')
    small+=reservation_cents(1200,'A100-80GB')+2*reservation_cents(1200,'L40S')
    assert small==1455 and small+45==1500  # smaller plan + bounded CPU controller


def test_promote_only_successful_positive_validation_gain():
    results={
        'negative':{'status':'success','variant':'crpd_main','policy':{'selection_score':1.,'reference_selection_score':2.},'test_score':99},
        'main':{'status':'success','variant':'crpd_main','policy':{'selection_score':3.,'reference_selection_score':2.},'test_score':0},
        'raw':{'status':'success','variant':'crpd_raw','policy':{'selection_score':2.1,'reference_selection_score':2.}},
        'failed':{'status':'timeout','variant':'crpd_raw','test_score':100},
    }
    assert promotion_candidates(results)==['main','raw']
    assert promotion_candidates({'negative':results['negative']})==[]


def test_recipe_retains_identity_precision_and_declares_changes():
    base=load_config('config/crpd/smoke.json')
    original=copy.deepcopy(base)
    for level in ('smoke','medium'):
        for trial in SMOKE_TRIALS:
            cfg=configure_trial(base,level,trial,'output/overnight')
            validate(cfg)
            assert cfg['models']==base['models'] and cfg['collection_decoding']==base['collection_decoding']
            assert cfg['horizon']==base['horizon'] and cfg['chunk_length']==base['chunk_length']
            assert cfg['head_positions']==trial['head_positions']
    assert base==original
