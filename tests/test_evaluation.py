"""Reporting refuses invented results or protocol mixtures (§11)."""
import pytest

from chunkratio.evaluation import METRICS, paired_interval, report_tables
from chunkratio.experiment import benchmark_plan, estimate_cost


def test_paired_prompt_interval() -> None:
    """Identical prompt pairs have constant bootstrap differences (§11.4)."""
    result = paired_interval({'a':2,'b':3},{'a':1,'b':2})
    assert result['low'] == result['high'] == result['mean_difference'] == 1
    with pytest.raises(ValueError):
        paired_interval({'a':1},{'b':1})


def test_report_validation_and_main_table() -> None:
    """Synthetic, mixed protocol and missing three-seed results are rejected (§11)."""
    rows = [dict(method='crpd_main',dataset=dataset,seed=seed,synthetic=False,split='test',
                 metrics={m:float(seed+1) for m in METRICS},protocols={m:'fixed' for m in METRICS},cost={'gpu_hours':2.})
            for dataset in ('DPO-MIX-7K','HelpSteer2') for seed in range(3)]
    result = report_tables(rows)
    assert 'crpd_main' in result and '2.0000 / 2.0000' in result
    rows[0]['synthetic'] = True
    with pytest.raises(ValueError,match='synthetic'):
        report_tables(rows)
    rows[0]['synthetic'] = False
    rows[0]['protocols']['mt_bench'] = 'different judge'
    with pytest.raises(ValueError,match='incomparable'):
        report_tables(rows)


def test_benchmark_plan_and_conditional_cost() -> None:
    """Official versus common-judge configurations and missing hardware are explicit (§11)."""
    plan = benchmark_plan('weights/student','crpd','gpt-4o-mini','outputs/adpa.json','output/eval')
    assert '--mode' in plan['jobs'][2]['argv'] and 'single' in plan['jobs'][2]['argv']
    assert 'truthfulqa_mc2' in ','.join(plan['jobs'][0]['argv'])
    assert estimate_cost(100,200,128)['estimated_gpu_hours'] is None
    assert estimate_cost(100,200,128,100,500)['estimated_gpu_hours'] > 0
