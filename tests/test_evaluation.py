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
    result = report_tables(rows,allow_incomplete=True)
    assert 'crpd_main' in result and '2.0000 / 2.0000' in result
    assert 'Between-seed sample SD' in result
    with pytest.raises(ValueError,match='rows missing'):
        report_tables(rows)
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


def narval_row(method: str, value: float, **extra: object) -> dict:
    """One NARVAL-layout row with every Table 1/2 metric."""
    from chunkratio.evaluation import NARVAL_OPEN_LLM, NARVAL_TABLE2
    metrics = {m:value for m,_ in NARVAL_OPEN_LLM+NARVAL_TABLE2}
    return {'method':method,'pair':'qwen3_1p7b_to_0p6b','metrics':metrics,**extra}


def test_narval_layout_separates_reported_rows() -> None:
    """Reported NARVAL numbers render in their own section, are labelled, and never mix with measured rows."""
    from chunkratio.evaluation import report_narval
    measured = [narval_row('crpd_main',v,seed=s,synthetic=False,split='test') for s,v in ((0,50.),(1,52.))]
    reported = [narval_row('narval',60.,source='NARVAL Table 1',reported=True)]
    text = report_narval(measured,reported)
    assert 'crpd_main uses prompts only' in text and 'excluded from every significance test' in text
    measured_part, reported_part = text.split('### Reported by NARVAL (not measured)')[0], text.split('### Reported by NARVAL (not measured)')[1]
    assert '| crpd_main | 51.00 |' in measured_part and 'narval' not in measured_part.split('## Table 1')[1].split('###')[1]
    assert '| narval | 60.00 |' in reported_part
    synthetic = [narval_row('narval',1.,source='NARVAL Table 1',reported=True,synthetic=True)]
    assert '(synthetic placeholder)' in report_narval([],synthetic,allow_incomplete=True)
    with pytest.raises(ValueError,match='reported=true'):
        report_narval(measured,[narval_row('narval',1.,source='paper')])
    with pytest.raises(ValueError,match='measured NARVAL rows'):
        report_narval(reported,[])
    with pytest.raises(ValueError,match='no measured rows'):
        report_narval([],reported)


def test_narval_plan_fails_closed_and_uses_config_fields() -> None:
    """Few-shot counts, metric names, judges and lm-eval version come only from config (§11.4)."""
    import json
    from pathlib import Path
    from chunkratio.experiment import narval_benchmark_plan
    cfg = json.loads(Path('config/crpd/narval/qwen3_8b_to_1p7b.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError,match='num_fewshot'):
        narval_benchmark_plan(cfg,'model','id','out')
    evaluation = cfg['narval']['evaluation']
    for i,task in enumerate(evaluation['open_llm']['tasks'].values()):
        task.update(task=task['task'] or f'task{i}',num_fewshot=i,metric='acc')
    evaluation['open_llm']['lm_eval_version'] = '0.0.test'
    evaluation['mt_bench'].update(judge='judge-x',judge_version='v1')
    evaluation['hh_rlhf'].update(dataset_revision='r',subset_ids_file='ids.json',subset_sha256='h',judge='j',
                                 judge_version='v',reference_outputs='ref.json',reference_model='ref')
    plan = narval_benchmark_plan(cfg,'model','id','out')
    gsm8k = next(j for j in plan['jobs'] if j['metric'] == 'gsm8k')
    assert gsm8k['argv'][gsm8k['argv'].index('--num_fewshot')+1] == '5' and plan['requires']['lm_eval'] == '0.0.test'
    assert next(j for j in plan['jobs'] if j['metric'] == 'mt_bench')['categories'][-1] == 'humanities'
