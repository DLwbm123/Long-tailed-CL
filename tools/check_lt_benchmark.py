"""Small independent checks for the schedule, accounting and report assembly."""
import json
from pathlib import Path
import tempfile

from lt_benchmark import check
from run_lt_benchmark import recovery_plan
from summarize_lt_benchmark import summarize


def main():
    check()
    c=dict(gpu_seconds_limit=1000,jobs=[dict(id='A',cap_seconds=100),dict(id='B',cap_seconds=100)])
    records,pending,done=recovery_plan(c,None)
    assert not records and [j['attempt_id'] for j in pending]==['A','B']
    old=dict(status='INCOMPLETE',gpu_seconds_limit=1000,jobs=[
        dict(id='A',logical_id='A',status='COMPLETE',exit_code=0,elapsed_seconds=80),
        dict(id='B',logical_id='B',status='INCOMPLETE',exit_code=1,elapsed_seconds=50)])
    try:
        recovery_plan(c,old)
        raise AssertionError('Unreviewed restart accepted')
    except ValueError: pass
    records,pending,done=recovery_plan(dict(c,recovery_note='diagnosed and fixed fixture'),old)
    assert done=={'A':'A'} and pending[0]['attempt_id']=='B_repair1'
    assert sum(r['elapsed_seconds'] for r in records)==130
    unlimited=dict(c,gpu_seconds_limit=None,jobs=[dict(id='A',cap_seconds=None)])
    assert len(recovery_plan(unlimited,None)[1])==1
    try:
        recovery_plan(dict(c,gpu_seconds_limit=2000,recovery_note='x'),old)
        raise AssertionError('Budget reset accepted')
    except ValueError: pass
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'public').mkdir();jobs=[]
        for dataset in ('ISIC','HK','CIFAR100LT'):
            for order in ('ordered','shuffled'):
                for arm in ('cf_linear','cf_prototype','cf_weighted','cf_group'):
                    name=f'{dataset}_{order}_{arm}';p=root/name;p.mkdir();jobs.append(dict(id=name))
                    (p/'STATUS.json').write_text(json.dumps(dict(status='COMPLETE',steps=7)))
                    (p/'metrics.json').write_text(json.dumps(dict(average_incremental_accuracy=.7,final_accuracy=.6,
                        final_balanced_accuracy=.5,task_forgetting=.1)))
                    (p/'diagnostics.json').write_text(json.dumps([dict(controller=dict(optimizer_steps=8,accepted=False),
                        activation=dict(weight_l1_change=0,prototype_metric_relative_change=.1))]))
        summarize(root,jobs)
        result=json.loads((root/'public/RESULTS.json').read_text())
        assert len(result['records'])==24 and len(result['paired_comparisons'])==18
        assert result['records']['HK_ordered_cf_group']['activation']['weighting_epochs']==0
    print('PASS: recovery keeps failures/costs, skips completion, blocks budget resets; complete 24-cell report')


if __name__=='__main__': main()
