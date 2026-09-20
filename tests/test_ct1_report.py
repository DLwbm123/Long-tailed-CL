"""Synthetic report coverage test; temporary arrays are not experiment results."""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import report_ct1 as report
from preflight_ct1 import ISIC_ORDERS,HK_ORDERS


def main():
    with tempfile.TemporaryDirectory(prefix='ct1_report_fixture_') as tmp:
        root=Path(tmp);pub=root/'output/public';private=root/'output/private/sealed'
        pub.mkdir(parents=True);private.mkdir(parents=True)
        entries=[];tasklock={};groups={}
        for dataset,orders,sizes in [('HK',HK_ORDERS,[2]*10+[3]),('ISIC',ISIC_ORDERS,[2]*4)]:
            n=sum(sizes);groups[dataset]={'head':list(range(2)),'mid':list(range(2,n-2)),'tail':list(range(n-2,n))}
            tasklock[dataset]={'orders':orders}
            for seed,order in orders.items():
                known=0
                for task,seen in enumerate(np.cumsum(sizes),1):
                    original=np.repeat(sorted(order[:seen]),2);y=np.array([order.index(int(c)) for c in original])
                    ids=np.array([f'{c:02d}_{j}' for c in sorted(order[:seen]) for j in range(2)])
                    for method in report.METHODS:
                        path=private/f'{dataset}_{seed}_{task}_{method}.npz'
                        np.savez(path,raw=np.eye(seen)[y],y=y,original=original,order=np.array(order),ids=ids,component=ids)
                        entries.append(dict(dataset=dataset,seed=seed,task=task,method=method,file=path.name,
                                            sha256=report.sha(path),seen=int(seen),known=int(known)))
                    known=seen
        report.write(pub/'DATA_AND_TASK_LOCK.json',tasklock)
        report.write(pub/'TRAJECTORIES_LOCK.json',dict(checkpoints=[{'synthetic':True}]*90,sealed_units=entries,risk_failures=[]))
        report.write(pub/'RESOURCE_LEDGER.json',dict(GPU_process_residence_seconds=0))
        report.write(pub/'ACCESS_formal.json',dict(formal_optimizer_steps=32340))
        (pub/'train_epoch_metrics.jsonl').write_text(''.join(json.dumps(dict(synthetic=True,index=i))+'\n' for i in range(900)))
        for name in ['transport_audit','risk_solver_audit']:(pub/(name+'.jsonl')).write_text('')
        cfg=root/'cfg.json';report.write(cfg,dict(root=str(root),frequency_groups=groups,source_commit='SYNTHETIC_REPORT_TEST_ONLY'))
        previous=os.environ.get('P20_CONFIG');os.environ['P20_CONFIG']=str(cfg)
        try:report.main()
        finally:
            if previous is None:os.environ.pop('P20_CONFIG')
            else:os.environ['P20_CONFIG']=previous
        audit=report.read(pub/'MEMORY_ACCESS_AND_RESOURCE_AUDIT.json')
        assert audit['main_metric_rows']==135 and audit['all_metric_rows']==270 and audit['all_per_class_rows']==2754
        with (pub/'bootstrap_intervals.csv').open() as f:
            import csv
            for row in csv.DictReader(f):assert float(row['mean_difference_pp'])==float(row['low'])==float(row['high'])==0.
    print('PASS: synthetic coverage, exact perfect metrics, shared bootstrap and zero paired differences')


if __name__=='__main__':main()
