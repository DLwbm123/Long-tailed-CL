"""Synthetic non-identity label-order regression; CPU only, isolated outputs."""
import sys,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_ct4f import report,sha
from threadpoolctl import threadpool_limits
import csv
with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
    root=Path(tmp)/'new';ref=Path(tmp)/'old';units=[];old=[]
    for r in (root,ref):
        for d in ('public','private/sealed'):(r/'output'/d).mkdir(parents=True)
    for ds in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for task in (1,2,3):
                C=2*task;y=np.repeat(np.arange(C),2);order=np.arange(C)[::-1];ids=np.array([f'{int(order[c])}-{i%2}' for i,c in enumerate(y)])
                good=np.eye(C)[y];bad=good.copy();bad[y==0]=np.eye(C)[1]
                for method in ('F1','P','C0','C2','C3'):
                    r=root if method=='F1' else ref;name=f'{ds}_{seed}_{task}_{method}.npz';path=r/'output/private/sealed'/name
                    np.savez(path,raw=good if method in ('C2','C3') else bad,y=y,order=order,original=order[y],ids=ids,component=ids)
                    e=dict(dataset=ds,seed=seed,task=task,method=method,seen=C,file=name,sha256=sha(path));(units if method=='F1' else old).append(e)
    (root/'output/public/PREDICTIONS_LOCK.json').write_text(json.dumps(dict(units=units)))
    (ref/'output/public/ONLINE_PREDICTIONS_LOCK.json').write_text(json.dumps(dict(units=old)))
    report(root,dict(ct3p_root=str(ref),frequency_groups={ds:dict(head=[0,1],middle=[2,3],tail=[4,5]) for ds in ('HK','ISIC')}))
    rows=list(csv.DictReader((root/'output/public/paired_BA_intervals.csv').open()))
    for x in rows:
        expected=100/(2*int(x['task'])) if x['contrast'] in ('C3-F1','C2-F1') else 0
        assert abs(float(x['difference_pp'])-expected)<1e-8
        assert abs(float(x['low'])-expected)<1e-8 and abs(float(x['high'])-expected)<1e-8
    print('PASS: non-identity label mapping, paired BA and bootstrap, 18/72 report')
