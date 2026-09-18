"""Full synthetic prefix matrix and common-component bootstrap equivalence."""
import csv,json,os,sys,tempfile,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from report_ct3p import main

def write(p,x):p.write_text(json.dumps(x))
def run():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);pub=root/'output/public';private=root/'output/private/sealed';pub.mkdir(parents=True);private.mkdir(parents=True)
        cfg=dict(root=str(root),frequency_groups={n:dict(head=[0,1],mid=[2,3],tail=[4,5]) for n in ('HK','ISIC')});write(root/'runtime.json',cfg);os.environ['P22_CONFIG']=str(root/'runtime.json')
        units=[];oracle=[]
        for name in ('HK','ISIC'):
            for seed in (1993,1994,1995):
                order=np.random.default_rng(seed).permutation(6)
                for task in (1,2,3):
                    seen=order[:2*task];original=np.repeat(sorted(seen),2);y=np.array([list(seen).index(c) for c in original]);ids=np.array([f'{c}_{i}' for c in sorted(seen) for i in (0,1)])
                    fields=dict(raw=np.eye(len(seen))[y],y=y,original=original,order=seen,ids=ids,component=ids)
                    for method in ('P','C0','C1','C2','C3'):
                        f=private/f'{name}_{seed}_{method}_{task}.npz';np.savez(f,**fields)
                        units.append(dict(dataset=name,seed=seed,task=task,method=method,seen=2*task,file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
                    if seed==1993 and task==3:
                        for source in ('U','F'):
                            f=private/f'{name}_{source}_Q11.npz';np.savez(f,**fields);oracle.append(dict(dataset=name,source=source,file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        write(pub/'ONLINE_PREDICTIONS_LOCK.json',dict(status='LOCKED',units=units));write(pub/'ORACLE_COMPLETE.json',dict(status='PASS'))
        (pub/'ORACLE_INDEX.jsonl').write_text('\n'.join(json.dumps(x) for x in oracle))
        epochs=[dict(optimizer_steps=55) for _ in range(120)];epochs[0]['optimizer_steps']+=20
        (pub/'TRAIN_EPOCH_METRICS.jsonl').write_text('\n'.join(json.dumps(x) for x in epochs))
        write(pub/'RESOURCE_AND_ACCESS_LEDGER.json',dict(formal_optimizer_steps=6620,formal_task_epochs=120,new_checkpoints=12,engineering_optimizer_steps=0,offline_diagnostic_old_fit_reads=0,GPU_process_residence_seconds=0))
        main()
        for file,n in [('validation_prefix_metrics.csv',90),('validation_prefix_per_class.csv',360),('oracle_prefix_terminal_metrics.csv',4),('oracle_prefix_terminal_per_class.csv',24)]:assert len(list(csv.DictReader((pub/file).open())))==n
        for row in csv.DictReader((pub/'bootstrap_intervals.csv').open()):
            assert all(float(row[k])==0 for k in ('difference_pp','low','high'))
        assert json.loads((pub/'NEXT_DECISION.json').read_text())['action']=='STOP'
    print('PASS: 90/360 online, 4/24 isolated offline, shared component draws, identical contrasts zero, STOP')
if __name__=='__main__':run()
