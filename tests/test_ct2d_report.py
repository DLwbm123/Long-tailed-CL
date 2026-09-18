"""Synthetic end-to-end report coverage and paired bootstrap identity check."""
import json,os,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from report_ct2d import main

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def run():
    with tempfile.TemporaryDirectory(prefix='p21-report-') as tmp:
        r=Path(tmp);pub=r/'output/public';private=r/'output/private/scores';pub.mkdir(parents=True);private.mkdir(parents=True)
        old=r/'old/output/private/sealed';old.mkdir(parents=True)
        cfg=dict(root=str(r),ct1_root=str(r/'old'),frequency_groups={})
        fits=[]
        for name,C in [('HK',23),('ISIC',8)]:
            cfg['frequency_groups'][name]=dict(head=list(range(C-2)),mid=[],tail=[C-2,C-1])
            original=np.repeat(np.arange(C),2);ids=np.array([f'{k:03d}' for k in range(len(original))])
            for seed in (1993,1994,1995):
                order=np.random.default_rng(seed).permutation(C);y=np.argsort(order)[original];raw=np.eye(C)[y]
                fields=dict(raw=raw,y=y,order=order,ids=ids,original=original,component=ids)
                np.savez(old/f'{name}_{seed}_PT-CB_t{11 if name=="HK" else 4:02d}.npz',**fields)
                for stream in ('U','R'):
                    for mode in ('Q00','Q10','Q01','Q11')+(('Q11-Risk',) if stream=='U' else ()):
                        np.savez(private/f'{name}_{seed}_{stream}_{mode}.npz',**fields,W=np.zeros((1,C)))
                        fits.append(dict(dataset=name,seed=seed,stream=stream,mode=mode,fit_BA=100,fit_accuracy=100))
        write(r/'runtime.json',cfg);os.environ['P21_CONFIG']=str(r/'runtime.json')
        write(pub/'GPU_PHASE_COMPLETE.json',dict(status='PASS'))
        import hashlib
        write(pub/'SCORES_LOCK.json',dict(formal_metric_units=54,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in private.glob('*.npz')}))
        write(pub/'ARCHIVE_SOURCE_AUDIT.json',dict(cpu_seconds=0))
        (pub/'oracle_fit_audit.jsonl').write_text('\n'.join(json.dumps(x) for x in fits))
        for f in ('geometry','routing','class_pair_distances','synthetic_diagnostics'):(pub/(f+'.jsonl')).write_text('{"synthetic":true}\n')
        (pub/'gradient_probes.jsonl').write_text('\n'.join(json.dumps(dict(model_unchanged=True,synthetic_direct_adapter_gradient_zero=True,optimizer_steps=0)) for _ in range(8)))
        (pub/'restore_audit.jsonl').write_text('{"synthetic":true}\n')
        (pub/'reproduction_audit.jsonl').write_text('\n'.join('{"status":"PASS"}' for _ in range(12)))
        (pub/'risk_counterfactual_audit.jsonl').write_text('')
        write(pub/'RESOURCE_AND_ACCESS_LEDGER.json',dict(diagnostic_old_train_image_reads=0,encoder_forward_calls=0,backward_probe_calls=0,new_analytic_fits=0,GPU_process_residence_seconds=0))
        main()
        import csv
        def rows(f):return list(csv.DictReader((pub/f).open()))
        assert len(rows('oracle_factorial_metrics.csv'))==54 and len(rows('oracle_factorial_per_class.csv'))==837
        assert all(float(x['balanced_accuracy'])==100 for x in rows('oracle_factorial_metrics.csv'))
        assert all(float(x[k])==0 for x in rows('oracle_bootstrap_intervals.csv') for k in ('mean_difference_pp','low','high'))
        assert read_status(pub/'NEXT_DECISION.json')=='STOP'
        print('PASS: synthetic 54/837 coverage, label-order invariance, paired identity bootstrap, STOP')
def read_status(p):return json.loads(p.read_text())['action']
if __name__=='__main__':run()
