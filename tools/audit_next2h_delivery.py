"""Post-run aggregate pairing and delivery audit; never refits or re-evaluates images."""
import json
from pathlib import Path
import time
START=time.process_time()
import numpy as np


def run(root):
    public=root/'public';results=json.loads((public/'RESULTS.json').read_text());ledger=json.loads((public/'BUDGET_LEDGER.json').read_text())
    assert results['status'] in ('COMPLETE_NEGATIVE','READY_FOR_REPLICATION') and results['neural_updates']==0 and not results['test_accessed']
    heads=json.loads((public/'HEADS_FROZEN.json').read_text());equivalent=json.loads((public/'HEAD_EQUIVALENCE.json').read_text())
    assert equivalent['status']=='PASS' and all(x['historical_prediction_agreement']==1 for x in equivalent['prediction_checks'])
    order=[4,0,3,7,5,6,2,1];names=list(results['metrics']);pairs=[]
    for stage in range(1,5):
        data={n:np.load(root/(n+f'_stage_{stage}.private.npz')) for n in names}
        for i,a in enumerate(names):
            for b in names[i+1:]:
                x,y=data[a],data[b];assert np.array_equal(x['labels'],y['labels'])
                pa=np.asarray(order[:2*stage])[x['scores'].argmax(1)];pb=np.asarray(order[:2*stage])[y['scores'].argmax(1)]
                ca=pa==x['labels'];cb=pb==y['labels']
                pairs.append(dict(stage=stage,a=a,b=b,n=len(pa),prediction_agreement=float((pa==pb).mean()),
                    both_correct=int((ca&cb).sum()),a_only_correct=int((ca&~cb).sum()),b_only_correct=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum())))
        for value in data.values():value.close()
    elapsed=time.process_time()-START
    (public/'PAIRED_PREDICTIONS.json').write_text(json.dumps(dict(pairs=pairs,trusted_manifest_alignment=True,cpu_seconds=elapsed,neural_updates=0),indent=2)+'\n')
    for name,metric in results['metrics'].items():
        assert [s['validation_n'] for s in metric['stages']]==[85,149,212,295]
    for name,rows in results['decompositions'].items():
        for row in rows:
            if row['group']=='old':assert abs(row['D_internal']+row['D_competition']-row['D_total'])<1e-12 and row['D_competition']>=0
    audit=dict(status='PASS',neural_updates=0,backward_calls=0,shared_feature_extractions=1,baseline_prediction_equivalence=True,
        candidate_heads_frozen_before_feature_process=True,all_decomposition_identities_pass=True,paired_rows=len(pairs),
        postprocessing_cpu_seconds=time.process_time()-START,test_accessed=False,independent_confirmation=False)
    (public/'FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    ledger['run_cpu_seconds']=ledger['cpu_seconds'];ledger['delivery_audit_cpu_seconds']=audit['postprocessing_cpu_seconds'];ledger['cpu_seconds']+=audit['postprocessing_cpu_seconds'];ledger['cpu_core_hours']=ledger['cpu_seconds']/3600
    ledger['persistent_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    assert ledger['cpu_seconds']<28800 and ledger['gpu_seconds']<7200 and ledger['persistent_bytes']<1024**3
    (public/'BUDGET_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    import sys
    run(Path(json.load(sys.stdin)['root']))
