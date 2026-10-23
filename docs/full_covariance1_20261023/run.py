import json,time,traceback
from pathlib import Path
import numpy as np
from core import fit,scores,independent_scores,self_check
from common import config,write,records,queries,metadata,metrics,unit

def main(cfg):
    p=Path(cfg['output']);p.mkdir(exist_ok=True)
    cost=dict(new_model_encodes=0,new_optimization=0,qualification_classes=2,formal_class_fits=0,materialized_class_factors=0,score_groups=0,score_rows=0)
    def status(phase,**kw):write(p/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**kw))
    status('qualification')
    assert json.loads((Path(cfg['source_root'])/'VERIFICATION.json').read_text())['state']=='PASS'
    toy=self_check();start=time.time();rr=records(cfg,[13,15]);proof=unit(np.concatenate([r['proof'] for r in rr]))
    model=fit(rr,cfg['relative_ridge']);a=scores(proof,model);b=independent_scores(proof,rr,cfg['relative_ridge'])
    error=max(float(abs(a[k]-b[k]).max()) for k in a)
    assert error<=cfg['score_tolerance'] and all(np.array_equal(a[k].argmax(1),b[k].argmax(1)) for k in a)
    seconds=time.time()-start;remaining=cfg['hard_deadline_epoch']-time.time();forecast=2*90/2*seconds+600
    qual=dict(state='PASS',toy=toy,dimension=1536,training_proof_rows=len(proof),score_error=error,prediction_mismatches=0,two_class_seconds=seconds,forecast_seconds=forecast,remaining_seconds=remaining,new_model_encodes=0,new_optimization=0,source_full_native_acceptance_composed=True)
    write(p/'QUALIFICATION.json',qual)
    assert forecast<remaining-60,'Frozen original-clock budget gate failed'
    del rr,model,a,b,proof
    seals=[]
    for task in range(6):
        seen=cfg['order'][:10+2*task];status('fit_and_seal',task=task)
        rr=records(cfg,seen);model=fit(rr,cfg['relative_ridge']);cost['formal_class_fits']+=len(rr)
        np.savez(p/f'model_{task}.private.npz',L=model['shared_factor'],means=np.stack([r['mean'] for r in rr]),patients=np.array([r['patients'] for r in rr]),seen=np.array(seen),rho=np.array([r['patients']/(r['patients']+1536) for r in rr]))
        seals.append(dict(task=task,seen=seen,at=time.time(),models=3,representation='Sealed shared factor, exact native moments by class reference, patient counts and fixed covariance rule'))
        del model,rr
    write(p/'FIT_SEALS.json',seals);status('all_readouts_sealed')
    write(p/'QUERY_OPEN.private.json',dict(at=time.time()))
    q=queries(cfg);meta=metadata(cfg);y=np.array([r['label'] for r in meta['val']])
    assert q.shape==(300,1536)
    anchor=json.loads((Path(cfg['source_root'])/'RESULTS.json').read_text());stages=[]
    for task in range(6):
        seen=cfg['order'][:10+2*task];status('score_new_readouts',task=task)
        rr=records(cfg,seen);model=fit(rr,cfg['relative_ridge']);sealed=np.load(p/f'model_{task}.private.npz')
        assert np.array_equal(sealed['seen'],seen) and np.array_equal(sealed['L'],model['shared_factor'])
        values=scores(q[np.isin(y,seen)],model);cost['materialized_class_factors']+=len(rr)
        np.savez(p/f'scores_{task}.private.npz',**values)
        mm={a:metrics(s,seen,y) for a,s in values.items()};mm['FROZEN_NATIVE']=anchor['stages'][task]['metrics']['FROZEN_NATIVE']
        assert anchor['stages'][task]['seen']==seen
        stages.append(dict(task=task,seen=seen,metrics=mm));cost['score_groups']+=3;cost['score_rows']+=sum(len(s) for s in values.values())
        del model,rr,values
    write(p/'COSTS.json',cost);write(p/'AGGREGATES_PENDING_VERIFY.json',dict(stages=stages));status('COMPLETE_PENDING_VERIFY')

if __name__=='__main__':
    cfg=config()
    try:main(cfg)
    except Exception:
        write(Path(cfg['output'])/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
