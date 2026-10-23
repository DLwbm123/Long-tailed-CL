import json,time
from pathlib import Path
import numpy as np
from common import config,write,records,queries,metadata,metrics,report
from core import independent_scores

cfg=config();start=time.time();p=Path(cfg['output']);meta=metadata(cfg)
seals=json.loads((p/'FIT_SEALS.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text())
assert len(seals)==6 and max(s['at'] for s in seals)<opened['at']
q=queries(cfg);y=np.array([r['label'] for r in meta['val']]);anchor=json.loads((Path(cfg['source_root'])/'RESULTS.json').read_text())
error=0.;mismatch=0;stages=[];eig_systems=0
for task in range(6):
    assert time.time()<cfg['hard_deadline_epoch']-60
    seen=cfg['order'][:10+2*task];rr=records(cfg,seen)
    for r in rr:
        real=[z for z in meta['train'] if z['label']==r['label']]
        assert r['images']==len(real) and r['patients']==len({z['patient'] for z in real})
    sealed=np.load(p/f'model_{task}.private.npz');assert np.array_equal(sealed['seen'],seen)
    P=sum(r['covariance'] for r in rr)/len(rr);P=(P+P.T)/2+cfg['relative_ridge']*np.trace(P)/1536*np.eye(1536)
    assert np.max(abs(sealed['L']@sealed['L'].T-P))<1e-10
    assert np.array_equal(sealed['means'],np.stack([r['mean'] for r in rr]))
    assert np.array_equal(sealed['patients'],[r['patients'] for r in rr])
    values=independent_scores(q[np.isin(y,seen)],rr,cfg['relative_ridge']);eig_systems+=3*len(rr)
    saved=np.load(p/f'scores_{task}.private.npz');mm={}
    for a,s in values.items():
        error=max(error,float(abs(s-saved[a]).max()));mismatch+=int((s.argmax(1)!=saved[a].argmax(1)).sum());mm[a]=metrics(s,seen,y)
    mm['FROZEN_NATIVE']=anchor['stages'][task]['metrics']['FROZEN_NATIVE'];stages.append(dict(task=task,seen=seen,metrics=mm))
    write(p/'verify_status.private.json',dict(phase='independent_eigensystems',task=task,error=error,mismatches=mismatch,updated=time.time()))
assert error<=cfg['score_tolerance'] and mismatch==0
cost=json.loads((p/'COSTS.json').read_text());assert cost['formal_class_fits']==90 and cost['score_groups']==18 and cost['score_rows']==4050 and cost['new_model_encodes']==cost['new_optimization']==0
result,decision=report(stages,cfg,meta['train']);verdict=dict(state='PASS',score_error=error,prediction_mismatches=0,new_score_groups=18,independent_eigensystems=eig_systems,reused_native_response_rows=300,patient_overlap=0,all_new_heads_before_query=True,new_model_encodes=0,new_optimization=0,wall_seconds=time.time()-start,scope='Independent original-coordinate eigensystems from source native class moments; composed with prior full12-layer native response acceptance, not a new backbone/image replay.')
for n,v in [('RESULTS.json',result),('DECISION.json',decision),('VERIFICATION.json',verdict)]:write(p/n,v)
write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))
