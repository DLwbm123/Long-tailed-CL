"""Bounded read-only acceptance of two sealed stages after frozen projection solver stopped.
No images, encoder, optimizer, query/validation scoring or solver retry.
"""
import json,os,time,hashlib,resource
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from verify import numpy_forward,numpy_risks,nstate,ce
from data import read_rows

def main(cfg):
    start=time.time();threadpool_limits(4);torch.set_num_threads(4)
    root=Path(cfg['output']);assert time.time()<cfg['hard_deadline_epoch']-60
    failure=json.loads((root/'FAILURE.private.json').read_text());assert 'projection solver qualification failed' in failure['error']
    assert not any((root/n).exists() for n in ['query.private.pt','FIT_SEAL.json','QUERY_OPEN.private.json','RESULTS.json','DECISION.json','VERIFICATION.json'])
    cost=json.loads((root/'COSTS.private.json').read_text());assert cost['validation_encodes']==0 and cost['science_train_encodes']==64466
    seals=json.loads((root/'STAGE_SEALS.json').read_text());assert [x['task'] for x in seals]==[0,1]
    train=read_rows(cfg,'train');val=read_rows(cfg,'val');assert not {r['group'] for r in train}&{r['group'] for r in val}
    previous={};prior=None;probe_error=0.;risk_error=0.;transition_error=0.;numpy_rows=0;details=[]
    for t in range(2):
        p=torch.load(root/f'stage_{t}.private.pt',map_location='cpu',weights_only=False)
        seen=cfg['order'][:sum(cfg['task_sizes'][:t+1])];assert p['seen']==seen==seals[t]['seen']
        assert seals[t]['science_train_encodes']==sum(r['label'] in seen for r in train)
        expected=[]
        for c in seen:
            patients={}
            for i,r in enumerate(train):
                if r['label']==c:patients.setdefault(r['group'],[]).append(i)
            rank=lambda g,path='':hashlib.sha256(f"{cfg['seed']}:{c}:{g}:{path}".encode()).digest()
            for g in sorted(patients,key=lambda g:(rank(g),g))[:cfg['memory_per_class']]:
                expected.append(min(patients[g],key=lambda i:(rank(g,train[i]['relative_path']),i)))
        assert sorted(r['id'] for r in p['memory_rows'])==sorted(expected)
        assert len(p['memory_rows'])==len({(r['label'],r['group']) for r in p['memory_rows']})
        for row,vec in zip(p['memory_rows'],p['memory_features']):
            assert row['relative_path']==train[row['id']]['relative_path']
            if row['id'] in previous:assert np.array_equal(previous[row['id']],vec.numpy())
            previous[row['id']]=vec.numpy().copy()
        arms={};oldn=len(prior['memory_rows']) if prior else 0
        mx=p['memory_features'][:oldn].numpy();my=np.array([seen.index(r['label']) for r in p['memory_rows'][:oldn]])
        for a in cfg['arms']:
            assert seals[t]['opportunities'][a]==2000+t*500
            s=nstate(p['models'][a]);probe=p['probe']
            objective=float(ce(numpy_forward(probe['x'].numpy(),s,cfg['logit_scale'])[1],probe['y'].numpy()).mean());numpy_rows+=len(probe['y'])
            probe_error=max(probe_error,abs(objective-probe['CE'][a]))
            arms[a]=dict(saved_training_probe_CE=objective,accepted_updates=seals[t]['accepted_updates'][a])
            if not t:
                assert all(torch.equal(p['models'][a][k],p['models']['ER'][k]) for k in s)
            else:
                pre=numpy_risks(mx,my,nstate(prior['models'][a]),cfg['logit_scale'])
                admitted=numpy_risks(mx,my,nstate(p['arrival_models'][a]),cfg['logit_scale'])
                end=numpy_risks(mx,my,s,cfg['logit_scale']);numpy_rows+=3*oldn
                ceiling=np.array(p['ceilings'][a]);risk_error=max(risk_error,float(abs(end-np.array(p['final_old_risks'][a])).max()))
                assert np.max(abs(ceiling-pre-cfg['stage_CE_allowance']))<cfg['risk_verification_tolerance']
                assert np.max(admitted-pre)<=cfg['expansion_CE_allowance']+cfg['risk_verification_tolerance']
                margin=float(np.max(end-ceiling))
                if a=='FEASIBLE':assert margin<=cfg['risk_verification_tolerance']
                arms[a].update(max_arrival_class_CE_increase=float(np.max(admitted-pre)),max_final_CE_minus_fixed_ceiling=margin,old_class_CE_before=pre.tolist(),old_class_CE_after=end.tolist())
        if t:
            tr=p['transition_probe'];bs=nstate(tr['before']);after=nstate(tr['after']);direction=tr['direction'].numpy().astype(np.float64)
            bv=np.concatenate([bs[k].ravel() for k in tr['parameter_order']]);av=np.concatenate([after[k].ravel() for k in tr['parameter_order']]);transition_error=float(abs(av-(bv-cfg['learning_rate']*tr['factor']*direction)).max());assert transition_error<2e-6
            G=tr['constraint_gradients'].numpy().astype(np.float64);H=G/np.maximum(np.linalg.norm(G,axis=1,keepdims=True),1e-12)
            assert np.min(H@direction)>=-cfg['projection_direction_tolerance']*2
            rr=numpy_risks(mx,my,after,cfg['logit_scale']);numpy_rows+=oldn
            assert np.all(rr<=np.array(p['ceilings']['FEASIBLE'])+cfg['risk_verification_tolerance'])
        details.append(dict(task=t,seen_classes=len(seen),memory_rows=len(p['memory_rows']),arms=arms));prior=p
    assert probe_error<=cfg['score_absolute_tolerance'] and risk_error<=cfg['risk_verification_tolerance']
    receipt=dict(state='PASS_PARTIAL_FAILURE_RECORD_ONLY',scope='Two existing sealed stages; not six-stage or 18-score scientific verification. Frozen solver failure retained; no failed QP reconstruction or retry.',sealed_stages=2,training_probe_objective_max_error=probe_error,class_risk_max_error=risk_error,parameter_transition_max_error=transition_error,metadata_memory_selection=True,old_features_unchanged=True,patient_overlap=0,first_stage_three_states_identical=True,stage1_FEASIBLE_risk_ceiling_verified=True,stage1_real_transition_and_first_order_direction_verified=True,numpy_model_forward_rows=numpy_rows,new_encodes=0,optimization=0,validation_scoring_rows=0,scientific_metrics_available=False,details=details,seconds=time.time()-start,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (root/'PARTIAL_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='details'}))
if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
