"""Independent saved-pixel full12-map and24-system acceptance; no training."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from core import ARMS,mask_receipt
from foundation import write,build_backbone
from inference import arrays,tokens,affine,objective
from data import read_rows,patient_draw

def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);p=Path(cfg['output']);gates=[];tick=time.time()
    load=lambda name:torch.load(p/name,map_location='cpu',weights_only=False)
    def check(name,value,passed):
        gates.append(dict(check=name,value=value,PASS=bool(passed)));write(p/'VERIFICATION_PROGRESS.json',gates)
        assert passed,name
        assert time.time()<cfg['hard_deadline_epoch']-60
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');base=[r for r in train if r['label'] in cfg['order'][:10]]
    check('patient_isolation_and_val_schema',dict(train=len(train),val=len(val)),len(train)==68058 and len(val)==300 and [r['id'] for r in val]==list(range(300)) and not {r['group'] for r in train}&{r['group'] for r in val})
    ms=json.loads((p/'MAP_SEAL.json').read_text());fit=json.loads((p/'FIT_SEAL.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text());stages=json.loads((p/'STAGE_SEALS.json').read_text())
    check('arrival_freeze_seals',dict(models=sum(x['models'] for x in stages)),ms['labels']==cfg['order'][:10] and ms['base_updates']==2000 and ms['all_later_optimization_forbidden'] and len(stages)==6 and ms['at']<=stages[0]['at'] and max(x['at'] for x in stages)<=fit['at']<opened['at'] and fit['models']==24)
    b=build_backbone(cfg,'cpu');states={ARMS[0]:arrays(b.state_dict())};states.update({a:arrays(load(f'map_{a}.private.pt')) for a in ARMS[1:]})
    query=load('query.private.pt');recorded=load('query_features.private.pt');check('saved_query_pixels_and_labels',dict(rows=len(query['raw']),dtype=str(query['raw'].dtype)),tuple(query['raw'].shape)==(300,3,224,224) and query['raw'].dtype==torch.float32 and query['labels'].tolist()==[r['label'] for r in val])
    features={a:[] for a in ARMS};feature_error=0.
    for i in range(300):
        for a in ARMS:
            z=tokens(query['raw'][i:i+1].numpy(),states[a]);f=affine(z);error=float(abs(f-recorded[a][i:i+1].numpy()).max());feature_error=max(feature_error,error);features[a].append(f)
            check('full12_query_map',dict(row=i,arm=a,error=error),error<=cfg['full_chain_tolerance'])
    features={a:np.concatenate(v) for a,v in features.items()}
    proof=load('training_proof.private.pt');counts={c:0 for c in cfg['order']};proof_error=0.
    for identity,v in proof.items():
        check('training_proof_metadata',dict(id=int(identity)),train[identity]['id']==identity and train[identity]['label']==v['label'] and set(v['features'])==set(ARMS));counts[v['label']]+=1
        for a in ARMS:
            f=affine(tokens(v['raw'][None].numpy(),states[a]));error=float(abs(f-v['features'][a][None].numpy()).max());proof_error=max(proof_error,error)
            check('full12_training_proof',dict(id=int(identity),arm=a,error=error),error<=cfg['full_chain_tolerance'])
    check('two_patients_per_class_proof',dict(rows=len(proof),counts=counts),len(proof)==40 and all(n==2 for n in counts.values()) and all(len({train[i]['group'] for i,v in proof.items() if v['label']==c})==2 for c in counts))
    endpoint_error=0.
    for phase in ['A','B']:
        endpoint=load(f'endpoint_{phase}.private.pt');seed=cfg['seed']+100000+(1000 if phase=='A' else 2000);ids,labels=patient_draw(base,cfg['order'][:10],seed);receipt=mask_receipt(seed+1)
        check('actual_final_draw',dict(phase=phase,rows=len(ids)),endpoint['ids']==[base[i]['id'] for i in ids] and endpoint['labels'].tolist()==[cfg['order'][:10].index(c) for c in labels] and endpoint['receipt']==receipt)
        for a in ARMS[1:]:
            # Phase A stores actual post-update tokens because its map is subsequently trained in B.
            # Phase B uses the final sealed full map, independently from saved FP32 pixels.
            zs=endpoint['tokens'][a].numpy() if phase=='A' else np.concatenate([tokens(endpoint['raw'][i:i+1].numpy(),states[a]) for i in range(16)])
            h=load(f'endpoint_head_{phase}_{a}.private.pt');value=objective(zs,endpoint['raw'].numpy(),endpoint['labels'].numpy(),receipt,arrays(h['decoder']),h['head'].numpy(),a,phase);error=abs(value-endpoint['objectives'][a]);endpoint_error=max(endpoint_error,error)
            check('post_update_objective',dict(phase=phase,arm=a,error=error,interface='saved actual tokens' if phase=='A' else 'saved pixels independent full12'),error<=cfg['score_absolute_tolerance'])
    results=[];previous={a:None for a in ARMS};score_error=0.;mismatch=0;score_rows=0
    for task in range(6):
        pack=load(f'stage_{task}.private.pt');seen=cfg['order'][:10+2*task];check('arrived_columns',dict(task=task),pack['seen']==seen);take=np.isin(query['labels'].numpy(),seen);labels=query['labels'].numpy()[take];stored=load(f'scores_{task}.private.pt');metrics={}
        for a in ARMS:
            system=pack['systems'][a];matrix=system['matrix'].numpy();rhs=system['rhs'].numpy();head=pack['heads'][a].numpy();residual=float(abs(matrix@head-rhs).max());independent=np.linalg.solve(matrix,rhs);score=features[a][take]@independent;error=float(abs(score-stored[a].numpy()).max());score_error=max(score_error,error);prediction=np.asarray(seen)[score.argmax(1)];diff=int(np.sum(prediction!=np.asarray(seen)[stored[a].numpy().argmax(1)]));mismatch+=diff;score_rows+=int(take.sum())
            check('independent_system_and_scores',dict(task=task,arm=a,residual=residual,score_error=error,prediction_mismatch=diff),residual<=1e-9 and error<=cfg['score_absolute_tolerance'] and diff==0)
            gram=(matrix-cfg['ridge']*np.eye(1537))*len(seen);prior=np.zeros_like(gram)
            if previous[a] is not None:
                prior,old_rhs,old_n=previous[a];check('immutable_old_rhs',dict(task=task,arm=a),float(abs(rhs[:,:old_n]*len(seen)-old_rhs*old_n).max())<=1e-10)
            arrival=np.zeros_like(gram);tri=np.triu_indices(1537);arrival[tri]=system['arrived_upper_triangle'].numpy();arrival+=np.triu(arrival,1).T;recurrence=float(abs(gram-prior-arrival).max());check('immutable_old_moments',dict(task=task,arm=a,error=recurrence),recurrence<=1e-9);previous[a]=(gram,rhs,len(seen))
            correct={str(c):int(np.sum((labels==c)&(prediction==c))) for c in seen};metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/len(labels))
        results.append(dict(task=task,seen=seen,metrics=metrics))
    events=json.loads((p/'NUMERIC_GATES.json').read_text());check('all_online_numeric_gates',dict(count=len(events)),all(v.get('PASS',True) for v in events))
    cost=json.loads((p/'COSTS.json').read_text());check('frozen_update_and_population_counts',cost['arms'],all(cost['arms'][a]['updates']==2000 and cost['arms'][a]['forward_rows']==32000 and cost['arms'][a]['backward_rows']==32000 and cost['endpoint_rows'][a]==32 for a in ARMS[1:]) and all(cost['descriptor_rows'][a]==68358 and cost['score_rows'][a]==1350 and cost['ridge_solves'][a]==6 for a in ARMS) and cost['validation_encodes']==300 and all(v['updates']==2 for v in cost['qualification_arms'].values()))
    from reporting import report
    result,decision=report(results,cfg,train);write(p/'RESULTS.json',result);write(p/'DECISION.json',decision);write(p/'VERIFICATION.json',dict(state='PASS',checks=len(gates),unique_query_map_rows=1200,proof_rows=40,score_groups=24,score_rows=score_rows,score_error=score_error,prediction_mismatches=mismatch,full_query_feature_error=feature_error,proof_error=proof_error,endpoint_error=endpoint_error,wall_seconds=time.time()-tick,scope='Full12 NumPy maps from saved raw FP32 pixels for query/proof/final B endpoint; A objective from saved actual post-update tokens. No new source images or optimizer replay.'));write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:write(Path(cfg['output'])/'VERIFICATION_FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
