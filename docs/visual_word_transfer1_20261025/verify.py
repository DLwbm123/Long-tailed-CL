"""Independent saved-pixel full12-map and24-system acceptance; no training."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from core import ARMS,shuffle_receipt
from word_data import word_draw
import numpy_word
import random
from foundation import write,build_backbone
from inference import arrays,tokens,affine
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
    check('arrival_freeze_seals',dict(models=sum(x['models'] for x in stages)),ms['labels']==cfg['order'][:10] and ms['base_updates']==cfg['base_updates'] and ms['all_later_optimization_forbidden'] and len(stages)==6 and ms['at']<=stages[0]['at'] and max(x['at'] for x in stages)<=fit['at']<opened['at'] and fit['models']==24)
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
    discovery=load('discovery.private.pt');groups={}
    for i,r in enumerate(base):groups.setdefault(r['group'],[]).append(i)
    check('discovery_uses_exact_current_base_patients',dict(patients=len(groups)),groups==discovery['groups'] and set(groups)==set(discovery['means']))
    names=sorted(groups);vectors=np.stack([discovery['means'][g].numpy() for g in names]);norm_error=float(abs(np.linalg.norm(vectors,axis=1)-1).max());check('patient_mean_unit_vectors',dict(error=norm_error),norm_error<=1e-10)
    rng=random.Random(cfg['seed']+1);anchors=rng.sample(names,100);coordinates=[]
    for i,anchor in enumerate(anchors):
        distances=np.square(vectors-vectors[names.index(anchor)]).sum(1);order=sorted(range(len(names)),key=lambda j:(float(distances[j]),names[j]))[:64];patients=([anchor]+[names[j] for j in order if names[j]!=anchor])[:64]
        while True:
            coordinate=(rng.randint(8,104),rng.randint(8,104))
            if coordinate not in coordinates:coordinates.append(coordinate);break
        w=discovery['words'][i];check('full_base_independent_nearest64_word',dict(word=i),w==dict(anchor=anchor,patients=patients,top=coordinate[0],left=coordinate[1]))
    endpoint=load('endpoint.private.pt');seed=cfg['seed']+100000+cfg['base_updates'];expected=word_draw(discovery['words'],groups,seed);ids,classes=patient_draw(base,cfg['order'][:10],seed+2);check('actual_paired_draws_and_permutations',dict(word_rows=len(expected)),endpoint['word_receipt']==expected and endpoint['shuffle']==shuffle_receipt(seed+1) and endpoint['supervised_ids']==[base[i]['id'] for i in ids] and endpoint['supervised_labels'].tolist()==[cfg['order'][:10].index(c) for c in classes] and endpoint['word_labels'].tolist()==[r['word'] for r in expected])
    crop_error=0.
    for i,r in enumerate(expected):crop_error=max(crop_error,float(abs(numpy_word.crop(endpoint['word_raw'][i:i+1].numpy(),r['top'],r['left'])-endpoint['word_target'][i:i+1].numpy()).max()))
    shuffled=numpy_word.corrupt(endpoint['word_target'].numpy(),endpoint['shuffle']);shuffle_error=float(abs(shuffled-endpoint['word_input'].numpy()).max());check('independent_endpoint_crop_and_corruption',dict(crop_error=crop_error,shuffle_error=shuffle_error),crop_error<=2e-7 and shuffle_error==0)
    endpoint_error=0.
    for a in ARMS[1:]:
        x,t,y=(endpoint['supervised_raw'],endpoint['supervised_raw'],endpoint['supervised_labels']) if a==ARMS[1] else (endpoint['word_input'],endpoint['word_target'],endpoint['word_labels']);zs=np.concatenate([tokens(x[i:i+1].numpy(),states[a]) for i in range(16)]);h=load(f'endpoint_head_{a}.private.pt');value=numpy_word.objective(zs,t.numpy(),y.numpy(),a,arrays(h['decoder']),arrays(h['word_head']),h['disease_head'].numpy());error=abs(value-endpoint['objectives'][a]);endpoint_error=max(endpoint_error,error);check('actual_post_update_full12_endpoint',dict(arm=a,error=error),error<=cfg['score_absolute_tolerance'])
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
    cost=json.loads((p/'COSTS.json').read_text());check('frozen_update_and_population_counts',cost['arms'],all(cost['arms'][a]['updates']==cfg['base_updates'] and cost['arms'][a]['forward_rows']==16*cfg['base_updates'] and cost['arms'][a]['backward_rows']==16*cfg['base_updates'] and cost['endpoint_rows'][a]==16 for a in ARMS[1:]) and all(cost['descriptor_rows'][a]==68358 and cost['score_rows'][a]==1350 and cost['ridge_solves'][a]==6 for a in ARMS) and cost['validation_encodes']==300 and all(v['updates']==2 for v in cost['qualification_arms'].values()))
    from reporting import report
    result,decision=report(results,cfg,train);write(p/'RESULTS.json',result);write(p/'DECISION.json',decision);write(p/'VERIFICATION.json',dict(state='PASS',checks=len(gates),unique_query_map_rows=1200,proof_rows=40,score_groups=24,score_rows=score_rows,score_error=score_error,prediction_mismatches=mismatch,full_query_feature_error=feature_error,proof_error=proof_error,endpoint_error=endpoint_error,wall_seconds=time.time()-tick,scope='Full12 NumPy maps from saved raw FP32 pixels for query/proof/all endpoints. Full saved patient-mean nearest-cohort recomputation; means independently checked online but discovery images not all independently reencoded. No new source images or optimizer replay.'));write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:write(Path(cfg['output'])/'VERIFICATION_FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
