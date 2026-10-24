"""Automatic independent full native-chain, frozen-map and 24-system acceptance."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from foundation import write,build_backbone
from training import ARMS
from inference import body_state,numpy_affine
from native_reference import np_tokens
from numpy_objective import objective
from data import read_rows,patient_draw
from geometry import view_receipt


def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);p=Path(cfg['output']);gates=[];tick=time.time()
    load=lambda name:torch.load(p/name,map_location='cpu',weights_only=False)
    def check(name,value,passed):
        gates.append(dict(check=name,value=value,PASS=bool(passed)));write(p/'VERIFICATION_PROGRESS.json',gates)
        assert passed,name
        assert time.time()<cfg['hard_deadline_epoch']-60
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    check('patient_isolation',dict(train=len(train),val=len(val)),len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val})
    for i,r in enumerate(train):r['id']=i
    base=[r for r in train if r['label'] in cfg['order'][:10]]
    map_seal=json.loads((p/'MAP_SEAL.json').read_text());fit=json.loads((p/'FIT_SEAL.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text());stages=json.loads((p/'STAGE_SEALS.json').read_text())
    check('arrival_freeze_and_seal_order',dict(maps=map_seal['maps'],stage_models=sum(x['models'] for x in stages)),map_seal['labels']==cfg['order'][:10] and map_seal['at']<=stages[0]['at'] and len(stages)==6 and max(x['at'] for x in stages)<=fit['at']<opened['at'] and fit['models']==24)
    backbone=build_backbone(cfg,'cpu');lower_state=body_state(backbone,0,8,False);bodies={'NATIVE_RIDGE':body_state(backbone,8,12)}
    for a in ARMS[1:]:bodies[a]={k:v.numpy() for k,v in load(f'map_{a}.private.pt').items()}
    query=load('query.private.pt');recorded=load('query_features.private.pt');query_features={a:[] for a in ARMS};lower_rows=0;feature_error=0.
    for i in range(len(query['lower'])):
        lower=np_tokens(query['embedding'][i:i+1].numpy(),lower_state,apply_norm=False)
        half=query['lower'][i:i+1].numpy();ulp=np.maximum(np.abs(np.spacing(half)),np.float16(2**-24)).astype(np.float64)
        ratio=float((abs(lower-half)/(.5*ulp+cfg['full_chain_tolerance'])).max())
        check('query_full_lower8_quantized_interface',dict(row=i,bound_ratio=ratio),ratio<=1);lower_rows+=1
        for a in ARMS:
            feature=numpy_affine(half,bodies[a]);query_features[a].append(feature)
            feature_error=max(feature_error,float(abs(feature-recorded[a][i:i+1].numpy()).max()))
    check('all_four_query_maps',dict(max_error=feature_error,rows=1200),feature_error<=cfg['score_absolute_tolerance'])
    query_features={a:np.concatenate(v) for a,v in query_features.items()}
    proof=load('training_proof.private.pt');proof_error=0.;proof_counts={c:0 for c in cfg['order']}
    for identity,value in proof.items():
        lower=np_tokens(value['embedding'][None].numpy(),lower_state,apply_norm=False)
        proof_error=max(proof_error,float(abs(lower-value['lower_FP64'][None].numpy()).max()));lower_rows+=1;proof_counts[value['label']]+=1
        for a,features in value['features'].items():
            error=float(abs(numpy_affine(value['lower'][None].numpy(),bodies[a])-features[None].numpy()).max())
            check('training_map_proof',dict(id=int(identity),arm=a,error=error),error<=cfg['score_absolute_tolerance'])
    check('full_training_lower_proof',dict(rows=len(proof),error=proof_error),len(proof)==40 and all(n==2 for n in proof_counts.values()) and proof_error<=cfg['full_chain_tolerance'])
    probe=load('training_endpoint.private.pt');ids,labels=patient_draw(base,cfg['order'][:10],cfg['seed']+100000+cfg['base_updates']);expected_ids=[base[i]['id'] for i in ids];receipt=view_receipt(cfg['seed']+100001+cfg['base_updates'],32)
    check('last_actual_draw_and_correspondence',dict(rows=len(probe['ids'])),probe['ids']==expected_ids and probe['receipt']==receipt and probe['labels'].tolist()==[cfg['order'][:10].index(c) for c in labels])
    endpoint_error=0.
    for a in ARMS[1:]:
        raw=np.concatenate([np_tokens(probe['tokens'][i:i+1].numpy(),bodies[a]) for i in range(64)])
        head=load(f'probe_head_{a}.private.pt');proj={k:v.numpy() for k,v in head['projectors'].items()}
        value=objective(raw,probe['labels'].numpy(),probe['receipt'],proj,head['head'].numpy(),a);error=abs(value-probe['post_update_objectives'][a]);endpoint_error=max(endpoint_error,error)
        check('actual_training_batch_post_update_objective',dict(arm=a,error=error),error<=cfg['score_absolute_tolerance'])
    score_error=0.;mismatch=0;score_rows=0;results=[];previous={a:None for a in ARMS}
    for task in range(6):
        pack=load(f'stage_{task}.private.pt');seen=cfg['order'][:10+2*task];check('arrived_system_columns',dict(task=task),pack['seen']==seen)
        take=np.isin(query['labels'].numpy(),seen);actual_labels=query['labels'].numpy()[take];stored=load(f'scores_{task}.private.pt');metrics={}
        for a in ARMS:
            system=pack['systems'][a];matrix=system['matrix'].numpy();rhs=system['rhs'].numpy();head=pack['heads'][a].numpy();residual=float(abs(matrix@head-rhs).max())
            independent=np.linalg.solve(matrix,rhs);error=float(abs(query_features[a][take]@independent-stored[a].numpy()).max());score_error=max(score_error,error)
            predictions=np.asarray(seen)[(query_features[a][take]@independent).argmax(1)];difference=int(np.sum(predictions!=np.asarray(seen)[stored[a].numpy().argmax(1)]));mismatch+=difference;score_rows+=int(take.sum())
            check('independent_allseen_normal_system_and_scores',dict(task=task,arm=a,residual=residual,score_error=error,prediction_mismatch=difference),residual<=1e-9 and error<=cfg['score_absolute_tolerance'] and difference==0)
            gram=(matrix-cfg['ridge']*np.eye(1537))*len(seen)
            if previous[a] is None:prior=np.zeros_like(gram)
            else:
                prior,old_rhs,old_count=previous[a];check('old_class_rhs_unchanged',dict(task=task,arm=a),np.max(abs(rhs[:,:old_count]*len(seen)-old_rhs*old_count))<=1e-10)
            arrival=np.zeros_like(gram);tri=np.triu_indices(1537);arrival[tri]=system['arrived_upper_triangle'].numpy();arrival=arrival+np.triu(arrival,1).T
            recurrence=float(abs(gram-prior-arrival).max());check('immutable_old_moment_coordinates',dict(task=task,arm=a,error=recurrence),recurrence<=1e-9);previous[a]=(gram,rhs,len(seen))
            correct={str(c):int(np.sum((actual_labels==c)&(predictions==c))) for c in seen};total=sum(correct.values());metrics[a]=dict(correct=correct,total_correct=total,BA_percent=100*total/len(actual_labels))
        results.append(dict(task=task,seen=seen,metrics=metrics))
    events=json.loads((p/'NUMERIC_GATES.json').read_text());check('all_online_numeric_checks',dict(events=len(events)),all(g.get('PASS',True) for g in events))
    cost=json.loads((p/'COSTS.json').read_text());check('complete_declared_update_and_score_counts',cost['arms'],all(cost['arms'][a]['updates']==2000 and cost['arms'][a]['first_forward_rows']==128000 and cost['arms'][a]['replay_forward_rows']==128000 and cost['arms'][a]['backward_rows']==128000 and cost['score_rows'][a]==1350 for a in ARMS[1:]) and cost.get('successful_validation_encodes',cost['validation_encodes'])==300)
    if (p/'RESUME_VALIDATION_RECEIPT.json').exists():
        resumed=json.loads((p/'RESUME_VALIDATION_RECEIPT.json').read_text())
        check('authorized_validation_resume_no_training',resumed,resumed['training_counters_unchanged'] and resumed['sealed_files_unchanged'] and resumed['additional_optimizer_updates']==0 and resumed['successful_validation_encodes']==300 and cost['validation_encodes']==resumed['prior_partial_validation_encodes']+300)
        checks=json.loads((p/'RESUME_NUMERIC_GATES.json').read_text())
        check('resume_numeric_checks',dict(events=len(checks)),all(v['PASS'] for v in checks))
    from reporting import report
    result,decision=report(results,cfg,train)
    write(p/'RESULTS.json',result);write(p/'DECISION.json',decision);write(p/'VERIFICATION.json',dict(state='PASS',checks=len(gates),full_lower_rows=lower_rows,unique_query_map_rows=1200,score_groups=24,score_rows=score_rows,score_error=score_error,prediction_mismatches=mismatch,training_lower_proof_error=proof_error,endpoint_objective_error=endpoint_error,wall_seconds=time.time()-tick,scope='Independent saved FP16 numeric interfaces, all frozen maps, full query and actual post-update training-batch endpoint. Online patient moments checked before release; no new image encoding or full optimizer trajectory replay.'))
    write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        write(Path(cfg['output'])/'VERIFICATION_FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
