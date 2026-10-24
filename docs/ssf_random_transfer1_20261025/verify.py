"""Independent full native and SSF maps, nonlinear readouts and24packed systems."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from foundation import build_backbone,write
from inference import arrays,tokens
from core import ARMS,numpy_affine,unpack_numpy
from data import read_rows,patient_draw

def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);p=Path(cfg['output']);tick=time.time();checks=[];load=lambda n:torch.load(p/n,map_location='cpu',weights_only=False)
    def check(name,value,passed):
        checks.append(dict(check=name,value=value,PASS=bool(passed)));write(p/'VERIFICATION_PROGRESS.json',checks);assert passed,name;assert time.time()<cfg['hard_deadline_epoch']-60
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');check('patient_isolation',{},len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val});ms=json.loads((p/'MAP_SEAL.json').read_text());stages=json.loads((p/'STAGE_SEALS.json').read_text());fit=json.loads((p/'FIT_SEAL.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text());check('frozen_map_and_arrival_seals',ms,ms['optimizer_steps']==1000 and ms['SSF_frozen'] and ms['RP_frozen'] and ms['trainable_backbone_parameters']==0 and ms['dimensions']==cfg['dimensions'] and len(stages)==6 and sum(x['models'] for x in stages)==24 and ms['at']<=stages[0]['at'] and max(x['at'] for x in stages)<=fit['at']<opened['at'])
    b=build_backbone(cfg,'cpu');state=arrays(b.state_dict());mods=arrays(load('ssf.private.pt'));W=load('projection.private.pt').numpy();query=load('query.private.pt');recorded=load('query_features.private.pt');check('query_labels_and_pixels',dict(rows=len(query['raw'])),query['raw'].dtype==torch.float32 and tuple(query['raw'].shape)==(300,3,224,224) and query['labels'].tolist()==[r['label'] for r in val]);features={a:[] for a in ARMS};feature_error=0.
    for i in range(300):
        maps={'native':tokens(query['raw'][i:i+1].numpy(),state),'ssf':tokens(query['raw'][i:i+1].numpy(),state,mods)}
        for a in ARMS:
            z=maps['ssf' if a.startswith('SSF') else 'native']
            f=numpy_affine(z,a,W);error=float(abs(f-recorded[a][i:i+1].numpy()).max());feature_error=max(feature_error,error);features[a].append(f);check('full12_query_spatial_map',dict(row=i,arm=a,error=error),error<=cfg['full_chain_tolerance'])
    features={a:np.concatenate(v) for a,v in features.items()};proof=load('training_proof.private.pt');counts={c:0 for c in cfg['order']};proof_error=0.
    for identity,v in proof.items():
        check('proof_metadata',dict(id=int(identity)),train[identity]['label']==v['label'] and set(v['features'])==set(ARMS));counts[v['label']]+=1;maps={'native':tokens(v['raw'][None].numpy(),state),'ssf':tokens(v['raw'][None].numpy(),state,mods)}
        for a in ARMS:
            z=maps['ssf' if a.startswith('SSF') else 'native']
            error=float(abs(numpy_affine(z,a,W)-v['features'][a][None].numpy()).max());proof_error=max(proof_error,error);check('full12_training_proof',dict(id=int(identity),arm=a,error=error),error<=cfg['full_chain_tolerance'])
    check('proof_patient_population',dict(rows=len(proof)),len(proof)==40 and all(n==2 for n in counts.values()) and all(len({train[i]['group'] for i,v in proof.items() if v['label']==c})==2 for c in counts))
    results=[];previous={a:None for a in ARMS};score_error=0.;mismatch=0;score_rows=0
    for task in range(6):
        stage=load(f'stage_{task}.private.pt');seen=cfg['order'][:10+2*task];check('arrived_columns',dict(task=task),stage['seen']==seen);take=np.isin(query['labels'].numpy(),seen);labels=query['labels'].numpy()[take];stored=load(f'scores_{task}.private.pt');metrics={}
        for a in ARMS:
            s=stage['systems'][a];d=s['d'];matrix=unpack_numpy(s['matrix_upper'].numpy(),d);rhs=s['rhs'].numpy();h=stage['heads'][a].numpy();residual=float(abs(matrix@h-rhs).max());ind=np.linalg.solve(matrix,rhs);scores=features[a][take]@ind;error=float(abs(scores-stored[a].numpy()).max());score_error=max(score_error,error);pred=np.asarray(seen)[scores.argmax(1)];diff=int(np.sum(pred!=np.asarray(seen)[stored[a].numpy().argmax(1)]));mismatch+=diff;score_rows+=int(take.sum());check('independent_joint_system_and_scores',dict(task=task,arm=a,residual=residual,score_error=error,prediction_mismatch=diff),residual<=1e-9 and error<=cfg['score_absolute_tolerance'] and diff==0)
            gram=(matrix-cfg['ridge']*np.eye(d))*len(seen);prior=np.zeros_like(gram)
            if previous[a] is not None:
                prior,old_rhs,old_n=previous[a];check('old_rhs_fixed',dict(task=task,arm=a),float(abs(rhs[:,:old_n]*len(seen)-old_rhs*old_n).max())<=1e-10)
            arrival=unpack_numpy(s['arrival_upper'].numpy(),d);recurrence=float(abs(gram-prior-arrival).max());check('old_moment_coordinates_fixed',dict(task=task,arm=a,error=recurrence),recurrence<=1e-9);previous[a]=(gram,rhs,len(seen));correct={str(c):int(np.sum((labels==c)&(pred==c))) for c in seen};metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/len(labels))
        results.append(dict(task=task,seen=seen,metrics=metrics))
    events=json.loads((p/'NUMERIC_GATES.json').read_text());cost=json.loads((p/'COSTS.json').read_text());check('all_online_gates',dict(events=len(events)),all(v.get('PASS',True) for v in events));check('native_shared_encoding_and_score_counts',cost,cost['optimizer_steps']==1000 and cost['formal_optimizer']==dict(updates=1000,forward_rows=16000,backward_rows=16000) and cost['SSF_training_native_encodes']==68058 and cost['SSF_validation_encodes']==300 and cost['training_native_encodes']==68058 and cost['validation_encodes']==300 and all(cost['descriptor_rows'][a]==68358 and cost['ridge_solves'][a]==6 and cost['score_rows'][a]==1350 for a in ARMS))
    from inference import cosine_loss
    endpoint=load('endpoint.private.pt');check('endpoint_actual_draw',dict(rows=len(endpoint['raw'])),len(endpoint['raw'])==16 and endpoint['step']==999 and all(train[i]['label'] in cfg['order'][:10] for i in endpoint['source_ids']) and endpoint['labels'].tolist()==[cfg['order'].index(train[i]['label']) for i in endpoint['source_ids']]);losses=[];endpoint_error=0.
    base=[r for r in train if r['label'] in cfg['order'][:10]];ix,lab=patient_draw(base,cfg['order'][:10],cfg['seed']+999);check('final_seeded_patient_image_draw',{},endpoint['source_ids']==[base[i]['id'] for i in ix] and endpoint['labels'].tolist()==[cfg['order'].index(c) for c in lab])
    for i in range(16):
        z=tokens(endpoint['raw'][i:i+1].numpy(),state,mods);err=float(abs(z-endpoint['features'][i:i+1].numpy()).max());endpoint_error=max(endpoint_error,err);losses.append(cosine_loss(z,endpoint['head'].numpy(),endpoint['labels'][i:i+1].numpy()));check('full12_final_postupdate_endpoint',dict(row=i,error=err),err<=cfg['full_chain_tolerance'])
    error=abs(np.mean(losses)-endpoint['objective']);check('final_CE_endpoint_target',dict(error=float(error)),error<=cfg['score_absolute_tolerance'])
    from reporting import report
    result,decision=report(results,cfg,train);write(p/'RESULTS.json',result);write(p/'DECISION.json',decision);write(p/'VERIFICATION.json',dict(state='PASS',checks=len(checks),full_native_reference_rows=340,full_SSF_reference_rows=356,endpoint_error=endpoint_error,query_spatial_map_rows=1200,proof_spatial_map_rows=160,score_groups=24,score_rows=score_rows,feature_error=feature_error,proof_error=proof_error,score_error=score_error,prediction_mismatches=mismatch,wall_seconds=time.time()-tick,scope='340savedrawFP32 pixels each pass native and SSF full12NumPy maps plus16finalSSFdraws.4readouts,24systems;no newsourceimages oroptimization. No fulloptimizertrajectory replay.'));write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:write(Path(cfg['output'])/'VERIFICATION_FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
