"""Independent full native NumPy attention and patient-balanced normal equations."""
import os,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

def np_stats(phi,rows):
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['group'],[]).append(i)
    w=np.zeros(len(phi))
    for ix in groups.values():w[ix]=1/(len(groups)*len(ix))
    return phi.T@(w[:,None]*phi),np.sum(w[:,None]*phi,0),len(groups)
def np_affine(z):
    z=np.asarray(z,dtype=np.float64);z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    return np.concatenate([z,np.ones((len(z),1))],1)/np.sqrt(2)
def np_draws(rows,classes,seed,steps):
    import hashlib
    groups={c:{'S':{},'Q':{}} for c in classes};left=set();right=set()
    for i,r in enumerate(rows):
        digest=hashlib.sha256(f"{seed}:patient_crossfit:{r['group']}".encode()).digest();side='S' if int(digest[0])<128 else 'Q';groups[r['label']][side].setdefault(r['group'],[]).append(i)
        (left if side=='S' else right).add(r['group'])
    assert not left&right and all(v['S'] and v['Q'] for v in groups.values());rng=np.random.default_rng(seed);result=[]
    for _ in range(steps):
        cs=[int(c) for c in rng.choice(classes,8,replace=False)];sides={'S':[],'Q':[]}
        for c in cs:
            n=min(4,len(groups[c]['S']),len(groups[c]['Q']))
            for side in ['S','Q']:
                pp=list(groups[c][side])
                for j in rng.choice(len(pp),n,replace=False):sides[side].append(int(rng.choice(groups[c][side][pp[int(j)]])))
        assert not {rows[i]['group'] for i in sides['S']}&{rows[i]['group'] for i in sides['Q']}
        ix=sides['S']+sides['Q'];assert len(ix)==(62 if 15 in cs else 64);result.append(ix+[-1]*(64-len(ix)))
    return np.array(result,dtype=np.int32)
def report(stages,cfg,counts):
    arms=cfg['arms'];mean={a:float(np.mean([s['metrics'][a]['BA_percent'] for s in stages])) for a in arms};groups=dict(tail4=[7,14,15,16],tail7=[c for c,n in counts.items() if n<250],initial10=cfg['order'][:10],later10=cfg['order'][10:]);freq={a:{g:sum(stages[-1]['metrics'][a]['correct'][str(c)] for c in cs) for g,cs in groups.items()} for a in arms};final={a:stages[-1]['metrics'][a]['total_correct'] for a in arms};a=cfg['primary_arm'];g=cfg['gates'];gates=dict(final=final[a]>=g['final_min_correct'],mean=mean[a]>=g['mean_min_percent'],tail7=freq[a]['tail7']>=g['tail7_min'],initial=freq[a]['initial10']>=g['initial10_min'],later=freq[a]['later10']>=g['later10_min'])
    for b in arms[:-1]:gates.update({b+'_final':final[a]>=final[b]+4,b+'_mean':mean[a]>=mean[b]+1,b+'_tail4':freq[a]['tail4']>=freq[b]['tail4']+2,b+'_later':freq[a]['later10']>=freq[b]['later10']})
    return dict(stages=stages,mean_BA_percent=mean,frequency=freq),dict(gates=gates,proceed=all(gates.values()),no_parameter_rescue=True)
def main(cfg):
    from threadpoolctl import threadpool_limits
    from data import read_rows
    from core import ARMS,np_evidence
    from run import build_backbone,body_state,write
    threadpool_limits(4);torch.set_num_threads(4);start=time.time();p=Path(cfg['output']);load=lambda n:torch.load(p/n,map_location='cpu',weights_only=False)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');assert not {r['group'] for r in train}&{r['group'] for r in val};base=[r for r in train if r['label'] in cfg['order'][:10]]
    assert np.array_equal(np_draws(base,cfg['order'][:10],cfg['seed'],cfg['base_updates']),load('paired_draws.private.pt').numpy())
    fit=json.loads((p/'FIT_SEAL.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text());seals=json.loads((p/'STAGE_SEALS.json').read_text());mapseal=json.loads((p/'MAP_SEAL.json').read_text());assert len(seals)==6 and fit['models']==24 and mapseal['at']<min(s['at'] for s in seals) and max(s['at'] for s in seals)<fit['at']<opened['at']
    b=build_backbone(cfg,'cpu').double();s=body_state(b);q=load('query_tokens.private.pt').numpy();savedfeatures=load('query_features.private.pt');features={};feature_error=0.;probe_error=0.;normal_error=0.;score_error=0.;mismatch=0;np_rows=0;proof_rows=0
    initial=load('initial_prompts.private.pt');assert initial.shape==(8,6,768)
    for a in ARMS:
        prompt=load(f'prompts_{a}.private.pt');assert (prompt is None)==(a==ARMS[0])
        if prompt is not None:assert prompt.shape==initial.shape and not torch.equal(prompt,initial)
        vals=[]
        for i in range(0,len(q),cfg['microbatch']):
            assert time.time()<cfg['hard_deadline_epoch']-60
            vals.append(np_evidence(q[i:i+cfg['microbatch']],s,None if prompt is None else prompt.numpy()));np_rows+=len(q[i:i+cfg['microbatch']])
        features[a]=np.concatenate(vals);feature_error=max(feature_error,float(abs(features[a]-savedfeatures[a].numpy()).max()));assert feature_error<cfg['score_absolute_tolerance']
    records=[];stages=[];labels=np.array([r['label'] for r in val])
    for task in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];old=sum(cfg['task_sizes'][:task]);pack=load(f'stage_{task}.private.pt');assert pack['seen']==seen
        for c in seen[old:]:
            r=load(f'class_{c}.private.pt');rr=[x for x in train if x['label']==c];assert r['label']==c and r['images']==len(rr) and r['patients']==len(set(x['group'] for x in rr))
            for a in ARMS:
                rec=r[a];assert rec['online_independent_error']<1e-9 and abs(float(rec['gram'][-1,-1])-.5)<1e-9
                prompt=load(f'prompts_{a}.private.pt');z=np_evidence(rec['proof_tokens'].numpy(),s,None if prompt is None else prompt.numpy());proof_rows+=len(z);probe_error=max(probe_error,float(abs(z-rec['proof_features'].numpy()).max()))
            records.append(r)
        mask=np.isin(labels,seen);metrics={}
        for a in ARMS:
            mat=sum(r[a]['gram'].numpy() for r in records)/len(records)+cfg['ridge']*np.eye(1537);rhs=np.stack([r[a]['mean'].numpy()/len(records) for r in records],1);head=np.linalg.solve(mat,rhs);normal_error=max(normal_error,float(abs(mat@pack['heads'][a].numpy()-rhs).max()));score=np_affine(features[a][mask])@head;saved=load(f'scores_{task}_{a}.private.pt').numpy();score_error=max(score_error,float(abs(score-saved).max()));mismatch+=int((score.argmax(1)!=saved.argmax(1)).sum());pred=np.array(seen)[score.argmax(1)];correct={str(c):int(((pred==c)&(labels[mask]==c)).sum()) for c in seen};metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()))
        stages.append(dict(task=task,seen=seen,metrics=metrics))
    assert max(feature_error,score_error)<=cfg['score_absolute_tolerance'] and probe_error<cfg['FP32_probe_tolerance'] and normal_error<1e-9 and mismatch==0
    cost=json.loads((p/'COSTS.json').read_text());assert cost['science_prefix_rows']==68058 and cost['qualification_prefix_rows']==78 and cost['validation_prefix_rows']==300 and cost['formal_solves']==24
    actual_rows=int((load('paired_draws.private.pt').numpy()>=0).sum())
    for a in ARMS[1:]:
        assert all(cost['arms'][a][k]==actual_rows for k in ['logical_rows','cached_body_forward_rows','recomputed_body_forward_rows','body_backward_rows'])
        assert cost['arms'][a]['inner_dual_solves']==(2*cfg['base_updates'] if a.startswith('RIDGE') else 0)
    assert cost['arms'][ARMS[0]]['updates']==0 and all(cost['arms'][a]['updates']==cfg['base_updates'] for a in ARMS[1:])
    assert all(cost['arms'][a]['mapping_rows']==68058 and cost['arms'][a]['deploy_query_rows']==300 for a in ARMS)
    resources=json.loads((p.parent/'guardian_resources.private.json').read_text());assert resources['peak_output_bytes']<cfg['output_budget_bytes'] and resources['RSS_sampled_peak_bytes']<cfg['RSS_budget_bytes']
    assert json.loads((p/'QUALIFICATION.json').read_text())['state']=='PASS' and json.loads((p/'FEASIBILITY.json').read_text())['state']=='PASS'
    result,decision=report(stages,cfg,Counter(r['label'] for r in train));decision['mechanism_OUT_over_IN']=all(v for k,v in decision['gates'].items() if k.startswith('RIDGE_IN_'));verdict=dict(state='PASS',full_native_NumPy_query_rows=np_rows,full_native_NumPy_classproof_rows=proof_rows,score_groups=24,score_error=score_error,feature_error=feature_error,FP32_classproof_error=probe_error,normal_equation_error=normal_error,prediction_mismatches=mismatch,patient_overlap=0,paired_base_draws_independently_reconstructed=True,current_complete_patient_moments_checked_online=True,scope='Independent full12-layer NumPy mapping, saved arrival proofs and all normal systems. Not a replay of every optimization trajectory or old raw image.',new_encodes=0,new_optimization=0,wall_seconds=time.time()-start)
    for n,v in [('RESULTS.json',result),('DECISION.json',decision),('VERIFICATION.json',verdict)]:write(p/n,v)
    write(p/'status.private.json',dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time()))
if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
