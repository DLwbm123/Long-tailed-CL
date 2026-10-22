"""Independent NumPy mappings, weighted systems, arrivals and sealed scores."""
import os,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch

def mapping(x,ref,kind):
    x=np.asarray(x,dtype=np.float64)
    if kind=='linear':phi=x
    else:
        a=ref['anchors'].numpy();dist=np.maximum(np.sum(x*x,1)[:,None]+np.sum(a*a,1)[None]-2*x@a.T,0)
        phi=np.exp(-dist/float(ref['sigma']))-ref['center'].numpy()
    phi=phi/np.maximum(np.linalg.norm(phi,axis=1,keepdims=True),1e-12)
    return np.concatenate([phi,np.ones((len(phi),1))],1)/np.sqrt(2)

def stats(phi,rows):
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['group'],[]).append(i)
    weights=np.zeros(len(phi))
    for ix in groups.values():weights[ix]=1/(len(groups)*len(ix))
    return phi.T@(weights[:,None]*phi),np.sum(weights[:,None]*phi,axis=0),len(groups)

def np_system(records,arm,lam):
    kind='linear' if arm=='LINEAR_BALANCED' else 'kernel';mass=np.array([r['patients'] for r in records],dtype=np.float64);mass=mass/mass.sum() if arm=='KERNEL_NATURAL' else np.ones(len(records))/len(records)
    gram=sum(w*r[kind]['gram'].numpy() for w,r in zip(mass,records));rhs=np.stack([w*r[kind]['mean'].numpy() for w,r in zip(mass,records)],1)
    return gram+lam*np.eye(len(gram)),rhs

def summarize(stages,cfg,counts):
    mean={a:float(np.mean([s['metrics'][a]['BA_percent'] for s in stages])) for a in cfg['arms']};groups=dict(tail4=[7,14,15,16],tail7=[c for c,n in counts.items() if n<250],initial10=cfg['order'][:10],later10=cfg['order'][10:]);freq={a:{g:sum(stages[-1]['metrics'][a]['correct'][str(c)] for c in cs) for g,cs in groups.items()} for a in cfg['arms']}
    final=lambda a:stages[-1]['metrics'][a]['total_correct'];a=cfg['primary_arm'];g=cfg['gates'];gates=dict(final=final(a)>=g['final_min_correct'],mean=mean[a]>=g['mean_min_percent'],tail7=freq[a]['tail7']>=g['tail7_min'],initial=freq[a]['initial10']>=g['initial10_min'],later=freq[a]['later10']>=g['later10_min'])
    for b in ['LINEAR_BALANCED','KERNEL_NATURAL']:gates.update({b+'_final':final(a)>=final(b)+4,b+'_mean':mean[a]>=mean[b]+1,b+'_tail4':freq[a]['tail4']>=freq[b]['tail4']+2,b+'_later':freq[a]['later10']>=freq[b]['later10']})
    return dict(stages=stages,mean_BA_percent=mean,frequency=freq),dict(gates=gates,proceed=all(gates.values()),no_parameter_rescue=True)

def main(cfg):
    from threadpoolctl import threadpool_limits
    from data import read_rows
    threadpool_limits(4);torch.set_num_threads(4);start=time.time();p=Path(cfg['output']);load=lambda n:torch.load(p/n,map_location='cpu',weights_only=False)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');assert not {r['group'] for r in train}&{r['group'] for r in val}
    ref=load('reference.private.pt');base=[r for r in train if r['label'] in cfg['order'][:10]];pp={}
    for i,r in enumerate(base):pp.setdefault((r['label'],r['group']),[]).append(i)
    rank=lambda g,path='':hashlib.sha256(f"{cfg['seed']}:{g[0]}:{g[1]}:{path}".encode()).digest()
    groups=sorted(pp,key=lambda g:(rank(g),g));gs=[];used=set()
    for c in sorted(set(r['label'] for r in base)):
        g=next(g for g in groups if g[0]==c and g[1] not in used);gs.append(g);used.add(g[1])
    for g in groups:
        if len(gs)>=cfg['anchors']:break
        if g[1] not in used:gs.append(g);used.add(g[1])
    ix=[min(pp[g],key=lambda i:(rank(g,base[i]['relative_path']),i)) for g in gs]

    assert ref['anchor_paths']==[base[i]['relative_path'] for i in ix]
    assert len(ref['anchor_paths'])==cfg['anchors'] and len(set(ref['anchor_groups']))==cfg['anchors']
    aa=ref['anchors'].numpy();dist=np.maximum(np.sum(aa*aa,1)[:,None]+np.sum(aa*aa,1)[None]-2*aa@aa.T,0);tri=dist[np.triu_indices(len(aa),1)];sigma=np.partition(tri,(len(tri)-1)//2)[(len(tri)-1)//2];assert abs(sigma-float(ref['sigma']))<1e-10
    assert np.max(np.abs(np.exp(-dist/float(ref['sigma'])).mean(0)-ref['center'].numpy()))<1e-10
    fit=json.loads((p/'FIT_SEAL.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text());seals=json.loads((p/'STAGE_SEALS.json').read_text());assert len(seals)==6 and fit['models']==18 and max(s['at'] for s in seals)<fit['at']<opened['at']
    query=load('query.private.pt');labels=np.array([r['label'] for r in val]);phi={k:mapping(query['features'].numpy(),ref,k) for k in ['linear','kernel']};records=[];stages=[];score_error=normal_error=probe_error=0.;mismatch=0
    for task in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];old=sum(cfg['task_sizes'][:task]);pack=load(f'stage_{task}.private.pt');assert pack['seen']==seen
        for c in seen[old:]:
            rec=load(f'class_{c}.private.pt');assert rec['label']==c and rec['online_independent']['state']=='PASS';rr=[r for r in train if r['label']==c];assert rec['images']==len(rr) and rec['patients']==len(set(r['group'] for r in rr))
            for kind in ['linear','kernel']:
                pr=rec['map_probe'][kind];assert np.max(np.abs(mapping(pr['features'].numpy(),ref,kind)-pr['mapping'].numpy()))<=cfg['score_absolute_tolerance']
            assert all(abs(float(rec[k]['gram'][-1,-1])-.5)<1e-9 and abs(float(rec[k]['mean'][-1])-1/np.sqrt(2))<1e-9 for k in ['linear','kernel']);records.append(rec)
        mask=np.isin(labels,seen);metrics={}
        for a in cfg['arms']:
            mat,rhs=np_system(records,a,cfg['ridge']);head=np.linalg.solve(mat,rhs);savedhead=pack['heads'][a].numpy();normal_error=max(normal_error,float(abs(mat@savedhead-rhs).max()));kind='linear' if a=='LINEAR_BALANCED' else 'kernel'
            score=phi[kind][mask]@head;saved=load(f'scores_{task}_{a}.private.pt').numpy();score_error=max(score_error,float(abs(score-saved).max()));mismatch+=int(np.sum(score.argmax(1)!=saved.argmax(1)))
            probe=pack['probe'];pr=mapping(probe['features'].numpy(),ref,kind)@head;probe_error=max(probe_error,float(abs(pr-probe['scores'][a].numpy()).max()))
            pred=np.array(seen)[score.argmax(1)];correct={str(c):int(np.sum((pred==c)&(labels[mask]==c))) for c in seen};metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()))
        stages.append(dict(task=task,seen=seen,metrics=metrics))
    assert max(score_error,probe_error)<=cfg['score_absolute_tolerance'] and normal_error<1e-9 and mismatch==0
    cost=json.loads((p/'COSTS.json').read_text());assert cost['science_train_encodes']==68058 and cost['qualification_encodes']==20 and cost['validation_encodes']==300 and cost['formal_solves']==18 and cost['formal_optimizer_updates']==0
    resources=json.loads((p.parent/'guardian_resources.private.json').read_text());assert resources['peak_output_bytes']<cfg['output_budget_bytes'] and resources['RSS_sampled_peak_bytes']<cfg['RSS_budget_bytes']
    result,decision=summarize(stages,cfg,Counter(r['label'] for r in train));verdict=dict(state='PASS',score_groups=18,score_error=score_error,prediction_mismatches=mismatch,normal_equation_error=normal_error,probe_error=probe_error,patient_overlap=0,anchor_metadata_reselected=True,old_statistics_files_read_only=True,arrival_statistics_independently_recomputed_before_release=True,scope='Independent NumPy full class-moment systems and scores; arriving complete descriptors checked online before release. Not re-encoding old images.',new_encodes=0,new_optimization=0,wall_seconds=time.time()-start)
    for name,value in [('RESULTS.json',result),('DECISION.json',decision),('VERIFICATION.json',verdict)]: (p/name).write_text(json.dumps(value,indent=2)+'\n')
    (p/'status.private.json').write_text(json.dumps(dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time())))
if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
