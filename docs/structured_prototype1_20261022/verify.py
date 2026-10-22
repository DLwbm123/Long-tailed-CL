"""Independent NumPy evidence map, energy/objective and sealed endpoint acceptance."""
import json,os,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch


def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def softmax(x):
    e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
def logsum(x):
    m=x.max(-1);return m+np.log(np.exp(x-m[...,None]).sum(-1))
def arrays(s):return {k:v.detach().cpu().numpy().astype(np.float64) for k,v in s.items()}
def numpy_map(tokens,s,cfg):
    raw=norm(np.asarray(tokens,dtype=np.float64));h=norm(raw@s['project.weight'].T+s['project.bias'])
    assignment=softmax(h[:,1:]@norm(s['queries']).T/cfg['assignment_temperature'])
    attention=assignment/np.maximum(assignment.sum(1,keepdims=True),1e-12)
    local=norm(attention.transpose(0,2,1)@h[:,1:])
    return np.concatenate([h[:,0],local.reshape(len(h),-1)],1)
def numpy_score(x,s,records,arm,cfg):
    glob=cfg['logit_scale']*norm(x)@norm(s['head']).T
    if arm=='GLOBAL':return glob
    d=cfg['part_dimension'];z=x[:,d:].reshape(-1,cfg['parts'],d)
    mu=np.stack([r['mean'] for r in records]);v=np.stack([r['calibrated' if arm=='CALIBRATED' else 'variance'] for r in records]);w=np.stack([r['weights'] for r in records])
    gaussian=-.5*(((z[:,None,None]-mu[None])**2/v[None])+np.log(2*np.pi*v)[None]).sum(-1)
    cm=(mu*w[:,:,None,None]).sum(1)
    attention=softmax((norm(z)[:,None]*norm(cm)[None]).sum(-1)/cfg['class_attention_temperature'])
    energy=cfg['logit_scale']/d*logsum((gaussian*attention[:,:,None]).sum(-1)+np.log(np.maximum(w,1e-30))[None])
    weights=softmax(s['mixing']);return weights[0]*glob+weights[1]*energy

def numpy_loss(x,y,s,records,arm,cfg):
    z=numpy_score(x,s,records,arm,cfg);ce=np.mean(logsum(z)-z[np.arange(len(y)),y])
    h=norm(s['head']);center=norm(np.stack([r['global_mean'] for r in records]));k=len(h)
    gram=h@h.T;mask=~np.eye(k,dtype=bool)
    return ce+cfg['center_alignment_weight']*(1-(h*center).sum(1)).mean()+cfg['class_geometry_weight']*((gram[mask]+1/(k-1))**2).mean()

def independent_fit(features,rows,classes,prior,cfg):
    # Separate implementation run while the actual arriving patient descriptors still exist.
    records=[];proof=[];d=cfg['part_dimension'];m=cfg['parts']
    for c in classes:
        groups={}
        for i,r in enumerate(rows):
            if r['label']==c:groups.setdefault(r['group'],[]).append(i)
        x=np.array([np.mean(features[ix],axis=0,dtype=np.float64) for ix in groups.values()])
        centers=[x[0].copy()]
        for j in range(1,min(cfg['modes'],len(x))):
            distance=np.min(np.array([np.sum((x-a)**2,axis=1) for a in centers]),axis=0);centers.append(x[np.argmax(distance)].copy())
        centers=np.array(centers)
        for _ in range(cfg['kmeans_iterations']):
            membership=np.argmin(np.sum((x[:,None]-centers[None])**2,axis=2),axis=1)
            centers=np.array([np.mean(x[membership==j],axis=0) if np.any(membership==j) else centers[j] for j in range(len(centers))])
        membership=np.argmin(np.sum((x[:,None]-centers[None])**2,axis=2),axis=1)
        parts=x[:,d:].reshape(-1,m,d);means=[];var=[];cal=[];counts=[];sums=[];squares=[]
        for j in range(cfg['modes']):
            a=parts[membership==j];n=len(a);counts.append(n)
            mu=np.mean(a,axis=0) if n else parts.mean(0);v=np.var(a,axis=0,ddof=1) if n>1 else np.zeros((m,d));rho=n/(n+cfg['prior_patient_strength'])
            means.append(mu);var.append(np.maximum(v,cfg['variance_floor']));cal.append(np.maximum(rho*v+(1-rho)*prior,cfg['variance_floor']))
            sums.append(a.sum(0));squares.append((a*a).sum(0))
        records.append(dict(label=c,mean=np.array(means),variance=np.array(var),calibrated=np.array(cal),weights=np.array(counts)/len(x),counts=np.array(counts),patients=len(x),global_mean=x.mean(0)))
        proof.append(dict(label=c,counts=np.array(counts),sums=np.array(sums),squares=np.array(squares),patient_local_mean=parts.mean(0)))
    return records,proof

def summarize(stages,cfg,counts):
    mean={a:float(np.mean([s['metrics'][a]['BA_percent'] for s in stages])) for a in cfg['arms']}
    groups=dict(tail4=[7,14,15,16],tail7=[c for c,n in counts.items() if n<250],initial10=cfg['order'][:10],later10=cfg['order'][10:])
    f={a:{g:sum(stages[-1]['metrics'][a]['correct'][str(c)] for c in cs) for g,cs in groups.items()} for a in cfg['arms']}
    final=lambda a:stages[-1]['metrics'][a]['total_correct'];a=cfg['primary_arm'];g=cfg['gates']
    gates=dict(final=final(a)>=g['final_min_correct'],mean=mean[a]>=g['mean_min_percent'],tail7=f[a]['tail7']>=g['tail7_min'],initial=f[a]['initial10']>=g['initial10_min'],later=f[a]['later10']>=g['later10_min'])
    for b in ['GLOBAL','UNSHRUNK']:
        gates.update({b+'_final':final(a)>=final(b)+4,b+'_mean':mean[a]>=mean[b]+1,b+'_tail4':f[a]['tail4']>=f[b]['tail4']+2,b+'_later':f[a]['later10']>=f[b]['later10']})
    return dict(stages=stages,mean_BA_percent=mean,frequency=f),dict(gates=gates,proceed=all(gates.values()),no_parameter_rescue=True)

def main(cfg):
    from threadpoolctl import threadpool_limits
    from data import read_rows
    threadpool_limits(4);torch.set_num_threads(4);start=time.time();p=Path(cfg['output'])
    tr,va=read_rows(cfg,'train'),read_rows(cfg,'val');assert not {r['group'] for r in tr}&{r['group'] for r in va}
    load=lambda name:torch.load(p/name,map_location='cpu',weights_only=False)
    fit=json.loads((p/'FIT_SEAL.json').read_text());opened=json.loads((p/'QUERY_OPEN.private.json').read_text());seals=json.loads((p/'STAGE_SEALS.json').read_text())
    assert len(seals)==6 and fit['models']==18 and max(s['at'] for s in seals)<fit['at']<opened['at']
    q=load('query.private.pt');mapping=load('evidence_map.private.pt');ms=arrays(mapping)
    nq=np.concatenate([numpy_map(q['tokens'][i:i+8].numpy(),ms,cfg) for i in range(0,len(va),8)])
    map_error=float(abs(nq-q['features'].numpy()).max());assert map_error<=cfg['score_absolute_tolerance']
    labels=np.array([r['label'] for r in va]);oldfeatures={};oldrecords={};score_error=probe_error=moment_error=0.;mismatch=0;stages=[]
    for t in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        pack=load(f'stage_{t}.private.pt');seen=cfg['order'][:sum(cfg['task_sizes'][:t+1])];assert seen==pack['seen']==seals[t]['seen']
        assert all(torch.equal(mapping[k],pack['mapping'][k]) for k in mapping)
        # Independent metadata patient/image reselection, not the runner helper.
        expected=[]
        for c in seen:
            pp={}
            for i,r in enumerate(tr):
                if r['label']==c:pp.setdefault(r['group'],[]).append(i)
            def key(g,path=''):return hashlib.sha256(f"{cfg['seed']}:{c}:{g}:{path}".encode()).digest()
            for g in sorted(pp,key=lambda g:(key(g),g))[:cfg['memory_per_class']]:expected.append(min(pp[g],key=lambda i:(key(g,tr[i]['relative_path']),i)))
        assert sorted(expected)==sorted(r['id'] for r in pack['memory_rows'])
        for r,x in zip(pack['memory_rows'],pack['memory']):
            i=r['id'];assert r['group']==tr[i]['group'] and r['label']==tr[i]['label']
            if i in oldfeatures:assert np.array_equal(oldfeatures[i],x.numpy())
            oldfeatures[i]=x.numpy().copy()
        assert pack['online_independent_stats']['state']=='PASS'
        for rec,proof in zip(pack['records'],pack['proofs']):
            c=rec['label']
            if c in oldrecords:
                assert all(np.array_equal(v,rec[k]) for k,v in oldrecords[c].items() if isinstance(v,np.ndarray))
            oldrecords[c]=rec
            for j,n in enumerate(proof['counts']):
                mu=proof['sums'][j]/n if n else proof['patient_local_mean'];v=(proof['squares'][j]-proof['sums'][j]**2/n)/(n-1) if n>1 else np.zeros_like(mu)
                rho=n/(n+cfg['prior_patient_strength'])
                moment_error=max(moment_error,float(abs(mu-rec['mean'][j]).max()),float(abs(np.maximum(v,cfg['variance_floor'])-rec['variance'][j]).max()),float(abs(np.maximum(rho*v+(1-rho)*pack['prior'],cfg['variance_floor'])-rec['calibrated'][j]).max()))
        mask=np.isin(labels,seen);metrics={};probe=pack['probe']
        for a in cfg['arms']:
            s=arrays(pack['models'][a]);score=numpy_score(nq[mask],s,pack['records'],a,cfg);saved=load(f'scores_{t}_{a}.private.pt').numpy()
            score_error=max(score_error,float(abs(score-saved).max()));mismatch+=int(np.sum(score.argmax(1)!=saved.argmax(1)))
            probe_error=max(probe_error,abs(numpy_loss(probe['x'].numpy(),probe['y'].numpy(),s,pack['records'],a,cfg)-probe['losses'][a]))
            pred=np.array(seen)[score.argmax(1)];correct={str(c):int(np.sum((pred==c)&(labels[mask]==c))) for c in seen}
            metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()))
        stages.append(dict(task=t,seen=seen,metrics=metrics))
    assert max(score_error,probe_error,moment_error)<=cfg['score_absolute_tolerance'] and mismatch==0
    cost=json.loads((p/'COSTS.json').read_text());assert cost['science_train_encodes']==68058 and cost['validation_encodes']==300 and cost['qualification_encodes']==20
    assert cost['primitive_updates']==cfg['primitive_updates'] and all(v==4500 for v in cost['head_updates'].values())
    assert all(v==288000 for v in cost['head_training_rows'].values()) and cost['mapping_frozen_after_base'] and cost['memory_rows_final']==615
    resources=json.loads((p.parent/'guardian_resources.private.json').read_text());assert resources['peak_output_bytes']<cfg['output_budget_bytes'] and resources['RSS_sampled_peak_bytes']<cfg['RSS_budget_bytes']
    results,decision=summarize(stages,cfg,Counter(r['label'] for r in tr))
    verdict=dict(state='PASS',score_groups=18,map_error=map_error,score_error=score_error,prediction_mismatch=mismatch,probe_objective_error=probe_error,moment_reconstruction_error=moment_error,old_memory_and_statistics_immutable=True,patient_overlap=0,memory_reselected=True,mapping_frozen=True,online_stats_recomputed_from_arriving_patient_descriptors=True,stats_scope='Independent NumPy at arrival before releasing current full descriptors; final moments from retained sums/squares. Not all historical patients retained or full trajectory replay.',new_encodes=0,new_optimization=0,wall_seconds=time.time()-start)
    for name,value in [('RESULTS.json',results),('DECISION.json',decision),('VERIFICATION.json',verdict)]: (p/name).write_text(json.dumps(value,indent=2)+'\n')
    (p/'status.private.json').write_text(json.dumps(dict(state='COMPLETE_VERIFIED',gate=decision['proceed'],updated=time.time())))

if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
