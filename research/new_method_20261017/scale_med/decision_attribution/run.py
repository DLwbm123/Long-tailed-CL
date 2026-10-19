"""Describe fixed argmax outcomes without fitting, forwarding, or rescoring models."""
import json,os,resource,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

ARMS=['G','R','P','C','S']

def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)

def outcomes(birth,current,y,old):
    B=birth.argmax(1)==y;U=current[:,:old].argmax(1)==y;F=current.argmax(1)==y
    assert bool((F<=U).all())
    z=np.stack([B,U,F]).astype(np.int64)
    op=z[1]-z[0];comp=z[2]-z[1];total=z[2]-z[0]
    assert np.array_equal(total,op+comp) and (comp<=0).all()
    return z

def selfcheck():
    b=np.array([[2,1],[2,1],[0,1],[0,1]])
    f=np.array([[0,2,3],[2,1,3],[2,1,0],[2,1,3]])
    z=outcomes(b,f,np.zeros(4,dtype=int),2)
    assert np.array_equal(z.sum(1),[2,3,1])
    assert (z[1]-z[0]).mean()==.25 and (z[2]-z[1]).mean()==-.5
    assert np.array_equal(outcomes(np.zeros((3,2)),np.zeros((3,3)),np.zeros(3,dtype=int),2),np.ones((3,3)))

def main(cfg):
    t=time.monotonic();torch.set_num_threads(1);selfcheck()
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    def timed_out(*_):raise TimeoutError('sealed wall budget')
    signal.signal(signal.SIGALRM,timed_out);signal.alarm(170)
    root=Path(cfg['output']);old=Path(cfg['reference']).resolve()
    allowed={old/'predictions.private.pt',old/'MODEL_SEAL.json',old/'results.json'}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(old):assert p in allowed,'Only sealed predictions and aggregate references allowed'
            assert not any(s in p.name for s in ['features.private','heads.private','transforms.private']), 'No feature or head access'
    sys.addaudithook(guard)
    seal=json.loads((old/'MODEL_SEAL.json').read_text());assert seal['source_commit']=='19bf801' and seal['common_alpha']==.01
    legacy=json.loads((old/'results.json').read_text());obj=torch.load(old/'predictions.private.pt',weights_only=False,map_location='cpu')
    y=obj['y'].numpy();groups=obj['groups'];assert len(y)==len(groups)==586
    assert [int((y==c).sum()) for c in range(6)]==[100,100,100,86,100,100]
    scores={};baseline=[];parity=[]
    for st in range(3):
        for a in ARMS:
            v=obj['scores'][str(st)][a]['0.01'].numpy();assert v.shape==(586,4+st) and np.isfinite(v).all()
            scores[st,a]=v
            for c in range(st+4):
                n=int((y==c).sum());correct=int((v[y==c].argmax(1)==c).sum())
                ref=legacy['metrics'][str(st)][a]['0.01'];assert correct==ref['correct'][c] and n==ref['counts'][c]
                baseline.append(dict(stage=st,arm=a,class_index=c,count=n,correct=correct,recall_pp=100*correct/n));parity.append(correct==ref['correct'][c])
    rng=np.random.default_rng(74003);samples={};points={};rows=[]
    for c in range(6):
        ids=np.where(y==c)[0];keys=sorted({groups[i] for i in ids});members=[np.array([j for j,i in enumerate(ids) if groups[i]==g]) for g in keys]
        sizes=np.array([len(m) for m in members]);draw=rng.integers(0,len(keys),size=(10000,len(keys)));den=sizes[draw].sum(1)
        for st in range(2):
            oldc=st+4
            if c>=oldc:continue
            for a in ARMS:
                z=outcomes(scores[st,a][ids],scores[st+1,a][ids],y[ids],oldc)
                hits=np.stack([z[:,m].sum(1) for m in members],1)
                boot=100*hits[:,draw].sum(2)/den;point=100*z.mean(1)
                assert np.max(np.abs((boot[2]-boot[0])-(boot[1]-boot[0]+boot[2]-boot[1])))<1e-12
                samples[st,a,c]=boot;points[st,a,c]=point
                B,U,F=z.astype(bool);event={
                    'old_set_rescue':int((~B&U).sum()),'old_set_harm':int((B&~U).sum()),
                    'new_capture_previously_correct':int((B&U&~F).sum()),'new_capture_just_rescued':int((~B&U&~F).sum())}
                oldpred=scores[st+1,a][ids,:oldc].argmax(1);fullpred=scores[st+1,a][ids].argmax(1)
                row=dict(transition=[st,st+1],arm=a,class_index=c,count=len(ids),groups=len(keys),birth_correct=int(B.sum()),current_old_correct=int(U.sum()),current_full_correct=int(F.sum()),old_ranking_net=int(U.sum())-int(B.sum()),new_competition_net=int(F.sum())-int(U.sum()),total_net=int(F.sum())-int(B.sum()),events=event,old_set_errors_by_winner=[int(((oldpred==j)&(oldpred!=c)).sum()) for j in range(oldc)],full_errors_by_winner=[int(((fullpred==j)&(fullpred!=c)).sum()) for j in range(oldc+1)])
                for name,p,bs in [('old_ranking',point[1]-point[0],boot[1]-boot[0]),('new_competition',point[2]-point[1],boot[2]-boot[1]),('total',point[2]-point[0],boot[2]-boot[0])]:row[name+'_pp']=float(p);row[name+'_CI95_pp']=np.quantile(bs,[.025,.975]).tolist()
                rows.append(row)
    contrasts=[]
    for control in ['G','S']:
        cp=points[1,'C',4];gp=points[1,control,4];cb=samples[1,'C',4];gb=samples[1,control,4]
        birth=cp[0]-gp[0];final=cp[2]-gp[2];change=(cb[2]-cb[0])-(gb[2]-gb[0]);ci=np.quantile(change,[.025,.975]).tolist()
        row=dict(control=control,class_index=4,birth_advantage_pp=float(birth),final_advantage_pp=float(final),advantage_change_pp=float(final-birth),advantage_change_CI95_pp=ci,stable_shrink=bool(ci[1]<0))
        for name,i,j in [('old_ranking',1,0),('new_competition',2,1)]:
            d=(cb[i]-cb[j])-(gb[i]-gb[j]);value=float(cp[i]-cp[j]-gp[i]+gp[j]);adj=np.quantile(d,[.00625,.99375]).tolist()
            row[name]=dict(delta_pp=value,CI95_pp=np.quantile(d,[.025,.975]).tolist(),family4_adjusted_CI_pp=adj,stable_extra_loss=bool(adj[1]<0))
        assert abs(row['old_ranking']['delta_pp']+row['new_competition']['delta_pp']-row['advantage_change_pp'])<1e-12
        contrasts.append(row)
    macros=[]
    for st in range(2):
        for a in ARMS:
            p=np.stack([points[st,a,c] for c in range(st+4)]).mean(0);bs=np.stack([samples[st,a,c] for c in range(st+4)]).mean(0)
            row=dict(transition=[st,st+1],arm=a,birth_old_BA_pp=float(p[0]),current_restricted_old_BA_pp=float(p[1]),current_full_old_BA_pp=float(p[2]))
            for name,i,j in [('old_ranking',1,0),('new_competition',2,1),('total',2,0)]:row[name+'_pp']=float(p[i]-p[j]);row[name+'_CI95_pp']=np.quantile(bs[i]-bs[j],[.025,.975]).tolist()
            macros.append(row)
    assert len(rows)==45 and len(baseline)==75 and all(parity)
    assert time.monotonic()-t<170 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*2**30
    gate=all(v['new_competition']['stable_extra_loss'] for v in contrasts) and contrasts[0]['stable_shrink']
    result=dict(status='COMPLETE',source_commit=cfg['source_commit'],reference_commit='19bf801',alpha=.01,training_calls=0,model_forward_calls=0,GPU_calls=0,cached_exposed_test_scores_reused=True,patient_isolation='UNKNOWN',parity=dict(all_original_counts_exact=True,cells=75),bootstrap=dict(draws=10000,seed=74003,unit='known_identity_group_within_class',primary_component_family=4,adjusted_percentiles=[.00625,.99375]),baseline=baseline,cells=rows,old_class_macros=macros,primary_class4_contrasts=contrasts,common_competition_loss_gate=bool(gate),method_success=False,next_decision='STOP_DIAGNOSTIC_NO_AUTOMATIC_EXTENSION',costs=dict(body_wall_seconds=time.monotonic()-t,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    write(root/'aggregate.json',result);signal.alarm(0)

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);start=time.time()
    try:
        main(cfg);write(root/'EXIT.json',dict(exit_code=0,entry_wall_seconds=time.time()-start))
    except Exception:
        write(root/'failure.private.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1,entry_wall_seconds=time.time()-start));raise
