"""CPU-only fixed support completion; held-out queries never select or fit."""
import json,os,sys,time,resource,traceback
from pathlib import Path
import numpy as np
import torch
from core_depth import PERM,mapped,project,planned
from core_attribution import weights,norm_fit

ARMS=['G','P','C'];CONDITIONS=['BASE','WITHIN','RANDOM','NEAR','FAR','ALL']
def write(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)

def choose(g,y,groups,rng):
    ids=torch.where(y==1)[0].tolist();ks=sorted({groups[i] for i in ids});members={k:sorted(i for i in ids if groups[i]==k) for k in ks}
    assert len(ks)==12,'UNRESOLVED_TRAIN_GROUP_COUNT'
    multi=[k for k in ks if len(members[k])>1];assert len(multi)<=6,'UNRESOLVED_BASE_GROUP_FEASIBILITY'
    rest=[k for k in ks if k not in multi];order=rng.permutation(len(rest)).tolist();basekeys=sorted(multi+[rest[j] for j in order[:6-len(multi)]])
    base=[members[k][0] for k in basekeys];extra=sorted(i for k in basekeys for i in members[k][1:]);assert len(extra)>=2,'UNRESOLVED_WITHIN_IMAGES'
    other=[i for i in range(len(y)) if int(y[i])!=1];initial=sorted(other+base);wg=weights(y[initial],[groups[i] for i in initial]);mu,scale=norm_fit(g[initial][:,PERM[:1318]],wg)
    core=((g[:,PERM[:1318]]-mu)/scale).numpy();pool=[members[k][0] for k in ks if k not in basekeys];assert len(pool)==6
    def acquire(far):
        remaining=pool.copy();support=base.copy();out=[]
        for _ in range(2):
            dist=((core[remaining,None,:]-core[np.array(support)[None,:],:])**2).sum(2).min(1)
            j=int(dist.argmax() if far else dist.argmin());v=remaining.pop(j);support.append(v);out.append(v)
        return out
    near,far=acquire(False),acquire(True);rand=[pool[j] for j in rng.permutation(len(pool))[:2]]
    selected=dict(BASE=base,WITHIN=base+extra[:2],RANDOM=base+rand,NEAR=base+near,FAR=base+far,ALL=ids)
    conditions={k:np.array(sorted(other+v),dtype=int) for k,v in selected.items()}
    audit={}
    for k,v in selected.items():
        assert len(set(v))==len(v) and all(int(y[i])==1 for i in v)
        audit[k]=dict(class1_rows=len(v),class1_groups=len({groups[i] for i in v}))
    assert [(audit[k]['class1_rows'],audit[k]['class1_groups']) for k in CONDITIONS[:5]]==[(6,6),(8,6),(8,8),(8,8),(8,8)]
    def radius(s):return float(np.sqrt(((core[pool,None,:]-core[np.array(s)[None,:],:])**2).sum(2).min(1).max()))
    r0=radius(base);rn=radius(base+near);rf=radius(base+far)
    manipulation=dict(base_radius=r0,near_radius=rn,far_radius=rf,near_ratio=rn/r0 if r0>1e-10 else None,far_ratio=rf/r0 if r0>1e-10 else None)
    return conditions,audit,manipulation

def top(c):
    _,u=torch.linalg.eigh(c);u=u[:,-128:];ii=u.abs().argmax(0);return u*u[ii,torch.arange(128)].sign()

def transforms(g,h,y,groups):
    w=weights(y,groups);gn,hn=norm_fit(g,w),norm_fit(h,w);gg=(g-gn[0])/gn[1];hh=(h-hn[0])/hn[1];core=gg[:,PERM[:1318]]
    A=core.T@(core*w[:,None])/4;A.diagonal().add_(.01);reg=torch.linalg.solve(A,core.T@(hh*w[:,None])/4);e=hh-core@reg;uc=top(e.T@(e*w[:,None])/4);up=top(hh.T@(hh*w[:,None])/4)
    out={}
    for a in ARMS:
        t=dict(arm=a,d=1318,g_norm=gn)
        if a!='G':t.update(h_norm=hn,U=up if a=='P' else uc)
        if a=='C':t['V']=reg@uc
        t['out_norm']=norm_fit(project(gg,None if a=='G' else hh,t),w);out[a]=t
    return out

def selfcheck():
    gen=torch.Generator().manual_seed(74134);data=torch.randn(17,1536,generator=gen,dtype=torch.float64)
    groups=['a','a','b','b']+[f'g{i}' for i in range(10)]+['query']*3;y=torch.ones(14,dtype=torch.long)
    first=choose(data[:14],y,groups[:14],np.random.default_rng(74132));data[14:]+=100000
    second=choose(data[:14],y,groups[:14],np.random.default_rng(74132))
    assert all(np.array_equal(first[0][k],second[0][k]) for k in CONDITIONS)
    assert first[1]==second[1] and first[2]==second[2]
    assert all(planned(a,1318)<=19058800 for a in ARMS)

def main(cfg):
    start=time.monotonic();torch.set_num_threads(4);resource.setrlimit(resource.RLIMIT_CPU,(1800,1800));selfcheck();root=Path(cfg['output']);cache=Path(cfg['cache']).resolve()
    def check(phase,**kw):
        assert time.monotonic()-start<580 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*2**30
        assert not torch.cuda.is_initialized(),'CPU_ONLY'
        write(root/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(cache.parent):assert p==cache and args[1] in ['r','rb'],'FROZEN_CACHE_ONLY'
            if not p.is_relative_to(root):assert not any(s in p.name for s in ['predictions.private','test.rows','test.features','heads.private','transforms.private']),'NO_HISTORICAL_FIT_OR_TEST'
    sys.addaudithook(guard)
    o=torch.load(cache,weights_only=False);g,h,y,groups,fold=o['g'],o['h'],o['y'],o['groups'],np.asarray(o['fold']);del o
    assert g.shape==(2555,1536) and h.shape==(2555,768) and [int((y==c).sum()) for c in range(4)]==[307,20,170,2058]
    assert bool(torch.isfinite(g).all()) and bool(torch.isfinite(h).all())
    expected=np.full(len(y),-1);rng=np.random.default_rng(74125)
    for c in range(4):
        keys=sorted({groups[i] for i in torch.where(y==c)[0].tolist()});rng.shuffle(keys);mapping={k:j%4 for j,k in enumerate(keys)}
        for i in torch.where(y==c)[0].tolist():expected[i]=mapping[groups[i]]
    assert np.array_equal(fold,expected)
    select_rng=np.random.default_rng(74132);artifacts=[];selection_private=[];audits=[];heads=0
    for f in range(4):
        ti=np.where(fold!=f)[0];qi=np.where(fold==f)[0];assert not {groups[i] for i in ti}&{groups[i] for i in qi}
        gt,ht,yt=g[ti],h[ti],y[ti];gg=[groups[i] for i in ti];cs,ca,coverage=choose(gt,yt,gg,select_rng);base=cs['BASE']
        ts=transforms(gt[base],ht[base],yt[base],[gg[i] for i in base]);model={}
        for a,t in ts.items():
            x=mapped(gt,ht,t)
            for k,idx in cs.items():
                assert not set(ti[idx])&set(qi);w=weights(yt[idx],[gg[i] for i in idx]);assert all(abs(float(w[yt[idx]==c].sum())-1)<1e-10 for c in range(4))
                xx=x[idx];A=xx.T@(xx*w[:,None])/4;A.diagonal()[:-1].add_(.01);B=xx.T@(torch.nn.functional.one_hot(yt[idx],4).double()*w[:,None])/4
                head=torch.cholesky_solve(B,torch.linalg.cholesky(A));assert bool(torch.isfinite(head).all());model[a,k]=head;heads+=1;check('fit',fold=f,heads=heads)
        artifacts.append(dict(fold=f,query_indices=qi,transforms=ts,heads=model))
        selection_private.append(dict(fold=f,conditions={k:ti[v].tolist() for k,v in cs.items()},query=qi.tolist()))
        audits.append(dict(fold=f,query_rows=len(qi),class1_query_rows=int((y[qi]==1).sum()),class1_query_groups=len({groups[i] for i in qi if int(y[i])==1}),query_group_overlap=0,transforms_fit_BASE_only=True,selection_outer_train_only=True,conditions=ca,coverage=coverage))
        write(root/'partial.json',dict(heads=heads,fold_audit=audits))
    assert heads==72 and len(artifacts)==4
    torch.save(artifacts,root/'heads.private.pt');write(root/'SELECTION.private.json',selection_private);write(root/'HEAD_SEAL.json',dict(heads=heads,transforms=12,C_geometry=4,P_geometry=4,query_scored=False,source_commit=cfg['source_commit']))
    check('evaluate',heads=heads)
    pred={(a,k):np.full(len(y),-1,dtype=int) for a in ARMS for k in CONDITIONS}
    for fobj in artifacts:
        qi=fobj['query_indices']
        for a,t in fobj['transforms'].items():
            x=mapped(g[qi],h[qi],t)
            for k in CONDITIONS:
                score=x@fobj['heads'][a,k];assert bool(torch.isfinite(score).all());pred[a,k][qi]=score.argmax(1).numpy()
    assert all(bool((v>=0).all()) for v in pred.values())
    yn=y.numpy();ids=np.where(yn==1)[0];keys=sorted({groups[i] for i in ids});assert len(keys)==16
    members=[np.array([i for i in ids if groups[i]==key]) for key in keys];gf=np.array([fold[m[0]] for m in members]);assert all(int((gf==f).sum())==4 for f in range(4))
    risk={key:np.array([(p[m]!=yn[m]).mean() for m in members]) for key,p in pred.items()}
    rng=np.random.default_rng(74133);draw=np.concatenate([rng.choice(np.where(gf==f)[0],(10000,4),replace=True) for f in range(4)],1)
    def interval(v):
        bs=100*v[draw].mean(1);return dict(benefit_pp=float(100*v.mean()),CI95_pp=np.quantile(bs,[.025,.975]).tolist(),family2_CI_pp=np.quantile(bs,[.0125,.9875]).tolist())
    effects={a:dict(independent=interval(risk[a,'WITHIN']-risk[a,'RANDOM']),coverage=interval(risk[a,'NEAR']-risk[a,'FAR'])) for a in ARMS}
    manipulation=all(v['coverage']['near_ratio'] is not None for v in audits)
    contrast=float(np.mean([v['coverage']['near_ratio']-v['coverage']['far_ratio'] for v in audits])) if manipulation else None
    manipulation=bool(manipulation and contrast>1e-10)
    primary=effects['G'];pass_i=primary['independent']['benefit_pp']>=5 and primary['independent']['family2_CI_pp'][0]>0
    pass_k=primary['coverage']['benefit_pp']>=5 and primary['coverage']['family2_CI_pp'][0]>0 and manipulation
    consistent=all(effects[a][k]['benefit_pp']>=0 for a in ['P','C'] for k in ['independent','coverage'])
    gate=bool(pass_i and pass_k and consistent)
    summaries=[]
    for (a,k),p in pred.items():
        recalls=[float((p[yn==c]==c).mean()) for c in range(4)]
        summaries.append(dict(arm=a,condition=k,class1_group_error_pp=float(100*risk[a,k].mean()),class1_image_error_pp=float(100*(p[ids]!=1).mean()),BA_pp=float(100*np.mean(recalls)),other_class_macro_recall_pp=float(100*np.mean([recalls[c] for c in [0,2,3]])),recall_by_class_pp=[100*v for v in recalls]))
    result=dict(status='COMPLETE',source_commit=cfg['source_commit'],heads=heads,C_geometry=4,P_geometry=4,training_rows=len(y),class1_query_rows=len(ids),class1_query_groups=len(keys),fold_audit=audits,summaries=summaries,effects=effects,primary=primary,coverage_manipulation_pass=manipulation,mean_near_minus_far_radius_ratio=contrast,independent_gate=bool(pass_i),coverage_gate=bool(pass_k),head_sign_consistency=bool(consistent),necessary_gate=gate,method_success=False,next_decision='ANALYZE_NOT_METHOD_SUCCESS' if gate else 'STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION',bootstrap=dict(draws=10000,seed=74133,primary_family=2),planned_state_bytes={a:planned(a,1318) for a in ARMS},test_used=False,permanent_validation_used=False,query_used_for_fit_or_selection=False,patient_isolation='UNKNOWN',costs=dict(body_wall_seconds=time.monotonic()-start,new_image_forwards=0,GPU_seconds=0))
    check('complete');write(root/'aggregate.json',result);write(root/'status.json',dict(state='COMPLETE',necessary_gate=gate))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);start=time.time()
    try:main(cfg);write(root/'EXIT.json',dict(exit_code=0))
    except Exception:write(root/'failure.private.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF);write(root/'COST_RECEIPT.json',dict(entry_wall_seconds=time.time()-start,cpu_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()))
