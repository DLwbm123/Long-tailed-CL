"""Engineering recovery from frozen features; scientific protocol unchanged."""
import json,os,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from core_depth import PERM,mapped,project,planned,BUDGET
from core_attribution import weights,norm_fit

def write(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)

def top(c,rank):
    _,u=torch.linalg.eigh(c);u=u[:,-rank:].cpu();i=u.abs().argmax(0)
    return u*u[i,torch.arange(rank)].sign()

def geometry(g,h,w,gn,hn,d=1318,rank=128):
    gg=((g-gn[0])/gn[1])[:,PERM[:d]].cuda();hh=((h-hn[0])/hn[1]).cuda();ww=w.cuda()
    a=gg.T@(gg*ww[:,None])/4;a.diagonal().add_(.01)
    A=torch.linalg.solve(a,gg.T@(hh*ww[:,None])/4);e=hh-gg@A
    U=top(e.T@(e*ww[:,None])/4,rank);V=(A@U.cuda()).cpu()
    return U,V

def transform(arm,d,g,h,w,gn,hn=None,U=None,V=None,outnorm=None):
    t=dict(arm=arm,d=d,g_norm=gn)
    if arm!='G':t.update(h_norm=hn,U=U)
    if arm=='C':t['V']=V
    if outnorm is None:
        gg=(g-gn[0])/gn[1];hh=None if arm=='G' else (h-hn[0])/hn[1]
        outnorm=norm_fit(project(gg,hh,t),w)
    t['out_norm']=outnorm
    return t

def selfcheck():
    gen=torch.Generator().manual_seed(74120);g=torch.randn(30,1536,generator=gen,dtype=torch.float64)
    h=torch.randn(30,768,generator=gen,dtype=torch.float64);w=torch.ones(24,dtype=torch.float64)
    gn=norm_fit(g[:24],w);hn=norm_fit(h[:24],w);u=torch.eye(768,dtype=h.dtype)[:,:3]
    t=transform('R',5,g[:24],h[:24],w,gn,hn,u)
    a=mapped(g[:24],h[:24],t);g[24:]+=10000;h[24:]-=10000
    t2=transform('R',5,g[:24],h[:24],w,norm_fit(g[:24],w),norm_fit(h[:24],w),u)
    assert torch.equal(a,mapped(g[:24],h[:24],t2))
    assert all(planned(a,d)<=BUDGET for a,d in [('G',1535),('G',1318),('R',1318),('P',1318),('C',1318)])

def main(cfg):
    start=time.monotonic();torch.set_num_threads(4);resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));selfcheck()
    root=Path(cfg['output']);old=Path(cfg['reference']).resolve()
    cache=Path(cfg['feature_cache']).resolve();prior=Path(cfg['prior_partial']).resolve()
    def check(phase,**kw):
        assert time.monotonic()-start<580,'wall cap'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*2**30,'RSS cap'
        if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<4*2**30,'GPU cap'
        write(root/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            assert not p.is_relative_to(old),'No old input, fitted state or later stage'
            if p.is_relative_to(cache.parent):
                assert p in [cache,prior] and args[1] in ['r','rb'],'Only frozen recovery inputs, read-only'
            if not p.is_relative_to(root):
                assert not any(v in p.name for v in ['predictions.private','test.rows','test.features','heads.private','transforms.private']), 'Forbidden historical fit/test'
    # The regression check distinguishes own output from historical reads.
    guard('open',(str(root/'predictions.private.tmp'),'w',0))
    try:guard('open',(str(old/'predictions.private.json'),'r',0))
    except AssertionError:pass
    else:raise AssertionError('Historical input guard selfcheck failed')
    sys.addaudithook(guard)
    probe=root/'output_check.private.json';write(probe,dict(ok=True));assert json.loads(probe.read_text())==dict(ok=True);probe.unlink()
    obj=torch.load(cache,weights_only=False);g,h,y=obj['g'],obj['h'],obj['y'];groups=obj['groups'];fold=np.asarray(obj['fold']);del obj
    assert g.shape==(2555,1536) and h.shape==(2555,768)
    assert bool(torch.isfinite(g).all()) and bool(torch.isfinite(h).all())
    assert [int((y==c).sum()) for c in range(4)]==[307,20,170,2058]
    expected=np.full(len(y),-1,dtype=int);rng=np.random.default_rng(74125)
    for c in range(4):
        keys=sorted({groups[i] for i in torch.where(y==c)[0].tolist()});rng.shuffle(keys);assign={k:j%4 for j,k in enumerate(keys)}
        for i in torch.where(y==c)[0].tolist():expected[i]=assign[groups[i]]
    assert np.array_equal(fold,expected)
    assert len({groups[i] for i in torch.where(y==1)[0].tolist()})==16
    assert all(len({groups[i] for i in np.where((fold==f)&(y.numpy()==1))[0]})==4 for f in range(4))
    previous=json.loads(prior.read_text());extract_seconds=0.;parity=None
    check('cache_loaded',training_rows=len(y))
    records={};fit_counts=dict(heads=0,C_geometry=0,P_geometry=0);audits=[];fold_metrics=[]
    random_U=torch.linalg.qr(torch.randn(768,128,generator=torch.Generator().manual_seed(74011),dtype=torch.float64),mode='reduced')[0]
    rg=np.random.default_rng(74126);rh=np.random.default_rng(74127)
    def keep(tag,ix,s):
        assert bool(torch.isfinite(s).all());yy=y[ix].numpy();sc=s.numpy();pred=sc.argmax(1)
        alt=sc.copy();alt[np.arange(len(yy)),yy]=-np.inf;mar=sc[np.arange(len(yy)),yy]-alt.max(1)
        if tag not in records:records[tag]=dict(error=np.full(len(y),np.nan),margin=np.full(len(y),np.nan))
        assert np.isnan(records[tag]['error'][ix]).all()
        records[tag]['error'][ix]=(pred!=yy);records[tag]['margin'][ix]=mar
    for f in range(4):
        ti=np.where(fold!=f)[0];qi=np.where(fold==f)[0]
        assert not {groups[i] for i in ti}&{groups[i] for i in qi}
        gt,ht,yt=g[ti],h[ti],y[ti];gg=[groups[i] for i in ti];w=weights(yt,gg)
        gn=norm_fit(gt,w);hn=norm_fit(ht,w);U,V=geometry(gt,ht,w,gn,hn);fit_counts['C_geometry']+=1
        pc=top(((ht-hn[0])/hn[1]).cuda().T@(((ht-hn[0])/hn[1])*w[:,None]).cuda()/4,128);fit_counts['P_geometry']+=1
        C0=transform('C',1318,gt,ht,w,gn,hn,U,V)
        gf=transform('G',1535,gt,ht,w,gn)
        gc=transform('G',1318,gt,ht,w,gn,outnorm=(gf['out_norm'][0][:1318],gf['out_norm'][1]))
        ts={'G_FULL':gf,'G_CORE':gc,'R_CORE':transform('R',1318,gt,ht,w,gn,hn,random_U),'P_CORE':transform('P',1318,gt,ht,w,gn,hn,pc),'C_CORE':C0}
        def fit(t,ww):
            x=mapped(gt,ht,t);a=x.T@(x*ww[:,None])/4;b=x.T@(torch.nn.functional.one_hot(yt,4).double()*ww[:,None])/4
            a.diagonal()[:-1].add_(.01);head=torch.cholesky_solve(b,torch.linalg.cholesky(a));s=mapped(g[qi],h[qi],t)@head
            fit_counts['heads']+=1;return s
        scores0={a:fit(t,w) for a,t in ts.items()}
        for a,s in scores0.items():keep(a,qi,s)
        cids=torch.where(yt==1)[0];keys=sorted({gg[i] for i in cids.tolist()});lookup={k:j for j,k in enumerate(keys)};gid=np.array([lookup[gg[i]] for i in cids.tolist()])
        def boot(r):
            cc=np.bincount(r.integers(0,len(keys),len(keys)),minlength=len(keys));wb=w.clone();wb[cids]*=torch.from_numpy(cc[gid])
            assert abs(float(wb[cids].sum())-1)<1e-10 and bool((wb>=0).all());return wb
        for rep in range(4):
            wg=boot(rg);wh=boot(rh);ng=norm_fit(gt,wg);nh=norm_fit(ht,wg)
            Ug,Vg=geometry(gt,ht,wg,gn,hn);Uf,Vf=geometry(gt,ht,wg,ng,nh);fit_counts['C_geometry']+=2
            variants={'FIXED':C0,'NORM':transform('C',1318,gt,ht,wg,ng,nh,U,V),'GEOM':transform('C',1318,gt,ht,wg,gn,hn,Ug,Vg,outnorm=C0['out_norm']),'FULL':transform('C',1318,gt,ht,wg,ng,nh,Uf,Vf)}
            for mode,t in variants.items():
                for head_mode,ww in [('H0',w),('H1',wh)]:
                    ss=scores0['C_CORE'] if mode=='FIXED' and head_mode=='H0' else fit(t,ww)
                    keep(f'{mode}/{head_mode}/{rep}',qi,ss)
                    fold_metrics.append(dict(fold=f,replicate=rep,mapping=mode,readout=head_mode,query_count=len(qi),class1_query=int((y[qi]==1).sum()),query_class1_error=float((ss.argmax(1)[y[qi]==1]!=1).double().mean())))
            check('fits',fold=f,replicate=rep,unique_heads=fit_counts['heads']);write(root/'partial.json',dict(fit_counts=fit_counts,conditions=fold_metrics))
        audits.append(dict(fold=f,train_rows=len(ti),query_rows=len(qi),train_class1_rows=int((yt==1).sum()),query_class1_rows=int((y[qi]==1).sum()),train_query_group_overlap=0,all_fit_indices_from_outer_train=True,normalization_geometry_and_heads_exclude_query=True))
    assert fit_counts==dict(heads=132,C_geometry=36,P_geometry=4) and len(fold_metrics)==128
    assert all(np.isfinite(v[k]).all() for v in records.values() for k in ['error','margin'])
    yy=y.numpy();c1=np.where(yy==1)[0];keys=sorted({groups[i] for i in c1});members=[np.array([i for i in c1 if groups[i]==k]) for k in keys];gfold=np.array([fold[m[0]] for m in members])
    def group_values(tag):return np.array([records[tag]['error'][m].mean() for m in members])
    risk={};summaries=[]
    for tag,v in records.items():
        gv=group_values(tag);risk[tag]=gv
        summaries.append(dict(condition=tag,class1_group_error_pp=float(100*gv.mean()),class1_image_error_pp=float(100*v['error'][c1].mean()),class1_margin_mean=float(v['margin'][c1].mean()),BA_image_pp=float(100*np.mean([1-v['error'][yy==c].mean() for c in range(4)]))))
    avg={f'{m}/{hh}':np.stack([risk[f'{m}/{hh}/{j}'] for j in range(4)]).mean(0) for m in ['FIXED','NORM','GEOM','FULL'] for hh in ['H0','H1']}
    base=avg['FIXED/H0'];DG=avg['GEOM/H0']-base;DN=avg['NORM/H0']-base;DH=avg['FIXED/H1']-base
    comparisons={'D_G':DG,'D_N':DN,'D_H':DH,'D_G_minus_D_N':DG-DN,'D_G_minus_D_H':DG-DH,'C_minus_G_CORE':risk['C_CORE']-risk['G_CORE']}
    br=np.random.default_rng(74128);draw=np.concatenate([br.choice(np.where(gfold==f)[0],size=(10000,4),replace=True) for f in range(4)],1)
    primary={}
    for name,v in comparisons.items():
        bs=100*v[draw].mean(1);primary[name]=dict(error_increase_pp=float(100*v.mean()),CI95_pp=np.quantile(bs,[.025,.975]).tolist(),family6_adjusted_CI_pp=np.quantile(bs,[.05/12,1-.05/12]).tolist())
    gate=primary['D_G']['error_increase_pp']>=5 and all(primary[k]['family6_adjusted_CI_pp'][0]>0 for k in ['D_G','D_G_minus_D_N','D_G_minus_D_H','C_minus_G_CORE'])
    controls={}
    for name,a,b in [('G_CORE_minus_G_FULL','G_CORE','G_FULL'),('C_minus_G_FULL','C_CORE','G_FULL'),('C_minus_R_CORE','C_CORE','R_CORE'),('C_minus_P_CORE','C_CORE','P_CORE')]:
        v=risk[a]-risk[b];controls[name]=dict(error_increase_pp=float(100*v.mean()),CI95_pp=np.quantile(100*v[draw].mean(1),[.025,.975]).tolist())
    interactions={hh:float(100*(avg[f'FULL/{hh}']-avg[f'GEOM/{hh}']-avg[f'NORM/{hh}']+avg[f'FIXED/{hh}']).mean()) for hh in ['H0','H1']}
    assert previous['fit_counts']==fit_counts
    assert len(previous['conditions'])==len(fold_metrics)
    max_error=max(abs(a['query_class1_error']-b['query_class1_error']) for a,b in zip(previous['conditions'],fold_metrics))
    assert max_error<1e-12,'Recovery auxiliary parity failed'
    assert all(all(a[k]==b[k] for k in ['fold','replicate','mapping','readout','query_count','class1_query']) for a,b in zip(previous['conditions'],fold_metrics))
    check('complete')
    result=dict(status='COMPLETE',source_commit=cfg['source_commit'],alpha=.01,global_core=1318,rank=128,outer_folds=4,repeats=4,training_rows=len(y),class1_query_rows=len(c1),class1_query_groups=len(keys),fold_audit=audits,fit_counts=fit_counts,conditions=fold_metrics,summaries=summaries,primary=primary,controls=controls,normalization_geometry_interaction_pp=interactions,necessary_geometry_risk_gate=bool(gate),method_success=False,next_decision='ANALYZE_NOT_METHOD_SUCCESS' if gate else 'STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION',logical_state_bytes={a:planned(k,d) for a,k,d in [('G_FULL','G',1535),('G_CORE','G',1318),('R_CORE','R',1318),('P_CORE','P',1318),('C_CORE','C',1318)]},recovery_auxiliary_parity_max=max_error,feature_cache_reused=True,feature_hook_parity=None,feature_hook_note='Frozen cache from original run; not re-extracted or rechecked',prior_failure_preserved=True,test_used=False,permanent_validation_used=False,old_fitted_transform_used=False,patient_isolation='UNKNOWN',bootstrap=dict(draws=10000,seed=74128,unit='known_group_within_outer_fold',primary_family=6),costs=dict(body_wall_seconds=time.monotonic()-start,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),feature_extraction_seconds=extract_seconds,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
    write(root/'aggregate.json',result)
    write(root/'predictions.private.json',dict(groups=groups,fold=fold.tolist(),records={k:{j:v.tolist() for j,v in val.items()} for k,val in records.items()}))
    write(root/'status.json',dict(state='COMPLETE',necessary_gate=bool(gate)))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);start=time.time()
    try:main(cfg);write(root/'EXIT.json',dict(exit_code=0,entry_wall_seconds=time.time()-start))
    except Exception:
        write(root/'failure.private.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1,entry_wall_seconds=time.time()-start));raise

    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        write(root/'COST_RECEIPT.json',dict(entry_wall_seconds=time.time()-start,cpu_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
