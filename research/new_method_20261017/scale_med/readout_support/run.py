"""Matched readout-only support coverage diagnostic; no parameter search."""
import os,json,time,resource,traceback,sys
from pathlib import Path
import numpy as np
import torch
from base import geometry,transform,write,selfcheck,PERM,mapped,weights,norm_fit

def main(cfg):
    start=time.monotonic();torch.set_num_threads(4);resource.setrlimit(resource.RLIMIT_CPU,(900,900));selfcheck()
    root=Path(cfg['output']);cache=Path(cfg['cache']).resolve();prior=Path(cfg['prior']).resolve()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(cache.parent):assert p==cache and args[1] in ['r','rb']
            if p.is_relative_to(prior.parent):assert p==prior and args[1] in ['r','rb']
            assert not any(x in p.name for x in ['test.rows','test.features','heads.private','transforms.private','predictions.private'])
    sys.addaudithook(guard)
    old=json.loads(prior.read_text());o=torch.load(cache,weights_only=False);g,h,y=o['g'],o['h'],o['y'];groups=o['groups'];fold=np.asarray(o['fold']);del o
    assert g.shape==(2555,1536) and h.shape==(2555,768) and int((y==1).sum())==20
    assert bool(torch.isfinite(g).all()) and bool(torch.isfinite(h).all())
    rng=np.random.default_rng(74125);expected=np.full(len(y),-1)
    for c in range(4):
        keys=sorted({groups[i] for i in torch.where(y==c)[0].tolist()});rng.shuffle(keys);assign={k:j%4 for j,k in enumerate(keys)}
        for i in torch.where(y==c)[0].tolist():expected[i]=assign[groups[i]]
    assert np.array_equal(fold,expected)
    records={};cells=[];audits=[];heads=0;rh=np.random.default_rng(74127)
    U0=torch.linalg.qr(torch.randn(768,128,generator=torch.Generator().manual_seed(74011),dtype=torch.float64),mode='reduced')[0]
    arms=['G_FULL','G_CORE','R_CORE','P_CORE','C_CORE']
    for f in range(4):
        ti=np.where(fold!=f)[0];qi=np.where(fold==f)[0];assert not {groups[i] for i in ti}&{groups[i] for i in qi}
        gt,ht,yt=g[ti],h[ti],y[ti];gg=[groups[i] for i in ti];w=weights(yt,gg);gn,hn=norm_fit(gt,w),norm_fit(ht,w);u,v=geometry(gt,ht,w,gn,hn)
        _,pc=torch.linalg.eigh(((ht-hn[0])/hn[1]).cuda().T@(((ht-hn[0])/hn[1])*w[:,None]).cuda()/4);pc=pc[:,-128:].cpu();ii=pc.abs().argmax(0);pc*=pc[ii,torch.arange(128)].sign()
        gf=transform('G',1535,gt,ht,w,gn)
        ts=dict(G_FULL=gf,G_CORE=transform('G',1318,gt,ht,w,gn,outnorm=(gf['out_norm'][0][:1318],gf['out_norm'][1])),R_CORE=transform('R',1318,gt,ht,w,gn,hn,U0),P_CORE=transform('P',1318,gt,ht,w,gn,hn,pc),C_CORE=transform('C',1318,gt,ht,w,gn,hn,u,v))
        xs={a:mapped(gt,ht,t) for a,t in ts.items()};qs={a:mapped(g[qi],h[qi],t) for a,t in ts.items()};one=torch.nn.functional.one_hot(yt,4).double()
        def fit(a,ww,mode,rep):
            nonlocal heads
            x=xs[a];A=x.T@(x*ww[:,None])/4;A.diagonal()[:-1].add_(.01);B=x.T@(one*ww[:,None])/4;s=qs[a]@torch.cholesky_solve(B,torch.linalg.cholesky(A));assert bool(torch.isfinite(s).all());err=(s.argmax(1)!=y[qi]).numpy().astype(float)
            tag=f'{a}/{mode}/{rep}';records.setdefault(tag,np.full(len(y),np.nan));assert np.isnan(records[tag][qi]).all();records[tag][qi]=err;heads+=1
            cells.append(dict(fold=f,arm=a,mode=mode,replicate=rep,class1_query=int((y[qi]==1).sum()),class1_image_error=float(err[(y[qi]==1).numpy()].mean())))
        for a in arms:fit(a,w,'BASE',0)
        ids=torch.where(yt==1)[0];keys=sorted({gg[i] for i in ids.tolist()});idx={k:j for j,k in enumerate(keys)};gid=np.array([idx[gg[i]] for i in ids.tolist()])
        for rep in range(4):
            counts=np.bincount(rh.integers(0,len(keys),len(keys)),minlength=len(keys));wd=w.clone();wr=w.clone();wd[ids]*=torch.from_numpy((counts[gid]>0).astype(float));wd[ids]/=wd[ids].sum();wr[ids]*=torch.from_numpy(counts[gid]);assert abs(float(wr[ids].sum())-1)<1e-10
            for a in arms:fit(a,wd,'DROP',rep);fit(a,wr,'REWEIGHT',rep)
            assert time.monotonic()-start<280 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*2**30 and torch.cuda.max_memory_allocated()<4*2**30
            write(root/'status.json',dict(state='RUNNING',fold=f,replicate=rep,heads=heads));write(root/'partial.json',dict(heads=heads,conditions=cells))
        audits.append(dict(fold=f,train_rows=len(ti),query_rows=len(qi),overlap=0,query_class1_rows=int((y[qi]==1).sum())))
    assert heads==180 and len(cells)==180 and len(records)==45 and all(np.isfinite(v).all() for v in records.values())
    cids=np.where(y.numpy()==1)[0];keys=sorted({groups[i] for i in cids});members=[np.array([i for i in cids if groups[i]==k]) for k in keys];assert len(keys)==16
    risk={tag:np.array([e[m].mean() for m in members]) for tag,e in records.items()};avg={}
    for a in arms:
        avg[a,'BASE']=risk[f'{a}/BASE/0']
        for m in ['DROP','REWEIGHT']:avg[a,m]=np.stack([risk[f'{a}/{m}/{j}'] for j in range(4)]).mean(0)
    gf=np.array([fold[m[0]] for m in members]);br=np.random.default_rng(74128);draw=np.concatenate([br.choice(np.where(gf==f)[0],(10000,4),replace=True) for f in range(4)],1)
    def stat(v):
        z=100*v[draw].mean(1);return dict(error_increase_pp=float(100*v.mean()),CI95_pp=np.quantile(z,[.025,.975]).tolist(),family3_CI_pp=np.quantile(z,[.05/6,1-.05/6]).tolist())
    components={};summaries=[]
    for a in arms:
        base,drop,rew=[avg[a,m] for m in ['BASE','DROP','REWEIGHT']];assert np.max(abs((rew-base)-(drop-base)-(rew-drop)))<1e-12
        components[a]=dict(deletion=stat(drop-base),multiplicity=stat(rew-drop),total=stat(rew-base))
        for mode,v in [('BASE',base),('DROP',drop),('REWEIGHT',rew)]:summaries.append(dict(arm=a,mode=mode,class1_group_error_pp=float(100*v.mean())))
    effect={a:avg[a,'REWEIGHT']-avg[a,'BASE'] for a in arms};primary={a:stat(effect['C_CORE']-effect[a]) for a in ['G_CORE','R_CORE','P_CORE']}
    gate=primary['G_CORE']['error_increase_pp']>=5 and all(v['family3_CI_pp'][0]>0 for v in primary.values())
    oldcells={(v['fold'],v['replicate']):v for v in old['conditions'] if v['mapping']=='FIXED' and v['readout']=='H1'}
    parity=max(abs(v['class1_image_error']-oldcells[v['fold'],v['replicate']]['query_class1_error']) for v in cells if v['arm']=='C_CORE' and v['mode']=='REWEIGHT');assert parity<1e-12
    write(root/'aggregate.json',dict(status='COMPLETE',source_commit=cfg['source_commit'],heads=heads,conditions=cells,summaries=summaries,components=components,primary=primary,necessary_C_specific_gate=bool(gate),method_success=False,next_decision='ANALYZE_ONLY' if gate else 'STOP_C_SPECIFIC_ATTRIBUTION',prior_C_parity_max=parity,fold_audit=audits,training_rows=len(y),class1_query_rows=len(cids),class1_groups=len(keys),test_used=False,permanent_validation_used=False,patient_isolation='UNKNOWN',geometry_fixed_before_readout_intervention=True,costs=dict(body_wall_seconds=time.monotonic()-start)))
    write(root/'status.json',dict(state='COMPLETE'))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);start=time.time()
    try:main(cfg);write(root/'EXIT.json',dict(exit_code=0))
    except Exception:write(root/'failure.private.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF);write(root/'COST_RECEIPT.json',dict(entry_wall_seconds=time.time()-start,cpu_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
