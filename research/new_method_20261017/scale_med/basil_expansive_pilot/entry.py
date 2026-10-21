"""BASIL expansive-readout transfer; frozen native encoder; not full FSCIL reproduction."""
import os, json, time, resource, traceback, sys
from pathlib import Path
from collections import Counter
import torch

torch.set_num_threads(4)
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
orig=torch.load

def write(n,v):
    (out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def weights(y,groups,K):
    sizes=Counter(groups)
    assert all(len({int(y[i]) for i,g in enumerate(groups) if g==key})==1 for key in sizes)
    nc=[len({groups[i] for i in torch.where(y==c)[0]}) for c in range(K)]
    assert all(nc)
    return torch.tensor([1/(K*nc[int(c)]*sizes[g]) for c,g in zip(y,groups)],dtype=torch.float64)

def transform(g,mp):
    return (((g-mp['gn'][0])/mp['gn'][1])[:,mp['ids']]-mp['on'][0])/mp['on'][1]

def fscores(g,t):
    z=transform(g,t['mapper']);return z@t['W'][:-1]+t['W'][-1]

def heads(g,t):
    q=torch.nn.functional.normalize(g,dim=1);p=q@t['means'].T
    real=torch.stack([(q@r.T).max(1).values for r in t['references']],1)
    tangent=torch.stack([(q@r.T).max(1).values for r in t['tangents']],1)
    ans=dict(F=fscores(g,t),P=p,R=.25*p+.75*real,T=.25*p+.75*tangent,M=.25*p+.75*torch.maximum(real,tangent))
    assert all(torch.isfinite(v).all() for v in ans.values())
    return ans

def metrics(s,y,gg,K):
    pred=s.argmax(1);count=torch.bincount(y,minlength=K);assert (count>0).all()
    ok=[int((pred[y==c]==c).sum()) for c in range(K)];rec=[a/int(b) for a,b in zip(ok,count)]
    return dict(BA=sum(rec)/K,correct=ok,counts=count.tolist(),recall=rec,
                group_BA=float(weights(y,gg,K)@(pred==y).double()),
                confusion=torch.bincount(y*K+pred,minlength=K*K).reshape(K,K).tolist())

def pairs(f,m,y,K):
    a=f.argmax(1)==y;b=m.argmax(1)==y;res=[]
    for c in range(K):
        take=y==c;r=int((~a&b&take).sum());h=int((a&~b&take).sum())
        assert r-h==int(b[take].sum())-int(a[take].sum())
        res.append(dict(class_id=c,rescued=r,harmed=h,net=r-h))
    return res

def gate(tab,folds,paired,isic):
    if not tab or not folds:return dict(pass_all=False,missing=True)
    f,m=tab['F'],tab['M'];K=len(f['correct']);protected=[c for c in range(K) if c!=1]
    cells=list(folds.values())
    g=dict(BA_GAIN=m['BA']>=f['BA']+.01-1e-12,
           CLASS1_GAIN=m['correct'][1]>=f['correct'][1]+1,
           CLASS1_NO_HARM=paired[1]['harmed']==0,
           CELL_BA=all(v['M']['BA']>=v['F']['BA']+ (0 if isic else .01)-1e-12 for v in cells),
           CELL_CLASS1=all(v['M']['correct'][1]>=v['F']['correct'][1]+(0 if isic else 1) for v in cells),
           ALLCLASS_COST=all(m['recall'][c]>=f['recall'][c]-.01-1e-12 for c in protected),
           CELL_ALLCLASS_COST=all(v['M']['recall'][c]>=v['F']['recall'][c]-.01-1e-12 for v in cells for c in protected),
           VS_PROTOTYPES=m['BA']>tab['P']['BA'] and m['BA']>tab['R']['BA'])
    if not isic:g['CELL_VS_PROTOTYPES']=all(v['M']['BA']>v['P']['BA'] and v['M']['BA']>v['R']['BA'] for v in cells)
    return dict(**g,pass_all=all(g.values()),missing=False)

def fit_means(g,y,gg,K,t,tag):
    assert len(g)==len(y)==len(gg) and torch.isfinite(g).all()
    p=weights(y,gg,K);mu=torch.stack([(p[y==c,None]*g[y==c]).sum(0)/p[y==c].sum() for c in range(K)])
    assert torch.isfinite(mu).all() and (mu.norm(dim=1)>1e-12).all()
    z=transform(g,t['mapper']);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1)
    a=x.T@(x*p[:,None]);a=(a+a.T)/2;b=x.T@(torch.nn.functional.one_hot(y,K).double()*p[:,None])
    ix=torch.triu_indices(len(a),len(a));ap=a[ix.unbind()].contiguous()
    olda=t['A'] if t['A'].ndim==1 else t['A'][ix.unbind()]
    err=max(float((ap-olda).abs().max()),float((b-t['B'][:,:K]).abs().max()))
    h=a.clone();h.diagonal()[:-1].add_(.01);res=float((h@t['W']-b).abs().max())
    write(tag+'.PARITY.json',dict(A_B_error=err,F_residual=res,native_norm_min=float(g.norm(dim=1).min()),native_norm_max=float(g.norm(dim=1).max()),support_class_counts=torch.bincount(y,minlength=K).tolist()))
    assert err<=1e-10 and res<=1e-7
    mu=torch.nn.functional.normalize(mu,dim=1)
    gen=torch.Generator().manual_seed(74410+{'ISIC0':0,'ISIC1':1,'HK':2}[tag])
    refs=[torch.nn.functional.normalize(g[y==c],dim=1).contiguous() for c in range(K)]
    tangents=[]
    for c in range(K):
        eta=torch.randn(96,g.shape[1],generator=gen,dtype=torch.float64)
        delta=eta-(eta@mu[c])[:,None]*mu[c]
        assert (delta.norm(dim=1)>1e-12).all()
        delta=torch.nn.functional.normalize(delta,dim=1)
        r=(mu[c]+.025*delta)/(1+.025**2)**.5
        assert float((r.norm(dim=1)-1).abs().max())<1e-10
        tangents.append(r)
    obj=dict(A=ap,B=b,W=t['W'],mapper=t['mapper'],means=mu,references=refs,tangents=tangents,alpha=.01,mixture=.75,tangent_sigma=.025,tangent_count=96,classes=K,source_commit=cfg['source_commit'])
    path=out/(tag+'.artifact.private.pt');torch.save(obj,path) # old fullartifact cap removed by explicit user authorization
    back=orig(path,map_location='cpu',weights_only=False)
    for key in ['A','B','W','means']:assert torch.equal(obj[key],back[key])
    for key in ['references','tangents']:assert all(torch.equal(a,b) for a,b in zip(obj[key],back[key]))
    for key in ['gn','on']:assert all(torch.equal(a,b) for a,b in zip(obj['mapper'][key],back['mapper'][key]))
    assert torch.equal(obj['mapper']['ids'],back['mapper']['ids'])
    s0=heads(g,obj);s1=heads(g,back);pe=max(float((s0[k]-s1[k]).abs().max()) for k in s0)
    assert pe<=1e-10
    write(tag+'.HEAD_SEAL.json',dict(bytes=path.stat().st_size,readback_exact=True,support_readback_score_maxabs=pe,query_pending=True,prototype_fits=1,ridge_fits=0,solves=0))
    return back

start=time.time();code=1
try:
    toy=torch.tensor([[1.,0.],[.5,0.]],dtype=torch.float64);q=torch.tensor([[1.,0.]],dtype=torch.float64)
    pp=(q/q.norm(dim=1)[:,None])@(toy/toy.norm(dim=1)[:,None]).T
    nn=-(q[:,None,:]-toy[None,:,:]).square().sum(2)
    assert torch.equal(pp,torch.ones_like(pp)) and nn[0,0]>nn[0,1]
    lo=dict(BA=.5,correct=[5]*4,recall=[.5]*4);hi=dict(BA=.6,correct=[6]*4,recall=[.6]*4)
    tt=dict(F=lo,M=hi,P=lo,R=lo);assert gate(tt,{'0':tt,'1':tt},[dict(harmed=0)]*4,True)['pass_all']
    assert not gate(dict(F=lo,M=lo,P=hi,R=hi),{'0':tt},[dict(harmed=0)]*4,True)['pass_all']
    assert gate({}, {}, [],True)['missing'];write('SYNTHETIC.json',dict(pass_all=True,solves=0))
    root=Path(cfg['producer_root']);prior=Path(cfg['prior_root']);hkf=Path(cfg['HK_F_artifact']);window=Path(cfg['window_root']).resolve()
    allowed={root/'prep/PLAN.private.pt',root/'prep/0.teacher.private.pt',root/'prep/1.teacher.private.pt',root/'evaluate/responses.private.pt',hkf}
    allowed|={prior/('HK.'+role+'.'+kind+'.private.'+ext) for role in ['S_old','E','Qgate'] for kind,ext in [('features','pt'),('rows','json')]}
    access=dict(query=False)
    def guard(ev,args):
        if ev=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if (p.is_relative_to(window) or p.is_relative_to(prior)) and not p.is_relative_to(out):assert p in allowed,'ASSET_DENIED'
            if p==root/'evaluate/responses.private.pt' or p.name.startswith(('HK.E.','HK.Qgate.')):assert access['query'],'QUERY_BEFORE_SEAL'
    sys.addaudithook(guard)
    plan=orig(root/'prep/PLAN.private.pt',weights_only=False);models=[]
    for k in [0,1]:
        take=plan['fold']==k;gg=[plan['groups'][i] for i in torch.where(take)[0]]
        assert not set(gg)&{plan['groups'][i] for i in torch.where(~take)[0]}
        t=orig(root/'prep'/f'{k}.teacher.private.pt',weights_only=False)
        models.append(fit_means(plan['cached_g'][take],plan['y'][take],gg,4,t,'ISIC'+str(k)))
    rr=json.loads((prior/'HK.S_old.rows.private.json').read_text());cc=orig(prior/'HK.S_old.features.private.pt',weights_only=False)
    keep=torch.tensor([r['label']<13 for r in rr]);rr=[r for r in rr if r['label']<13];hy=cc['y'][keep];hg=[r['identity_component'] for r in rr]
    assert torch.equal(hy,torch.tensor([r['label'] for r in rr])) and [g for g,k in zip(cc['components'],keep) if k]==hg
    g=torch.nn.functional.normalize(cc['x'][keep].double(),dim=1);hkbase=orig(hkf,weights_only=False)
    hm=fit_means(g,hy,hg,13,hkbase,'HK')
    write('MODEL_SEAL.json',dict(artifacts=3,prototype_fits=3,ridge_fits=0,solves=0,mapper_fits=0,encoder_forwards=0,GPU_initialized=False,all_query_pending=True))
    access['query']=True
    scores={n:torch.zeros(2555,4,dtype=torch.float64) for n in ['F','P','R','T','M']};cover=torch.zeros(2555,dtype=torch.long)
    for k,t in enumerate(models):
        take=plan['fold']!=k;ss=heads(plan['cached_g'][take],t)
        for n,v in ss.items():scores[n][take]=v
        cover[take]+=1
    prev=orig(root/'evaluate/responses.private.pt',weights_only=False);err=float((scores['F']-prev['score']['F_out']).abs().max());assert err<=1e-8 and (cover==1).all()
    y=plan['y'];gg=plan['groups'];tab={n:metrics(v,y,gg,4) for n,v in scores.items()};assert tab['F']['correct']==[237,13,64,1626]
    folds={str(k):{n:metrics(v[plan['fold']!=k],y[plan['fold']!=k],[gg[i] for i in torch.where(plan['fold']!=k)[0]],4) for n,v in scores.items()} for k in [0,1]}
    pair={n:pairs(scores['F'],scores[n],y,4) for n in ['M','P','R','T']};ig=gate(tab,folds,pair['M'],True)
    isic=dict(summary=tab,folds=folds,paired=pair,gate=ig,F_query_parity=err)
    write('ISIC.json',isic)
    hk={};paths={r['relative_path'] for r in rr};groups=set(hg);pooled={n:[] for n in scores};ys=[];ggall=[]
    for role in ['E','Qgate']:
        rows=json.loads((prior/('HK.'+role+'.rows.private.json')).read_text());keep=torch.tensor([r['label']<13 for r in rows]);rows=[r for r in rows if r['label']<13]
        cc=orig(prior/('HK.'+role+'.features.private.pt'),weights_only=False);yy=cc['y'][keep];gg=[r['identity_component'] for r in rows];ps={r['relative_path'] for r in rows}
        assert not ps&paths and not set(gg)&groups;paths|=ps;groups|=set(gg)
        assert torch.equal(yy,torch.tensor([r['label'] for r in rows]))
        qg=torch.nn.functional.normalize(cc['x'][keep].double(),dim=1);ss=heads(qg,hm);old=fscores(qg,hkbase)
        pe=float((ss['F']-old).abs().max());assert pe<=1e-8
        row={n:metrics(v,yy,gg,13) for n,v in ss.items()};rh={n:pairs(ss['F'],ss[n],yy,13) for n in ['M','P','R','T']}
        assert row['F']['correct']==cfg['HK_expected'][role]['correct'] and row['F']['counts']==cfg['HK_expected'][role]['counts']
        hk[role]=dict(summary=row,paired=rh,F_query_parity=pe)
        for n in scores:pooled[n].append(ss[n])
        ys.append(yy);ggall+=gg
    yy=torch.cat(ys);hs={n:torch.cat(v) for n,v in pooled.items()};ht={n:metrics(v,yy,ggall,13) for n,v in hs.items()};hp=pairs(hs['F'],hs['M'],yy,13)
    hgate=gate(ht,{r:hk[r]['summary'] for r in hk},hp,False)
    write('aggregate.json',dict(decision='BASIL_EXPANSIVE_JOINT_POINT_SIGNAL' if ig['pass_all'] and hgate['pass_all'] else 'STOP_BASIL_EXPANSIVE_PILOT',method_success=False,ISIC=isic,HK=dict(cohorts=hk,pooled=ht,gate=hgate),prototype_fits=3,ridge_fits=0,solves=0,mapper_fits=0,encoder_forwards=0,GPU_initialized=False,bootstrap=None,historically_exposed=True,patient_independence='UNKNOWN',source_commit=cfg['source_commit']))
    code=0
except BaseException as e:
    write('FAILURE.json',dict(type=type(e).__name__,no_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
    u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_EXIT.json',dict(exit_code=code));write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()))
raise SystemExit(code)
