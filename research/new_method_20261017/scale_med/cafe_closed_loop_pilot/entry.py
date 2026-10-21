"""CAFE pooled-feature closed-loop transfer; support-only selection, not full reproduction."""
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
    z=transform(g,t['mapper']);s=dict(F=fscores(g,t),P=p)
    s.update({n:z@w[:-1]+w[-1] for n,w in t['learned'].items()})
    assert all(torch.isfinite(v).all() for v in s.values());return s

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
           VS_PROTOTYPES=m['BA']>tab['T']['BA'] and m['BA']>tab['R']['BA'])
    if not isic:g['CELL_VS_PROTOTYPES']=all(v['M']['BA']>v['T']['BA'] and v['M']['BA']>v['R']['BA'] for v in cells)
    return dict(**g,pass_all=all(g.values()),missing=False)

def learn_closed_loop(g,y,gg,K,t,tag):
    """Pooled-feature CAFE mechanism transfer; support-only selection throughout."""
    z=transform(g,t['mapper']);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1)
    p=weights(y,gg,K);one=torch.nn.functional.one_hot(y,K).double()
    a=x.T@(x*p[:,None]);a=(a+a.T)/2;b=x.T@(one*p[:,None]);ix=torch.triu_indices(len(a),len(a));ap=a[ix.unbind()]
    olda=t['A'] if t['A'].ndim==1 else t['A'][ix.unbind()]
    err=max(float((ap-olda).abs().max()),float((b-t['B'][:,:K]).abs().max()))
    h=a.clone();h.diagonal()[:-1].add_(.01);res=float((h@t['W']-b).abs().max());assert err<1e-10 and res<1e-7
    write(tag+'.PARITY.json',dict(A_B_error=err,F_residual=res))
    mu=torch.stack([(p[y==c,None]*g[y==c]).sum(0)/p[y==c].sum() for c in range(K)]);mu=torch.nn.functional.normalize(mu,dim=1)
    counts=torch.bincount(y,minlength=K);order=sorted(range(K),key=lambda c:(-int(counts[c]),c));hh=(K+2)//3;mm=hh+K//3;head=order[:hh];medium=order[hh:mm];tail=order[mm:]
    gen=torch.Generator().manual_seed(74420+{'ISIC0':0,'ISIC1':1,'HK':2}[tag])
    ws={n:t['W'].clone().requires_grad_() for n in ['R','T','M']};vel={n:torch.zeros_like(t['W']) for n in ws};memory=[];history=[];backward=0;realsteps=0;synsteps=0;virtualsteps=0
    def logits(v,w):
        zz=transform(v,t['mapper']);return zz@w[:-1]+w[-1]
    def risk(w):
        ce=torch.nn.functional.cross_entropy(x@w,y,reduction='none');return torch.stack([(p[y==c]*ce[y==c]).sum()/p[y==c].sum() for c in range(K)])
    def pairloss(w,items):
        v=torch.stack([u[0] for u in items]);yy=torch.tensor([u[1] for u in items]);rr=torch.tensor([u[2] for u in items]);l=logits(v,w);margin=l[torch.arange(len(v)),yy]-l[torch.arange(len(v)),rr];goal=torch.maximum(torch.full_like(margin,.25),margin.detach()+.25);loss=.1*torch.nn.functional.softplus((goal-margin)/.1)
        return torch.stack([loss[yy==c].mean() for c in yy.unique()]).mean()
    for epoch in range(20):
        lr=.0001*(1+__import__('math').cos(__import__('math').pi*epoch/20))/2
        for n,w in ws.items():
            loss=(p*torch.nn.functional.cross_entropy(x@w,y,reduction='none')).sum();grad=torch.autograd.grad(loss,w)[0];backward+=1;realsteps+=1
            with torch.no_grad():vel[n].mul_(.9).add_(grad);w.add_(vel[n],alpha=-lr)
        proposals=[]
        with torch.no_grad():
            avg=(x@ws['M']+x@ws['R'])/2;other=avg.clone();other[torch.arange(len(y)),y]=-torch.inf;rival=other.argmax(1);margin=avg[torch.arange(len(y)),y]-other.max(1).values
            tm=margin[torch.tensor([int(c) in tail for c in y])];threshold=max(.001,float(torch.quantile(tm,.35)))
            for c in tail:
                ix0=torch.where(y==c)[0];bad=ix0[margin[ix0]<=0];low=ix0[(margin[ix0]>0)&(margin[ix0]<=threshold)]
                chosen=torch.cat([bad[torch.argsort(margin[bad])[:2]],low[torch.argsort(margin[low])[:2]]])
                for idx in chosen.tolist():
                    rv=int(rival[idx]);donor=rv if rv in head else max(head,key=lambda d:float(mu[c]@mu[d]));bank=torch.where(y==donor)[0];best=None
                    for v in range(4):
                        j=int(bank[torch.randint(len(bank),(1,),generator=gen)]);mask=torch.randperm(g.shape[1],generator=gen)[:max(1,int(.15*g.shape[1]))];syn=g[idx].clone();strength=.5 if margin[idx]<=0 else .375;syn[mask]+=strength*(g[j,mask]-g[idx,mask])
                        norm=syn.norm();cos=float(syn@g[idx]/(norm*g[idx].norm()));drift=float((syn-g[idx]).norm()/g[idx].norm());pp=(syn/norm)@mu.T;others=pp.clone();others[c]=-torch.inf
                        if cos<.90 or drift>.35 or float(pp[c]-others.max())<-.05:continue
                        sm=[float((logits(syn[None],ws[n])[0,c]-logits(syn[None],ws[n])[0,rv])) for n in ['M','R']];am=[float((x[idx]@ws[n])[c]-(x[idx]@ws[n])[rv]) for n in ['M','R']]
                        valid=all(s>a+.001 for s,a in zip(sm,am)) if margin[idx]<=0 else all(.001<=s<a-.001 for s,a in zip(sm,am))
                        if not valid:continue
                        goal=.05 if margin[idx]<=0 else max(.001,.7*float(margin[idx]));distance=abs(sum(sm)/2-goal)
                        if best is None or distance<best[0]:best=(distance,syn,c,rv)
                    if best is not None:proposals.append((best[1],best[2],best[3]))
            pool=memory+proposals;bounded=[]
            for c in tail:
                cc=[u for u in pool if u[1]==c];cc.sort(key=lambda u:abs(float(logits(u[0][None],ws['M'])[0,c]-logits(u[0][None],ws['M'])[0,u[2]])-.25));keep=[]
                for u in cc:
                    if len(keep)>=8:break
                    if any(float(torch.nn.functional.cosine_similarity(u[0][None],v[0][None]))>=.995 for v in keep):continue
                    keep.append(u)
                bounded+=keep
            before=risk(ws['R']).detach()
        accepted=[]
        for c in tail:
            trial=accepted+[u for u in bounded if u[1]==c]
            if not trial or len(trial)==len(accepted):continue
            grad=torch.autograd.grad(pairloss(ws['R'],trial),ws['R'])[0];backward+=1;virtualsteps+=1
            with torch.no_grad():after=risk(ws['R']-lr*grad);tc=sorted({u[1] for u in trial});ok=bool(after.mean()<before.mean() and (after[tc]<before[tc]).all() and after[head].mean()<=before[head].mean()+.0001 and after[medium].mean()<=before[medium].mean()+.0001)
            if ok:accepted=trial
        memory=[(u[0].detach(),u[1],u[2]) for u in accepted]
        # Same proposed bounded pool; T omits virtual risk admission. Not independent proposal generation.
        for n,items in [('M',memory),('T',bounded)]:
            if items:
                grad=torch.autograd.grad(pairloss(ws[n],items),ws[n])[0];backward+=1;synsteps+=1
                with torch.no_grad():ws[n].add_(grad,alpha=-lr)
        assert all(torch.isfinite(v).all() for v in ws.values())
        history.append(dict(epoch=epoch+1,proposals=len(proposals),bounded=len(bounded),accepted=len(memory),support_judge_not_validation=True))
    obj=dict(A=ap,B=b,W=t['W'],mapper=t['mapper'],means=mu,learned={n:w.detach() for n,w in ws.items()},memory=memory,classes=K,source_commit=cfg['source_commit'])
    path=out/(tag+'.artifact.private.pt');torch.save(obj,path);back=orig(path,weights_only=False)
    for n in ws:assert torch.equal(obj['learned'][n],back['learned'][n])
    assert torch.equal(obj['W'],back['W']) and torch.equal(obj['means'],back['means'])
    pe=max(float((heads(g,obj)[n]-heads(g,back)[n]).abs().max()) for n in ['F','P','R','T','M']);assert pe<1e-10
    write(tag+'.TRAIN.json',dict(history=history,head_classes=head,medium_classes=medium,tail_classes=tail,real_steps=realsteps,synthetic_steps=synsteps,virtual_steps=virtualsteps,backward_calls=backward,support_selection_only=True))
    write(tag+'.HEAD_SEAL.json',dict(bytes=path.stat().st_size,readback_exact=True,support_readback_score_maxabs=pe,query_pending=True))
    return back

start=time.time();code=1
try:
    toy=torch.tensor([[1.,0.],[.5,0.]],dtype=torch.float64);q=torch.tensor([[1.,0.]],dtype=torch.float64)
    pp=(q/q.norm(dim=1)[:,None])@(toy/toy.norm(dim=1)[:,None]).T
    nn=-(q[:,None,:]-toy[None,:,:]).square().sum(2)
    assert torch.equal(pp,torch.ones_like(pp)) and nn[0,0]>nn[0,1]
    lo=dict(BA=.5,correct=[5]*4,recall=[.5]*4);hi=dict(BA=.6,correct=[6]*4,recall=[.6]*4)
    tt=dict(F=lo,M=hi,P=lo,R=lo,T=lo);assert gate(tt,{'0':tt,'1':tt},[dict(harmed=0)]*4,True)['pass_all']
    assert not gate(dict(F=lo,M=lo,P=hi,R=hi,T=hi),{'0':tt},[dict(harmed=0)]*4,True)['pass_all']
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
        models.append(learn_closed_loop(plan['cached_g'][take],plan['y'][take],gg,4,t,'ISIC'+str(k)))
    rr=json.loads((prior/'HK.S_old.rows.private.json').read_text());cc=orig(prior/'HK.S_old.features.private.pt',weights_only=False)
    keep=torch.tensor([r['label']<13 for r in rr]);rr=[r for r in rr if r['label']<13];hy=cc['y'][keep];hg=[r['identity_component'] for r in rr]
    assert torch.equal(hy,torch.tensor([r['label'] for r in rr])) and [g for g,k in zip(cc['components'],keep) if k]==hg
    g=torch.nn.functional.normalize(cc['x'][keep].double(),dim=1);hkbase=orig(hkf,weights_only=False)
    hm=learn_closed_loop(g,hy,hg,13,hkbase,'HK')
    write('MODEL_SEAL.json',dict(artifacts=3,prototype_fits=3,real_head_training_runs=9,epochs_per_run=20,ridge_fits=0,solves=0,mapper_fits=0,encoder_forwards=0,GPU_initialized=False,all_query_pending=True))
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
    write('aggregate.json',dict(decision='CAFE_CLOSED_LOOP_JOINT_POINT_SIGNAL' if ig['pass_all'] and hgate['pass_all'] else 'STOP_CAFE_CLOSED_LOOP_PILOT',method_success=False,ISIC=isic,HK=dict(cohorts=hk,pooled=ht,gate=hgate),prototype_fits=3,real_head_training_runs=9,epochs_per_run=20,ridge_fits=0,solves=0,mapper_fits=0,encoder_forwards=0,GPU_initialized=False,bootstrap=None,historically_exposed=True,patient_independence='UNKNOWN',source_commit=cfg['source_commit']))
    code=0
except BaseException as e:
    write('FAILURE.json',dict(type=type(e).__name__,no_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
    u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_EXIT.json',dict(exit_code=code));write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()))
raise SystemExit(code)
