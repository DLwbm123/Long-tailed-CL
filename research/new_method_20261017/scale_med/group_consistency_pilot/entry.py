import os,json,time,resource,traceback
from pathlib import Path
from collections import Counter
import torch
from geometry import mapper,transform
torch.set_num_threads(4);cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def weights(y,groups,K):
 n=Counter(groups);nc={c:len({groups[i] for i in torch.where(y==c)[0]}) for c in range(K)};assert all(nc.values())
 return torch.tensor([1/(K*nc[int(c)]*n[g]) for c,g in zip(y,groups)],dtype=torch.float64)
def stats(x,y,groups,K):
 p=weights(y,groups,K);a=x.T@(x*p[:,None]);a=(a+a.T)/2;b=x.T@(torch.nn.functional.one_hot(y,K).double()*p[:,None]);by={}
 for i,g in enumerate(groups):by.setdefault(g,[]).append(i)
 if all(len(v)==1 for v in by.values()):return a,b,torch.zeros_like(a),p
 mu=torch.stack([x[v].mean(0) for v in by.values()]);mass=torch.stack([p[v].sum() for v in by.values()]);s=a-mu.T@(mu*mass[:,None]);s=(s+s.T)/2;s[-1,:]=0;s[:,-1]=0
 return a,b,s,p
def hess(a):
 h=a.clone();h.diagonal()[:-1].add_(.01);return h
def metrics(s,y,groups,K):
 pred=s.argmax(1);n=torch.bincount(y,minlength=K);assert (n>0).all();ok=[int((pred[y==c]==c).sum()) for c in range(K)];rec=[v/int(nn) for v,nn in zip(ok,n)];p=weights(y,groups,K)
 return dict(BA=sum(rec)/K,correct=ok,counts=n.tolist(),recall=rec,group_BA=float(p@(pred==y).double()),confusion=torch.bincount(y*K+pred,minlength=K*K).reshape(K,K).tolist())
def pairs(a,b,y,K):
 aa=a.argmax(1);bb=b.argmax(1);r=[]
 for c in range(K):
  take=y==c;x=aa[take]==c;z=bb[take]==c;R=int((~x&z).sum());H=int((x&~z).sum());assert R-H==int(z.sum())-int(x.sum());r.append(dict(class_id=c,rescued=R,harmed=H,net=R-H))
 return r
def gates(tab,fold,pair):
 if not tab or len(fold)!=2:return dict(pass_all=False,missing=True)
 f,c,r=[tab[k] for k in ['F','C','R']];v=dict(BA_GAIN=c['BA']>=f['BA']+.01-1e-12,CLASS1_GAIN=c['correct'][1]>=f['correct'][1]+1,FOLD_BA=all(v['C']['BA']>=v['F']['BA']-1e-12 for v in fold.values()),FOLD_CLASS1=all(v['C']['correct'][1]>=v['F']['correct'][1] for v in fold.values()),CLASS1_NO_HARM=pair[1]['harmed']==0,ALL_OTHER_COST=all(c['recall'][k]>=f['recall'][k]-.01-1e-12 for k in [0,2,3]),VS_ISOTROPIC=c['BA']>r['BA'] and c['correct'][1]>=r['correct'][1]);return dict(**v,pass_all=all(v.values()),missing=False)
def save_model(tag,a,b,w,mp,K,pen):
 p=out/(tag+'.artifact.private.pt');ix=torch.triu_indices(len(a),len(a));obj=dict(A=a[ix.unbind()].contiguous(),B=b,W=w,mapper=mp,alpha=.01,classes=K,penalty=pen,source_commit=cfg['source_commit']);torch.save(obj,p);assert p.stat().st_size<=19058800;c=orig(p,map_location='cpu',weights_only=False);assert torch.equal(c['A'],obj['A']) and torch.equal(c['B'],b) and torch.equal(c['W'],w)
 for k in ['gn','on']:assert all(torch.equal(t,u) for t,u in zip(c['mapper'][k],mp[k]))
 assert torch.equal(c['mapper']['ids'],mp['ids']);write(tag+'.HEAD_SEAL.json',dict(bytes=p.stat().st_size,readback_exact=True,query_pending=True))
def solve(a,b,tag,mp,K,pen):
 w=torch.linalg.solve(hess(a),b);e=float((hess(a)@w-b).abs().max());write(tag+'.SOLVE.json',dict(residual=e));assert e<=1e-7 and torch.isfinite(w).all();save_model(tag,a,b,w,mp,K,pen);return w
orig=torch.load
start=time.time();code=1
try:
 # No solve in synthetic checks.
 toy=torch.diag(torch.tensor([2.,1.,0.],dtype=torch.float64));iso=torch.diag(torch.tensor([1.5,1.5,0.],dtype=toy.dtype));assert toy.trace()==iso.trace() and torch.equal(toy[-1],torch.zeros(3,dtype=toy.dtype));assert gates({}, {}, [])['missing']
 lo=dict(BA=.5,correct=[5]*4,recall=[.5]*4);hi=dict(BA=.6,correct=[6]*4,recall=[.6]*4);tab=dict(F=lo,C=hi,R=lo);fs={str(k):tab for k in [0,1]};pp=[dict(harmed=0)]*4;assert gates(tab,fs,pp)['pass_all'];assert not gates(dict(F=lo,C=lo,R=hi),fs,pp)['pass_all'];write('SYNTHETIC.json',dict(pass_all=True,solves=0))
 root=Path(cfg['producer_root']);prior=Path(cfg['prior_root']);allowed={root/'prep/PLAN.private.pt',root/'prep/0.teacher.private.pt',root/'prep/1.teacher.private.pt',root/'evaluate/responses.private.pt'}|{prior/('HK.'+r+'.'+t+'.private.'+ext) for r in ['S_old','E','Qgate'] for t,ext in [('features','pt'),('rows','json')]};access=dict(query=False)
 import sys
 window=Path(cfg['window_root']).resolve()
 def guard(ev,args):
  if ev=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve()
   if (p.is_relative_to(window) or p.is_relative_to(prior)) and not p.is_relative_to(out):assert p in allowed,'ASSET_DENIED'
   if p==root/'evaluate/responses.private.pt' or p.name.startswith(('HK.E.','HK.Qgate.')):assert access['query'],'QUERY_BEFORE_SEAL'
 sys.addaudithook(guard)
 plan=orig(root/'prep/PLAN.private.pt',weights_only=False);prepared=[];checks=[];fits=0
 for k in [0,1]:
  take=plan['fold']==k;y=plan['y'][take];gg=[plan['groups'][j] for j in torch.where(take)[0]];t=orig(root/'prep'/f'{k}.teacher.private.pt',weights_only=False);z=transform(plan['cached_g'][take],t['mapper']);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);a,b,s,p=stats(x,y,gg,4);err=max(float((a-t['A']).abs().max()),float((b-t['B'][:,:4]).abs().max()));write(f'ISIC{k}.PARITY.json',dict(A_B_error=err));assert err<=1e-10
  d=len(a)-1;kap=float(s.trace())/d;assert kap>=0 and torch.isfinite(s).all();ac=a+s;ar=a.clone();ar.diagonal()[:-1].add_(kap);wc=solve(ac,b,f'ISIC{k}.C',t['mapper'],4,'one extra observed S');wr=solve(ar,b,f'ISIC{k}.R',t['mapper'],4,'same trace isotropic');fits+=2;prepared.append((t['mapper'],dict(F=t['W'],C=wc,R=wr)));checks.append(dict(fold=k,trace=float(s.trace()),isotropic_alpha_increment=kap,within_quadratic_F=float((t['W']*(s@t['W'])).sum())))
 rr=json.loads((prior/'HK.S_old.rows.private.json').read_text());cache=orig(prior/'HK.S_old.features.private.pt',weights_only=False);keep=torch.tensor([r['label']<13 for r in rr]);rr=[r for r in rr if r['label']<13];hy=cache['y'][keep];hg=[r['identity_component'] for r in rr];assert torch.equal(hy,torch.tensor([r['label'] for r in rr])) and [g for g,k in zip(cache['components'],keep) if k]==hg and len(set(hg))==len(hg)
 g=torch.nn.functional.normalize(cache['x'][keep].double(),dim=1);hp=weights(hy,hg,13);mp=mapper(g,hp);z=transform(g,mp);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);ha,hb,hs,_=stats(x,hy,hg,13);assert torch.equal(hs,torch.zeros_like(hs));hw=solve(ha,hb,'HK.FCR',mp,13,'S exact0; C/R aliases');fits+=1;assert fits==5;write('MODEL_SEAL.json',dict(real_solves=fits,synthetic_solves=0,HK_mapper_fits=1,GPU_initialized=False,HK_S_exact0=True,all_query_pending=True));access['query']=True
 score={n:torch.zeros(2555,4,dtype=torch.float64) for n in ['F','C','R']};cover=torch.zeros(2555,dtype=torch.long)
 for k,(m,heads) in enumerate(prepared):
  take=plan['fold']!=k;z=transform(plan['cached_g'][take],m)
  for n,h in heads.items():score[n][take]=z@h[:-1]+h[-1]
  cover[take]+=1
 prev=orig(root/'evaluate/responses.private.pt',weights_only=False);pe=float((score['F']-prev['score']['F_out']).abs().max());assert pe<=1e-8 and (cover==1).all();y=plan['y'];gg=plan['groups'];tab={n:metrics(v,y,gg,4) for n,v in score.items()};assert tab['F']['correct']==[237,13,64,1626];fold={str(k):{n:metrics(v[plan['fold']!=k],y[plan['fold']!=k],[gg[j] for j in torch.where(plan['fold']!=k)[0]],4) for n,v in score.items()} for k in [0,1]};paired={n:pairs(score['F'],score[n],y,4) for n in ['C','R']};gate=gates(tab,fold,paired['C']);write('ISIC.json',dict(summary=tab,folds=fold,paired=paired,gate=gate,F_parity=pe,scatter=checks))
 hk={};allpaths=set(r['relative_path'] for r in rr);allgroups=set(hg)
 for role in ['E','Qgate']:
  rows=json.loads((prior/('HK.'+role+'.rows.private.json')).read_text());keep=torch.tensor([r['label']<13 for r in rows]);rows=[r for r in rows if r['label']<13];cc=orig(prior/('HK.'+role+'.features.private.pt'),weights_only=False);yy=cc['y'][keep];gg=[r['identity_component'] for r in rows];paths={r['relative_path'] for r in rows};assert not paths&allpaths and not set(gg)&allgroups;allpaths|=paths;allgroups|=set(gg);assert torch.equal(yy,torch.tensor([r['label'] for r in rows]));qg=torch.nn.functional.normalize(cc['x'][keep].double(),dim=1);zz=transform(qg,mp);ss=zz@hw[:-1]+hw[-1];ms=metrics(ss,yy,gg,13);hk[role]=dict(summary={n:ms for n in ['F','C','R']},paired={n:pairs(ss,ss,yy,13) for n in ['C','R']},score_difference_exact0=True)
 write('aggregate.json',dict(decision='FIXED_CONSISTENCY_POINT_SIGNAL' if gate['pass_all'] else 'STOP_FIXED_GROUP_CONSISTENCY_PILOT',method_success=False,ISIC=dict(summary=tab,folds=fold,paired=paired,gate=gate,F_parity=pe,scatter=checks),HK=hk,HK_negative_control_pass=True,actual_solves=5,mapper_fits=1,encoder_forwards=0,parameter_updates=0,GPU_initialized=False,bootstrap=None,patient_isolation='UNKNOWN',historically_exposed=True,source_commit=cfg['source_commit']))
 code=0
except BaseException as e:
 write('FAILURE.json',dict(type=type(e).__name__,no_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_EXIT.json',dict(exit_code=code));write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()))
raise SystemExit(code)
