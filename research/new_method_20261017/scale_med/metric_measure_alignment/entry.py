"""Fixed target-measure intervention on original frozen F. Exactly two solves."""
import os,json,time,resource,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from metrics import transform,qweights,hessian,points,pairs
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(4)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def iw(y):
 n=torch.bincount(y,minlength=4);assert (n>0).all()
 return 1/(4*n[y].double())
def gate(tab,folds,pair):
 if set(tab)!={'G','I'} or set(folds)!={'0','1'}:return dict(pass_all=False,missing=True)
 g,i=tab['G'],tab['I'];c=dict(BA_GAIN=100*(i['BA']-g['BA'])>=1-1e-12,CLASS1_GAIN=i['correct'][1]>=g['correct'][1]+1,FOLD_BA=all(v['I']['BA']>=v['G']['BA']-1e-12 for v in folds.values()),FOLD_CLASS1=all(v['I']['correct'][1]>=v['G']['correct'][1] for v in folds.values()),CLASS1_NO_HARM=pair[1]['harmed']==0,OTHER_CLASS_COST=all(i['recall'][k]>=g['recall'][k]-.01-1e-12 for k in [0,2,3]))
 return dict(**c,pass_all=all(c.values()),missing=False)
def bootstrap(base,new,y,groups):
 rng=np.random.default_rng(74342);draw=[]
 for c in range(4):
  keys=sorted({g for j,g in enumerate(groups) if int(y[j])==c});counts=[];diff=[]
  for key in keys:
   take=torch.tensor([g==key for g in groups]);counts.append(int(take.sum()));diff.append(int((new[take]==y[take]).sum())-int((base[take]==y[take]).sum()))
  m=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=10000)
  draw.append((m@np.asarray(diff))/(m@np.asarray(counts)))
 vals={'BA_pp':100*np.mean(draw,axis=0),'class1_pp':100*draw[1]}
 return {k:dict(adjusted_CI=np.quantile(v,[.0125,.9875]).tolist(),seed=74342,draws=10000) for k,v in vals.items()}
def synthetic():
 y=torch.tensor([0,0,1,1,2,2,3,3]);groups=list('aabbccdd');assert torch.equal(qweights(y,groups),iw(y))
 y2=torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3]);g2=[str(c)+g for c in range(4) for g in ['a','b','b']];q=qweights(y2,g2);p=iw(y2);assert not torch.equal(q,p)
 assert all(abs(float(q[y2==c].sum())-.25)<1e-15 and abs(float(p[y2==c].sum())-.25)<1e-15 for c in range(4))
 a=torch.eye(4,dtype=torch.float64);h=hessian(a);assert h[-1,-1]==1 and torch.allclose(h.diagonal()[:-1]-1,torch.full((3,),.01,dtype=a.dtype),atol=1e-15,rtol=0)
 lo=dict(BA=.5,correct=[5]*4,recall=[.5]*4);hi=dict(BA=.6,correct=[6]*4,recall=[.6]*4);pair=[dict(harmed=0) for _ in range(4)];fs={str(k):dict(G=lo,I=hi) for k in [0,1]};assert gate(dict(G=lo,I=hi),fs,pair)['pass_all'];pair[1]['harmed']=1;assert not gate(dict(G=lo,I=hi),fs,pair)['pass_all'];assert gate({}, {}, [])['missing']
 yy=torch.arange(32)%4;gg=[str(i) for i in range(32)];bb=(yy+1)%4;ci=bootstrap(bb,yy,yy,gg);assert all(v['adjusted_CI'][0]>0 for v in ci.values());zero=bootstrap(yy,yy,yy,gg);assert all(v['adjusted_CI']==[0,0] for v in zero.values())
 write('SYNTHETIC.json',dict(pass_all=True,real_or_toy_solves=0,weights_gates_bootstrap=True))
def main():
 synthetic();old=Path(cfg['sealed_root']);allowed={old/'prep/PLAN.private.pt',old/'evaluate/responses.private.pt',old/'prep/0.teacher.private.pt',old/'prep/1.teacher.private.pt'};orig=torch.load
 def load(p):
  p=Path(p);assert p in allowed or p.parent==out and p.name.endswith('.artifact.private.pt'),'ASSET_DENIED'
  return orig(p,map_location='cpu',weights_only=False)
 # Data access is limited to original F PLAN, support-only teachers and sealed F replay responses.
 torch.load=load;window=Path(cfg['window_root']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(window) and not p.is_relative_to(out):assert p in allowed,'PRIVATE_WINDOW_ASSET_DENIED'
 import sys;sys.addaudithook(guard)
 plan=load(old/'prep/PLAN.private.pt');y=plan['y'];fold=plan['fold'];groups=plan['groups'];g=plan['cached_g'];assert len(y)==2555 and g.shape==(2555,1536)
 assert not {groups[j] for j in torch.where(fold==0)[0]} & {groups[j] for j in torch.where(fold==1)[0]}
 measure=[];inputs=[]
 for k in [0,1]:
  take=fold==k;yy=y[take];gg=[groups[j] for j in torch.where(take)[0]];pg=qweights(yy,gg);pi=iw(yy);cells=[]
  for c in range(4):
   idx=yy==c;ss=Counter(gg[j] for j in torch.where(idx)[0]);sizes=sorted(ss.values());ratio=pi[idx]/pg[idx]
   cells.append(dict(class_id=c,images=int(idx.sum()),groups=len(ss),group_size_min_median_max=[min(sizes),float(np.median(sizes)),max(sizes)],conditional_TV=float(2*(pg[idx]-pi[idx]).abs().sum()),I_over_G_weight_min_max=[float(ratio.min()),float(ratio.max())]))
  measure.append(dict(producer=k,classes=cells));inputs.append((take,yy,gg,pg,pi))
 write('MEASURE_SEAL.json',dict(folds=measure,query_statistics_used=False))
 if all(v['conditional_TV']==0 for m in measure for v in m['classes']):
  write('aggregate.json',dict(decision='STOP_TARGET_MEASURES_EQUAL',method_success=False,actual_solves=0,measure=measure));return
 prepared=[];fits=[];sizes={}
 for k,(take,yy,gg,pg,pi) in enumerate(inputs):
  teacher=load(old/'prep'/f'{k}.teacher.private.pt');mp=teacher['mapper'];z=transform(g[take],mp);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);target=torch.nn.functional.one_hot(yy,6).double()
  ga=x.T@(x*pg[:,None]);ga=(ga+ga.T)/2;gb=x.T@(target*pg[:,None]);err=max(float((ga-teacher['A']).abs().max()),float((gb-teacher['B']).abs().max()));resg=float((hessian(teacher['A'])@teacher['W']-teacher['B'][:,:4]).abs().max())
  write(f'{k}.PRECHECK.json',dict(G_stat_error=err,G_residual=resg));assert err<=1e-10 and resg<=1e-7
  a=x.T@(x*pi[:,None]);a=(a+a.T)/2;b=x.T@(target*pi[:,None]);assert torch.allclose(b[-1,:4],torch.full((4,),.25,dtype=b.dtype),atol=1e-12,rtol=0)
  w=torch.linalg.solve(hessian(a),b);res=float((hessian(a)@w-b).abs().max());write(f'{k}.SOLVE.json',dict(residual=res,alpha=.01));assert res<=1e-7 and torch.isfinite(w).all() and torch.equal(w[:,4:],torch.zeros_like(w[:,4:]))
  ix=torch.triu_indices(1536,1536);artifact=dict(mapper=mp,PERM=torch.randperm(1536,generator=torch.Generator().manual_seed(74010)),A=a[ix.unbind()].contiguous(),B=b,W=w,alpha=.01,class_ids=list(range(6)),original_frozen_encoder=True,bias_updates=0,source_commit=cfg['source_commit']);p=out/f'{k}.artifact.private.pt';torch.save(artifact,p);sizes[str(k)]=p.stat().st_size;assert sizes[str(k)]<=19058800
  back=load(p);assert torch.equal(back['A'],artifact['A']) and torch.equal(back['B'],b) and torch.equal(back['W'],w) and torch.equal(back['mapper']['ids'],mp['ids']) and all(torch.equal(xx,zz) for n in ['gn','on'] for xx,zz in zip(mp[n],back['mapper'][n]))
  parity=float(((x[:32]@w[:,:4])-(torch.cat([transform(g[take][:32],back['mapper']),torch.ones(min(32,len(z)),1,dtype=z.dtype)],1)@back['W'][:,:4])).abs().max());assert parity<=1e-10
  prepared.append((mp,w,teacher['W']));fits.append(dict(producer=k,alpha=.01,residual=res,G_residual=resg,G_stat_error=err,support_rows=len(yy),readback_score_error=parity));write(f'{k}.HEAD_SEAL.json',dict(query_excluded=True,artifact_bytes=sizes[str(k)],residual=res))
 write('MODEL_SEAL.json',dict(actual_solves=2,artifact_bytes=sizes,all_query_risk_pending=True,encoder_forwards=0,mapper_fits=0,alpha=.01))
 previous=load(old/'evaluate/responses.private.pt');assert torch.equal(previous['y'],y) and previous['groups']==groups
 score={n:torch.zeros(2555,4,dtype=torch.float64) for n in ['G','I']};coverage=torch.zeros(2555,dtype=torch.long);parity=0.
 for k,(mp,w,fw) in enumerate(prepared):
  take=fold!=k;z=transform(g[take],mp);score['I'][take]=z@w[:-1,:4]+w[-1,:4];score['G'][take]=z@fw[:-1]+fw[-1];coverage[take]+=1;parity=max(parity,float((score['G'][take]-previous['score']['F_out'][take]).abs().max()))
 write('QUERY_PARITY.json',dict(maxabs=parity));assert parity<=1e-8 and (coverage==1).all() and all(torch.isfinite(v).all() for v in score.values())
 tab={n:points(s,y,groups) for n,s in score.items()};folds={str(k):{n:points(s[fold!=k],y[fold!=k],[groups[j] for j in torch.where(fold!=k)[0]]) for n,s in score.items()} for k in [0,1]};paired=pairs(score['G'],score['I'],y);assert tab['G']['correct']==[237,13,64,1626]
 gates=gate(tab,folds,paired);ci=None;decision='STOP_METRIC_MEASURE_ALIGNMENT'
 if gates['pass_all']:
  ci=bootstrap(score['G'].argmax(1),score['I'].argmax(1),y,groups);decision='DEVELOPMENT_METRIC_MEASURE_SIGNAL' if all(v['adjusted_CI'][0]>0 for v in ci.values()) else 'STOP_METRIC_MEASURE_EVIDENCE_INSUFFICIENT'
 perpair={str(k):pairs(score['G'][fold!=k],score['I'][fold!=k],y[fold!=k]) for k in [0,1]}
 write('aggregate.json',dict(decision=decision,method_success=False,summary=tab,per_fold=folds,paired_vs_G=paired,per_fold_paired=perpair,gate=gates,bootstrap=ci,bootstrap_performed=ci is not None,measure=measure,fits=fits,actual_solves=2,artifact_bytes=sizes,baseline_score_parity=parity,baseline_count_exact=True,query_coverage_once=True,encoder_forwards=0,mapper_fits=0,parameter_updates=0,development_used=False,future_used=False,test_used=False,source_commit=cfg['source_commit']))
start=time.time();code=1
try:main();code=0
except BaseException as e:
 write('WORKER_FAILURE.json',dict(type=type(e).__name__,no_automatic_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_EXIT.json',dict(exit_code=code));write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()))
raise SystemExit(code)
