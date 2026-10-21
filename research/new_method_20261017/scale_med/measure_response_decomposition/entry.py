"""Exact finite normal-equation response; no new classifier fitting or intervention scoring."""
import os,json,time,resource,traceback,sys
from pathlib import Path
from collections import Counter
import torch
from metrics import transform,same_mapper,hessian,qweights,points,pairs
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(4)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def stat(v):
 if not len(v):return dict(count=0,mean=None,quantiles=None)
 assert torch.isfinite(v).all();return dict(count=len(v),mean=float(v.mean()),quantiles=torch.quantile(v,torch.tensor([0.,.1,.5,.9,1.],dtype=v.dtype)).tolist())
def pattern(v,total):
 if v is None or total is None:return dict(available=False,pass_all=False,negative_share=None)
 mass=sum(max(0.,-x) for x in v)
 if mass==0:return dict(available=False,pass_all=False,negative_share=None)
 share=max(0.,-v[2])/mass
 return dict(available=True,negative_share=share,pass_all=total<0 and all(v[2]<v[j] for j in [0,1,3]) and share>.5)
def synthetic():
 h=torch.tensor([[2.,1.],[1.,2.]],dtype=torch.float64);d1=torch.tensor([[1.,-1.],[2.,0.]],dtype=h.dtype);d2=-d1/2
 u=torch.linalg.solve(h,torch.cat([h@d1,h@d2],1));assert torch.allclose(u[:,:2]+u[:,2:],d1+d2,atol=1e-12,rtol=0)
 assert pattern([-.1,0.,-1.,.1],-1.)['pass_all'];assert not pattern([-1.,0.,-.1,.1],-1.)['pass_all'];assert not pattern([0.]*4,0.)['available'];assert not pattern(None,None)['available']
 write('SYNTHETIC.json',dict(pass_all=True,toy_system_solves=1,component_sum_and_pattern_success_failure_missing=True))
def unpack(a):
 d=1536;ix=torch.triu_indices(d,d);m=torch.zeros(d,d,dtype=torch.float64);m[ix.unbind()]=a;m[ix[1],ix[0]]=a;return m
BUCKETS=[(c,sgn) for c in range(4) for sgn in [-1,0,1]]
def main():
 synthetic();old=Path(cfg['producer_root']);prior=Path(cfg['measure_root']);allowed={old/'prep/PLAN.private.pt',old/'evaluate/responses.private.pt'}
 for k in [0,1]:allowed.update({old/'prep'/f'{k}.teacher.private.pt',prior/f'{k}.artifact.private.pt'})
 original=torch.load
 def load(p):
  p=Path(p);assert p in allowed,'ASSET_DENIED';return original(p,map_location='cpu',weights_only=False)
 torch.load=load;window=Path(cfg['window_root']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(window) and not p.is_relative_to(out):assert p in allowed,'NONALLOWLIST_PRIVATE_ASSET'
 sys.addaudithook(guard)
 plan=load(old/'prep/PLAN.private.pt');y=plan['y'];groups=plan['groups'];fold=plan['fold'];g=plan['cached_g'];assert len(y)==2555
 assert not {groups[j] for j in torch.where(fold==0)[0]} & {groups[j] for j in torch.where(fold==1)[0]}
 prepared=[];checks=[];counts=[]
 for k in [0,1]:
  take=fold==k;yy=y[take];gg=[groups[j] for j in torch.where(take)[0]];teacher=load(old/'prep'/f'{k}.teacher.private.pt');art=load(prior/f'{k}.artifact.private.pt');mp=teacher['mapper'];assert same_mapper(mp,art['mapper'])
  z=transform(g[take],mp);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);t=torch.nn.functional.one_hot(yy,6).double();pg=qweights(yy,gg);nc=torch.bincount(yy,minlength=4);pi=1/(4*nc[yy].double());dp=pi-pg
  ag=teacher['A'];bg=teacher['B'];ai=unpack(art['A']);bi=art['B'];wg=torch.cat([teacher['W'],torch.zeros(1536,2,dtype=z.dtype)],1);wi=art['W'];hi=hessian(ai)
  a0=x.T@(x*pg[:,None]);a0=(a0+a0.T)/2;a1=x.T@(x*pi[:,None]);a1=(a1+a1.T)/2
  staterr=max(float((a0-ag).abs().max()),float((a1-ai).abs().max()),float((x.T@(t*pg[:,None])-bg).abs().max()),float((x.T@(t*pi[:,None])-bi).abs().max()))
  normal=max(float((hessian(ag)@wg-bg).abs().max()),float((hi@wi-bi).abs().max()))
  m=Counter(gg);gc={c:len({gg[j] for j in torch.where(yy==c)[0]}) for c in range(4)};sign=torch.tensor([0 if gc[int(c)]*m[key]==int(nc[c]) else 1 if gc[int(c)]*m[key]>int(nc[c]) else -1 for c,key in zip(yy,gg)])
  residual=t-x@wg;rhs=[];binfo=[]
  for c,sgn in BUCKETS:
   subset=(yy==c)&(sign==sgn);rhs.append(x.T@(residual*(dp*subset)[:,None]));binfo.append(dict(source_class=c,weight_direction=sgn,support_rows=int(subset.sum()),signed_weight_mass=float(dp[subset].sum())))
  rb=bi-bg;ra=-(ai-ag)@wg;rhs_error=float((sum(rhs)-rb-ra).abs().max());allrhs=torch.cat(rhs+[rb,ra],1)
  u=torch.linalg.solve(hi,allrhs).reshape(1536,14,6)
  solve_error=float((hi@u.reshape(1536,84)-allrhs).abs().max());closure=max(float((u[:,:12].sum(1)-(wi-wg)).abs().max()),float((u[:,12]+u[:,13]-(wi-wg)).abs().max()))
  check=dict(producer=k,support_stat_error=staterr,normal_residual=normal,RHS_sum_error=rhs_error,response_system_residual=solve_error,deltaW_reconstruction_error=closure)
  write(f'{k}.ALGEBRA_CHECK.json',check);assert staterr<=1e-10 and normal<=1e-7 and rhs_error<=1e-10 and solve_error<=1e-8 and closure<=1e-8 and torch.isfinite(u).all()
  prepared.append((mp,wg,wi,u));checks.append(check);counts.append(dict(producer=k,buckets=binfo))
 write('COMPONENT_SEAL.json',dict(producers=2,source_buckets_per_producer=12,statistic_components_per_producer=2,real_response_system_solves=2,toy_system_solves=1,new_classifier_fits=0,query_not_used_for_components=True,all_components_finite=True,source_commit=cfg['source_commit']))
 previous=load(old/'evaluate/responses.private.pt');expected=json.loads((prior/'aggregate.json').read_text());score={n:torch.zeros(2555,4,dtype=torch.float64) for n in ['G','I']};cover=torch.zeros(2555,dtype=torch.long);cells=[];main_cells=[];maxerr=0.;parity=0.
 for k,(mp,wg,wi,u) in enumerate(prepared):
  take=fold!=k;yy=y[take];z=transform(g[take],mp);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);sg=x@wg[:,:4];si=x@wi[:,:4];parts=(x@u.reshape(1536,84)).reshape(len(x),14,6);source=parts[:,:12].reshape(len(x),4,3,6).sum(2)
  score['G'][take]=sg;score['I'][take]=si;cover[take]+=1;parity=max(parity,float((sg-previous['score']['F_out'][take]).abs().max()));assert parity<=1e-8
  gok=sg.argmax(1)==yy;iok=si.argmax(1)==yy
  for c in range(4):
   for j in range(4):
    if c==j:continue
    mg=sg[:,c]-sg[:,j];mi=si[:,c]-si[:,j];move=mi-mg;ss=source[:,:,c]-source[:,:,j];pp=parts[:,:,c]-parts[:,:,j]
    err=max(float((move-ss.sum(1)).abs().max()),float((move-pp[:,12]-pp[:,13]).abs().max()));maxerr=max(maxerr,err)
    for label,mask in [('ALL',yy==c),('G_correct',(yy==c)&gok),('I_new_harm',(yy==c)&gok&~iok)]:
     row=dict(producer=k,class_id=c,rival=j,subset=label,G_margin=stat(mg[mask]),I_margin=stat(mi[mask]),total_move=stat(move[mask]),source_classes=[stat(ss[mask,v]) for v in range(4)],buckets=[dict(source_class=cc,weight_direction=b,summary=stat(pp[mask,v])) for v,(cc,b) in enumerate(BUCKETS)],target_term=stat(pp[mask,12]),Gram_term=stat(pp[mask,13]))
     cells.append(row)
     if c==0 and j==2 and label in ['ALL','G_correct']:
      means=[v['mean'] for v in row['source_classes']];main_cells.append(dict(producer=k,subset=label,total_move=row['total_move']['mean'],source_class_means=means,pattern=pattern(means,row['total_move']['mean'])))
  write(f'{k}.QUERY_ALGEBRA.json',dict(margin_sum_error=maxerr,F_query_score_error=parity));assert maxerr<=1e-8
 assert len(cells)==72 and (cover==1).all() and len(main_cells)==4
 tab={n:points(s,y,groups) for n,s in score.items()};folds={str(k):{n:points(s[fold!=k],y[fold!=k],[groups[j] for j in torch.where(fold!=k)[0]]) for n,s in score.items()} for k in [0,1]}
 for n in ['G','I']:
  assert tab[n]['correct']==expected['summary'][n]['correct'] and abs(tab[n]['BA']-expected['summary'][n]['BA'])<=1e-12
  for k in ['0','1']:assert folds[k][n]['correct']==expected['per_fold'][k][n]['correct']
 paired=pairs(score['G'],score['I'],y);perpaired={str(k):pairs(score['G'][fold!=k],score['I'][fold!=k],y[fold!=k]) for k in [0,1]};contributions={}
 for k,v in folds.items():
  cc=[25*(a-b)/n for a,b,n in zip(v['I']['correct'],v['G']['correct'],v['G']['counts'])];assert abs(sum(cc)-100*(v['I']['BA']-v['G']['BA']))<=1e-10;contributions[k]=cc
 available=all(v['pattern']['available'] for v in main_cells);passed=all(v['pattern']['pass_all'] for v in main_cells);decision='UNRESOLVED_PATTERN' if not available else 'CLASS2_SOURCE_DOMINANT_PATTERN' if passed else 'MIXED_OR_NON_CLASS2_SOURCE_PATTERN'
 write('aggregate.json',dict(decision=decision,method_success=False,original_performance_decision='STOP_METRIC_MEASURE_ALIGNMENT',original_risk_gate=expected['gate'],bootstrap_performed=False,primary_pattern_cells=main_cells,summary=tab,per_fold=folds,paired_vs_G=paired,per_fold_paired=perpaired,BA_contribution_pp_by_fold=contributions,cells=cells,cell_count=72,checks=checks,support_buckets=counts,query_margin_sum_max_error=maxerr,F_query_score_error=parity,original_endpoint_counts_exact=True,real_response_system_solves=2,toy_system_solves=1,new_classifier_fits=0,encoder_forwards=0,parameter_updates=0,causal_identification=False,development_used=False,future_used=False,test_used=False,source_commit=cfg['source_commit']))
start=time.time();code=1
try:main();code=0
except BaseException as e:
 write('WORKER_FAILURE.json',dict(type=type(e).__name__,no_automatic_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_EXIT.json',dict(exit_code=code));write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()))
raise SystemExit(code)
