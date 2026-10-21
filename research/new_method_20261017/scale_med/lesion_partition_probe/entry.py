import os,json,time,resource,traceback
from pathlib import Path
from collections import defaultdict,Counter
import torch
from metrics import transform,qweights,hessian
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(4)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def assignments(y,groups,rng):
 by=defaultdict(list)
 for i,g in enumerate(groups):by[g].append(i)
 strata=defaultdict(list)
 for g,ids in sorted(by.items()):
  assert len(set(y[ids].tolist()))==1
  strata[int(y[ids[0]]),len(ids)].append(ids)
 real=[];pseudo=[];cells=[]
 for (c,m),gg in sorted(strata.items()):
  rr=[i for ids in gg for i in ids];eligible=m>1 and len(gg)>1
  pp=torch.tensor(rr)[torch.randperm(len(rr),generator=rng)].tolist() if eligible else rr
  pg=[pp[i:i+m] for i in range(0,len(pp),m)]
  real+=gg;pseudo+=pg
  changed=sum(set(a)!=set(b) for a,b in zip(gg,pg))
  cells.append(dict(class_id=c,size=m,groups=len(gg),images=len(rr),eligible=eligible,changed_groups=changed))
 return real,pseudo,cells

def mainflag(rows):
 if len(rows)!=2:return False
 return all(v['eligible_changed'] and all(v['real'][key]<v['pseudo'][key]-1e-10*max(1,abs(v['pseudo'][key])) for key in ['feature_trace','score_energy']) for v in rows)
def moments(z,w,part,y,weights):
 centered=w-w.mean(1,keepdim=True);scores=z@centered;M=[];mass=[];cells=[]
 for c in range(4):
  tr=sc=0.;groups=0;multi=0
  for ids in part:
   if int(y[ids[0]])!=c:continue
   zz=z[ids];ss=scores[ids];a=float(weights[ids].sum());mu=zz.mean(0);M.append(mu);mass.append(a)
   tr+=a*float((zz-mu).square().sum()/len(ids));sc+=a*float((ss-ss.mean(0)).square().sum()/len(ids));groups+=1;multi+=len(ids)>1
  cells.append(dict(class_id=c,groups=groups,multi_groups=multi,feature_trace=tr,score_energy=sc))
 mu=torch.stack(M);a=torch.tensor(mass,dtype=z.dtype);S=z.T@(z*weights[:,None])-mu.T@(mu*a[:,None]);S=(S+S.T)/2
 tr=sum(v['feature_trace'] for v in cells);sc=sum(v['score_energy'] for v in cells);err=max(abs(float(S.trace())-tr),abs(float((centered*(S@centered)).sum())-sc));assert err<1e-8
 kap=tr/z.shape[1];iso=torch.eye(z.shape[1],dtype=z.dtype)*kap;act=S@centered
 return dict(feature_trace=tr,score_energy=sc,classes=cells,isotropic_kappa=kap,isotropic_score_energy=float((centered.square().sum())*kap),relative_Frobenius_anisotropy=float(torch.linalg.vector_norm(S-iso)/torch.linalg.vector_norm(S)) if tr>1e-15 else None,contrast_action_departure_from_isotropic=float(torch.linalg.vector_norm(act-iso@centered)/torch.linalg.vector_norm(act)) if float(torch.linalg.vector_norm(act))>1e-15 else None,scatter_quadratic_closure=err)
def synthetic():
 z=torch.tensor([[-1.],[-1.],[1.],[1.]],dtype=torch.float64);y=torch.zeros(4,dtype=torch.long);p=torch.full((4,),.25,dtype=z.dtype);w=torch.tensor([[1.,-1.]],dtype=z.dtype)
 a=moments(z,w,[[0,1],[2,3]],y,p);b=moments(z,w,[[0,2],[1,3]],y,p);assert a['feature_trace']==0 and b['feature_trace']==1 and b['score_energy']==2
 good=dict(real=a,pseudo=b,eligible_changed=True);assert mainflag([good,good]) and not mainflag([good,dict(real=b,pseudo=a,eligible_changed=True)])
 _,_,cells=assignments(torch.tensor([0,0]),['a','b'],torch.Generator().manual_seed(74370));assert not any(v['eligible'] for v in cells)
 write('SYNTHETIC.json',dict(pass_all=True,solves=0,compact_crossed_and_gate_and_singletons=True))
def main():
 synthetic();root=Path(cfg['producer_root']);allowed={root/'prep/PLAN.private.pt',root/'prep/0.teacher.private.pt',root/'prep/1.teacher.private.pt'};orig=torch.load
 def load(p):
  assert Path(p) in allowed,'ASSET_DENIED'
  return orig(p,map_location='cpu',weights_only=False)
 torch.load=load
 import sys
 window=Path(cfg['window_root']).resolve()
 def guard(ev,args):
  if ev=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve()
   if p.is_relative_to(window) and not p.is_relative_to(out):assert p in allowed,'PRIVATE_WINDOW_ASSET_DENIED'
 sys.addaudithook(guard)
 plan=load(root/'prep/PLAN.private.pt');assert len(plan['y'])==2555;prepared=[];rng=torch.Generator().manual_seed(74370);seals=[]
 for k in [0,1]:
  take=plan['fold']==k;y=plan['y'][take];groups=[plan['groups'][j] for j in torch.where(take)[0]];real,pseudo,cs=assignments(y,groups,rng);p=qweights(y,groups);pp=torch.zeros_like(p)
  for ids in pseudo:pp[ids]=p[ids[0]]
  assert torch.equal(p,pp) and sorted(i for ids in pseudo for i in ids)==list(range(len(y)))
  prepared.append((take,y,p,real,pseudo,cs));seals.append(dict(producer=k,support_rows=len(y),strata=cs,per_row_weights_exact=True))
 write('PLAN_SEAL.json',dict(seed=74370,folds=seals,query_used=False,features_not_compared_yet=True));rows=[]
 for k,(take,y,p,real,pseudo,cs) in enumerate(prepared):
  teacher=load(root/'prep'/f'{k}.teacher.private.pt');z=transform(plan['cached_g'][take],teacher['mapper']);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);a=x.T@(x*p[:,None]);a=(a+a.T)/2;b=x.T@(torch.nn.functional.one_hot(y,6).double()*p[:,None]);err=max(float((a-teacher['A']).abs().max()),float((b-teacher['B']).abs().max()));res=float((hessian(teacher['A'])@teacher['W']-teacher['B'][:,:4]).abs().max());write(f'{k}.PARITY.json',dict(A_B_error=err,F_residual=res));assert err<=1e-10 and res<=1e-7
  rr=moments(z,teacher['W'][:-1],real,y,p);ss=moments(z,teacher['W'][:-1],pseudo,y,p);elig=[dict(class_id=c,eligible_groups=sum(v['groups'] for v in cs if v['class_id']==c and v['eligible']),changed_groups=sum(v['changed_groups'] for v in cs if v['class_id']==c),comparison_status='EVALUABLE' if any(v['class_id']==c and v['eligible'] and v['changed_groups'] for v in cs) else 'NOT_EVALUABLE') for c in range(4)]
  rows.append(dict(producer=k,real=rr,pseudo=ss,eligibility=elig,eligible_changed=any(v['changed_groups'] for v in elig),parity=dict(A_B_error=err,F_residual=res)))
 write('aggregate.json',dict(decision='SUPPORT_MEMBERSHIP_COMPACTNESS_PATTERN' if mainflag(rows) else 'MIXED_OR_NO_COMPACTNESS_PATTERN',folds=rows,primary_pass=mainflag(rows),method_success=False,query_scored=False,class1_directly_evaluable_both=all(v['eligibility'][1]['comparison_status']=='EVALUABLE' for v in rows),actual_fits=0,actual_solves=0,source_commit=cfg['source_commit'],clinical_noise_identified=False,no_automatic_performance_followup=True))
start=time.time();code=1
try:main();code=0
except BaseException as e:
 write('FAILURE.json',dict(error_type=type(e).__name__,no_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()));write('WORKER_EXIT.json',dict(exit_code=code))
raise SystemExit(code)
