"""Only sealed scores; exact scalar-threshold states and paired oracle bootstrap."""
import os,json,time,resource,traceback,sys
from pathlib import Path
import numpy as np
import torch
ARMS=['G','P','C'];DRAWS=10000;SEED=74154

def scalar(x):
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def write(p,x):
 encoded=json.dumps(x,indent=2,allow_nan=False,default=scalar);q=p.with_suffix(p.suffix+'.tmp');q.write_text(encoded);q.replace(p)
def metric(pred,y,w,k):
 return np.stack([(w*((pred==c)&(y==c))).sum(1)/(w*(y==c)).sum(1) for c in range(k)],axis=1)
def plan(s,y):
 k=s.shape[1];others=s.copy();others[:,1]=-np.inf;j=others.argmax(1);r=s[:,1]-others.max(1);priority=(j==0).astype(int)
 order=np.lexsort((priority,-r));rr=r[order];pp=priority[order]
 allowed=np.r_[True,(rr[:-1]!=rr[1:])|(pp[:-1]!=pp[1:]),True]
 assert len(allowed)==len(y)+1
 return dict(order=order,allowed=allowed,other=j,original=s.argmax(1),k=k)
def envelope(pl,y,w,ref):
 order=pl['order'];k=pl['k'];b=len(w);n=len(y);den=np.stack([(w*(y==c)).sum(1) for c in range(k)],axis=1)
 hits=[]
 for c in range(k):
  if c==1:
   a=w[:,order]*(y[order]==1);hits.append(np.c_[np.zeros(b),np.cumsum(a,axis=1)])
  else:
   a=w*((y==c)&(pl['other']==c));hits.append(a.sum(1)[:,None]-np.c_[np.zeros(b),np.cumsum(a[:,order],axis=1)])
 rec=[hits[c]/den[:,c,None] for c in range(k)];other=sum(rec[c] for c in range(k) if c!=1)/(k-1);ba=sum(rec)/k
 feasible=(other>=np.delete(ref,1,axis=1).mean(1)[:,None]-.01-1e-12)&(ba>=ref.mean(1)[:,None]-1e-12)&pl['allowed'][None,:]
 candidate=np.where(feasible,rec[1],-np.inf);ix=candidate.argmax(1);ok=feasible.any(1);U=candidate[np.arange(b),ix];U[~ok]=np.nan
 chosen=np.stack([z[np.arange(b),ix] for z in rec],axis=1)
 return U,ok,chosen,feasible.sum(1),ix

def rank_fusion(arrays):
 return sum(((s[:,:,None]>s[:,None,:]).sum(2)+.5*((s[:,:,None]==s[:,None,:]).sum(2)-1)) for s in arrays)/len(arrays)
def bootstrap_weights(y,groups,draws,seed):
 rng=np.random.default_rng(seed);w=np.empty((draws,len(y)),dtype=np.float64)
 for c in sorted(set(y.tolist())):
  ids=np.where(y==c)[0];keys=sorted({groups[i] for i in ids});lookup={g:i for i,g in enumerate(keys)};ng=len(keys)
  draw=rng.integers(0,ng,(draws,ng));counts=np.bincount((draw+np.arange(draws)[:,None]*ng).ravel(),minlength=draws*ng).reshape(draws,ng)
  w[:,ids]=counts[:,[lookup[groups[i]] for i in ids]]
 return w

def analyze(scores,ty,groups,out,draws=DRAWS):
 y=ty.numpy();table={};primary={};ref0={};plans0={};names=[a+'_'+m for a in ARMS for m in ['ridge','CE']]+['FUSION_ridge','FUSION_CE']
 for st in range(3):
  k=st+4;mask=y<k;yy=y[mask];ones=np.ones((1,len(yy)));stage={n:scores[n,st].numpy()[mask] for n in names[:6]}
  for m in ['ridge','CE']:stage['FUSION_'+m]=rank_fusion([stage[a+'_'+m] for a in ARMS])
  for n in names:
   arm=n.split('_')[0];pl=plan(stage[n],yy);ref=metric(stage[arm+'_ridge'].argmax(1),yy,ones,k);orig=metric(pl['original'],yy,ones,k)[0];U,ok,rec,nfeas,ix=envelope(pl,yy,ones,ref)
   cell=dict(original_recall=orig.tolist(),original_BA=float(orig.mean()),reference_recall=ref[0].tolist(),reference_BA=float(ref.mean()),status='FEASIBLE' if ok[0] else 'UNRESOLVED_INFEASIBLE',U=None if not ok[0] else float(U[0]),envelope_recall=None if not ok[0] else rec[0].tolist(),envelope_BA=None if not ok[0] else float(rec[0].mean()),other_macro_cost_pp=None if not ok[0] else float(100*(np.delete(rec[0],1).mean()-np.delete(ref[0],1).mean())),feasible_states=int(nfeas[0]),decision_states=int(pl['allowed'].sum()),tail_gain_pp=None if not ok[0] else float(100*(U[0]-orig[1])),can_reach_reference_tail=None if not ok[0] else bool(U[0]>=ref[0,1]-1e-12))
   table.setdefault(n,{})[str(st)]=cell
   if st==0 and arm in ARMS:plans0[n]=pl;ref0[arm]=stage[arm+'_ridge'].argmax(1)
 write(out/'ENVELOPES.json',table)
 yy=y[y<4];gg=[groups[i] for i in np.where(y<4)[0]];w=bootstrap_weights(yy,gg,draws,SEED);boot={n:np.full(draws,np.nan) for n in plans0}
 for begin in range(0,draws,128):
  ww=w[begin:begin+128]
  for n,pl in plans0.items():
   ref=metric(ref0[n.split('_')[0]],yy,ww,4);boot[n][begin:begin+len(ww)]=envelope(pl,yy,ww,ref)[0]
 for a in ARMS:
  ce=table[a+'_CE']['0'];ri=table[a+'_ridge']['0'];values=(boot[a+'_CE']-boot[a+'_ridge'])*100;valid=np.isfinite(values)
  complete=ce['U'] is not None and ri['U'] is not None and bool(valid.all())
  primary[a]=dict(delta_U_pp=None if ce['U'] is None or ri['U'] is None else 100*(ce['U']-ri['U']),feasible_paired_draws=int(valid.sum()),total_draws=draws,CI95_pp=np.quantile(values,[.025,.975]).tolist() if complete else None,CI3_adjusted_pp=np.quantile(values,[.05/6,1-.05/6]).tolist() if complete else None,gate=None if not complete else bool(100*(ce['U']-ri['U'])<=-5 and np.quantile(values,1-.05/6)<0))
 complete=all(x['gate'] is not None for x in primary.values());passed=complete and all(x['gate'] for x in primary.values());decision='UNRESOLVED_INFEASIBILITY_NO_EXTENSION' if not complete else ('RANK_DAMAGE_NECESSARY_GATE_PASSED' if passed else 'STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION')
 result=dict(status='COMPLETE',envelopes=table,primary=primary,primary_gate=passed if complete else None,decision=decision,method_success=False,oracle_only=True,deployable_gain=False,bootstrap=dict(draws=draws,seed=SEED,family=3,reoptimized_every_draw=True,infeasible_draws_not_dropped=True),prediction_cells=18,diagnostic_cells=24,no_training=True,no_features=True,no_head_forward=True,historical_test_exposed=True)
 write(out/'aggregate.json',result);return result

def selfcheck(out):
 rng=np.random.default_rng(3)
 for case in range(3):
  s=np.round(rng.normal(size=(24,4)),1);y=np.arange(24)%4;pl=plan(s,y);w=np.stack([np.ones(24),np.repeat([1,2,1,2,1,2],4)])
  ref=metric(s.argmax(1),y,w,4);U,ok,rec,nf,ix=envelope(pl,y,w,ref)
  others=s.copy();others[:,1]=-np.inf;j=others.argmax(1);r=s[:,1]-others.max(1);direct=[]
  direct.append(j.copy())
  for t in sorted(set(r.tolist()),reverse=True):
   direct.append(np.where((r>t)|((r==t)&(j>1)),1,j));direct.append(np.where(r>=t,1,j))
  direct.append(np.ones(len(y),dtype=int));vectors={tuple(v) for v in direct};prefix=set()
  for i in np.where(pl['allowed'])[0]:
   pred=j.copy();pred[pl['order'][:i]]=1;prefix.add(tuple(pred))
  assert prefix==vectors and tuple(s.argmax(1)) in prefix
  best=np.full(2,-np.inf)
  for v in vectors:
   rr=metric(np.array(v),y,w,4);feas=(np.delete(rr,1,1).mean(1)>=np.delete(ref,1,1).mean(1)-.01-1e-12)&(rr.mean(1)>=ref.mean(1)-1e-12);best=np.maximum(best,np.where(feas,rr[:,1],-np.inf))
  assert np.allclose(U,best)
 # All class scores tied, including class0-vs-class1 tie boundary.
 s=np.zeros((8,4));y=np.arange(8)%4;pl=plan(s,y);assert int(pl['allowed'].sum())==2
 perfect=np.eye(4)[y];ref=metric(perfect.argmax(1),y,np.ones((1,8)),4);U,ok,*_=envelope(pl,y,np.ones((1,8)),ref);assert not ok[0] and np.isnan(U[0])
 scores={}
 for st in range(3):
  for a in ARMS:
   for m in ['ridge','CE']:scores[a+'_'+m,st]=torch.eye(st+4,dtype=torch.float64)[(torch.arange(6).repeat_interleave(4)%(st+4))]
 ty=torch.arange(6).repeat_interleave(4);gg=['g'+str(i//2) for i in range(24)];sub=out/'complete';sub.mkdir();r=analyze(scores,ty,gg,sub,draws=10000);assert len(r['primary'])==3 and len(r['envelopes'])==8
 assert all(x['feasible_paired_draws']==10000 for x in r['primary'].values())
 f=out/'strict.json';write(f,dict(b=np.bool_(True)));before=f.read_text()
 try:write(f,dict(n=float('nan')))
 except ValueError:pass
 else:raise AssertionError('NaN accepted')
 assert f.read_text()==before
 inj=out/'injected';inj.mkdir()
 def fail():write(inj/'checkpoint.json',dict(saved=True));raise RuntimeError('synthetic only')
 assert guarded(fail,inj) is None and json.loads((inj/'EXIT.json').read_text())['exit_code']==1 and (inj/'COST_RECEIPT.json').exists()
 result=dict(state='PASSED',synthetic_only=True,exact_tie_states=True,prefix_equals_direct_enumeration=True,zero_bias_state_present=True,infeasible_returns_null=True,all_three_primary_comparisons=True,reoptimized_bootstrap=True,strict_JSON=True,failure_receipts=True,GPU_initialized=torch.cuda.is_initialized());assert not result['GPU_initialized'];write(out/'PRECHECK.json',result);return result

def real(cfg,out):
 source=Path(cfg['predictions']).resolve();allowed={source,Path(cfg['prior_aggregate']).resolve()}
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(out):return
  if p.suffix.lower() in ['.pt','.pth','.safetensors','.jpg','.png','.jpeg','.npy','.npz'] or p.name=='aggregate.json':assert p in allowed,'outside sealed inputs'
 sys.addaudithook(guard)
 obj=torch.load(source,map_location='cpu',weights_only=False);ty=obj['y'];gg=obj['groups'];scores=obj['scores'];assert len(ty)==586 and len(set(gg))==500 and int((ty==1).sum())==100 and len({gg[i] for i in torch.where(ty==1)[0].tolist()})==70
 assert len(scores)==18 and all(s.dtype==torch.float64 and s.shape==(586,st+4) and bool(torch.isfinite(s).all()) for (n,st),s in scores.items())
 r=analyze(scores,ty,gg,out);prior=json.loads(Path(cfg['prior_aggregate']).read_text())
 for n in prior['summary']:
  for st,m in prior['summary'][n].items():assert np.allclose(r['envelopes'][n][st]['original_recall'],m['recall'],atol=0,rtol=0)
 assert not torch.cuda.is_initialized();r.update(source_commit=cfg['source_commit'],prediction_source_commit='53f01d75bd3c1ccb88392f6e078250f3c7fee755',prior_aggregate_commit='54b0e6f0b46e965b135d9fe42b77e0cb0012f10f',original_18_metrics_exact=True,source_shape_and_finite_passed=True,known_groups=500,class1_groups=70,CUDA_initialized=False)
 write(out/'aggregate.json',r);return r

def guarded(action,out):
 start=time.monotonic();code=1
 try:r=action();code=0;return r
 except BaseException as e:
  write(out/'FAILURE.json',dict(error=type(e).__name__,no_retry=True,decision='UNRESOLVED_ENGINEERING_NO_EXTENSION'));(out/'FAILURE.private.txt').write_text(traceback.format_exc());return None
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(out/'COST_RECEIPT.json',dict(body_wall_seconds=time.monotonic()-start,process_CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()));write(out/'EXIT.json',dict(exit_code=code))

if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);resource.setrlimit(resource.RLIMIT_CPU,(cfg['cpu_budget'],cfg['cpu_budget']));torch.set_num_threads(4)
 result=guarded(lambda:selfcheck(out) if cfg['mode']=='PREFLIGHT' else real(cfg,out),out);sys.exit(0 if result is not None else 1)
