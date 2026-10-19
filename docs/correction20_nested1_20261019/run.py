"""Final-20 nested training-only correction. No image or validation loader exists here."""
import json,os,resource,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from core import weights,targets,loss,spectrum,matched_alpha,metrics


def write(p,obj):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)


def contrasts(scores,y,g):
 p={k:v.argmax(1)==y for k,v in scores.items()};w=weights(y,g);w/=w.sum();iw=1/(20*np.bincount(y,minlength=20)[y]);out={}
 for arm in scores:
  for ref in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction']:
   if arm==ref:continue
   rescue=p[arm]&~p[ref];harm=~p[arm]&p[ref]
   out[arm+'-'+ref]=dict(rescue=int(rescue.sum()),harm=int(harm.sum()),delta_BA=float(iw@(rescue.astype(float)-harm)),delta_PP=float(w@(rescue.astype(float)-harm)),PP_rescue=float(w@rescue),PP_harm=float(w@harm))
 c=p['cls768_ridge'];j=p['scale_joint'];masks={'both_correct':c&j,'CLS_only':c&~j,'joint_only':~c&j,'both_wrong':~c&~j}
 return out,{name:dict(images=int(m.sum()),correct={arm:int((ok&m).sum()) for arm,ok in p.items()}) for name,m in masks.items()}


RUNTIME={}
def main(cfg):
 torch.set_num_threads(cfg['CPU_threads']);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
 out=Path(cfg['output']);out.mkdir(exist_ok=False);source=Path(cfg['cache_run']);start=time.time();cost=dict(new_encodes=0,ridge_attempts=0,ridge_completed=0,correction_attempts=0,correction_completed=0,closure_evaluations=0,attempts=1)
 RUNTIME.update(start=start,cost=cost);write(out/'config.private.json',cfg)
 def status(phase,**kw):
  elapsed=time.time()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;gpu=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
  state=dict(state='RUNNING',phase=phase,started=start,updated=time.time(),wall_seconds=elapsed,costs=cost,RSS_peak_bytes=rss,GPU_peak_reserved_bytes=gpu,**kw);write(out/'status.private.json',state)
  assert elapsed<cfg['max_wall_seconds'] and cfg['source_research_measured_seconds']+cfg['engineering_seconds']+elapsed<cfg['total_recorded_wall_cap']
  assert rss<=cfg['RSS_budget_bytes'] and gpu<=cfg['GPU_reserved_budget_bytes']
  assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=cfg['output_budget_bytes']
  assert cost['ridge_attempts']<=9 and cost['correction_attempts']<=6 and cost['new_encodes']==0
 status('bind_training_cache')
 assert json.loads((source/'status.private.json').read_text())['state']=='COMPLETE'
 identity=json.loads((source/'config.private.json').read_text());assert identity['seed']==cfg['seed'] and identity['order']==cfg['order'];del identity
 ix=np.load(source/'train_index.private.npz');y=ix['labels'].astype(int);g=ix['groups'];f=ix['folds'];masses=ix['weights'];bounds=ix['bounds'];cache=np.load(source/'train768.private.npy',mmap_mode='r')
 assert cache.shape==(68058,1536) and np.array_equal(np.unique(y),np.arange(20)) and np.array_equal(np.unique(f),np.arange(3))
 order=np.argsort(g,kind='stable');assert np.all((g[order][1:]!=g[order][:-1])|(f[order][1:]==f[order][:-1]))
 assert np.max(np.abs(masses-weights(y,g)))<1e-14
 counts=np.array([[len(np.unique(g[(y==c)&(f==h)])) for h in range(3)] for c in range(20)]);assert counts.min()>0
 write(out/'FOLD_AUDIT.json',dict(images=len(y),patients=len(np.unique(g)),class_fold_patients=counts.tolist(),class_fold_images=[[int(((y==c)&(f==h)).sum()) for h in range(3)] for c in range(20)],validation_access=False))
 grams=torch.zeros(20,3,1537,1537,dtype=torch.float64);sums=torch.zeros(20,3,1537,dtype=torch.float64)
 for c in range(20):
  for lo in range(int(bounds[c]),int(bounds[c+1]),2048):
   stop=min(lo+2048,int(bounds[c+1]));v=torch.cat([torch.from_numpy(np.array(cache[lo:stop],copy=True)).double(),torch.ones(stop-lo,1,dtype=torch.float64)],1)
   for h in range(3):
    mask=f[lo:stop]==h;x=v[mask];w=torch.from_numpy(masses[lo:stop][mask]);grams[c,h]+=x.T@(x*w[:,None]);sums[c,h]+=(x*w[:,None]).sum(0)
  status('training_moments',class_index=c)
 ci=torch.tensor(list(range(768))+[1536]);eyeC=torch.eye(769,dtype=torch.float64);eyeJ=torch.eye(1537,dtype=torch.float64)
 def moment(selected):
  ratio=torch.from_numpy(counts.sum(1)/counts[:,selected].sum(1));cg=grams[:,selected].sum(1)*ratio[:,None,None];b=(sums[:,selected].sum(1)*ratio[:,None]).T
  raw=cg.sum(0);base=cg[:10].sum(0);scale=float(base[:768,:768].trace()/base[768:1536,768:1536].trace())**.5
  fac=torch.ones(1537,dtype=torch.float64);fac[:768]=2**-.5;fac[768:1536]=scale*2**-.5
  return raw,b,fac
 def ridge(a,b):
  cost['ridge_attempts']+=1;status('ridge_fit');v=torch.linalg.solve(a,b);cost['ridge_completed']+=1
  residual=float((a@v-b).abs().max()/max(1.,float(b.abs().max())));assert residual<1e-9
  return v,residual
 def score(ids,head,which='joint'):
  values=np.empty((len(ids),20))
  for off in range(0,len(ids),2048):
   rows=ids[off:off+2048];x=torch.cat([torch.from_numpy(np.array(cache[rows],copy=True)).double(),torch.ones(len(rows),1,dtype=torch.float64)],1)
   values[off:off+len(rows)]=(x[:,ci]@head['CLS'] if which=='CLS' else (x*head['factors'])@head[which]).numpy()
  return values
 singleton={};bindings=[]
 for h in range(3):
  raw,b,fac=moment([h]);hc,rc=ridge(raw[ci][:,ci]+eyeC,b[ci]);hj,rj=ridge(raw*fac[:,None]*fac[None,:]+eyeJ,b*fac[:,None]);singleton[h]=dict(CLS=hc,joint=hj,factors=fac)
  torch.save(singleton[h],out/f'inner_head_{h}.private.pt');bindings.append(dict(kind='inner_teacher',training_folds=[h],CLS_residual=rc,joint_residual=rj))
 for outer in range(3):
  allowed=[h for h in range(3) if h!=outer];ids=np.flatnonzero(f!=outer);raw,b,fac=moment(allowed);ng=raw*fac[:,None]*fac[None,:];nb=b*fac[:,None]
  head=torch.load(source/f'fold_head_5_{outer}.private.pt',weights_only=True)
  assert float((head['factors']-fac).abs().max())<1e-12
  rc=float(((raw[ci][:,ci]+eyeC)@head['CLS']-b[ci]).abs().max()/max(1.,float(b[ci].abs().max())))
  rj=float(((ng+eyeJ)@head['joint']-nb).abs().max()/max(1.,float(nb.abs().max())));assert max(rc,rj)<1e-9
  evc=spectrum(raw[ci][:,ci]);df=float((evc/(evc+1)).sum());alpha=matched_alpha(spectrum(ng),df);hd,rd=ridge(ng+alpha*eyeJ,nb)
  innerC=np.empty((len(ids),20));innerJ=innerC.copy()
  for hold in allowed:
   train=next(h for h in allowed if h!=hold);mask=f[ids]==hold;innerC[mask]=score(ids[mask],singleton[train],'CLS');innerJ[mask]=score(ids[mask],singleton[train])
  ys=y[ids];groups=g[ids];ww=weights(ys,groups);repair,keep,target,temp,cap=targets(innerC,innerJ,ys,ww);ww/=ww.sum()
  np.savez(out/f'inner_{outer}.private.npz',rows=ids,CLS=innerC,joint=innerJ,repair=repair,keep=keep,target=target,temperature=temp,cap=cap,weights=ww)
  bindings.append(dict(kind='outer_anchor',outer=outer,training_folds=allowed,reused=True,CLS_residual=rc,joint_residual=rj,df_residual=rd,df_target=df,matched_alpha=alpha,temperature=temp,keep_cap=cap,repair_images=int(repair.sum()),keep_images=int(keep.sum())))
  pre=1/torch.sqrt(torch.diag(ng)/20+2*cfg['regularizer']);z=torch.empty((len(ids),1537),dtype=torch.float64,device='cuda')
  for off in range(0,len(ids),2048):
   rows=ids[off:off+2048];z[off:off+len(rows),:1536]=torch.from_numpy(np.array(cache[rows],copy=True)).to(device='cuda',dtype=torch.float64);z[off:off+len(rows),1536]=1
  z*=fac.cuda();base=z@head['joint'].cuda()/temp;xp=z*pre.cuda();del z
  ty=torch.from_numpy(ys).cuda();tw=torch.from_numpy(ww).cuda();tr=torch.from_numpy(repair.astype(float)).cuda();tk=torch.from_numpy(keep.astype(float)).cuda();tt=torch.from_numpy(target).cuda();tp=pre.cuda();audits=[];models=dict(CLS=head['CLS'],joint=head['joint'],df_joint=hd,factors=fac,temperature=temp,precondition=pre,matched_alpha=alpha)
  for utility in [False,True]:
   arm='utility_correction' if utility else 'plain_correction';cost['correction_attempts']+=1;status('correction_fit',outer=outer,arm=arm);fitstart=time.time();last=[None]
   def fun(vector):
    cost['closure_evaluations']+=1;status('correction_fit',outer=outer,arm=arm)
    u=torch.tensor(vector.reshape(1537,20),dtype=torch.float64,device='cuda',requires_grad=True);v=loss(u,xp,base,ty,tw,tr,tk,tt,tp,utility,cfg['regularizer']);v.backward();grad=u.grad.detach().cpu().numpy().ravel().copy();value=float(v.detach());assert np.isfinite(value) and np.isfinite(grad).all();last[0]=value
    return value,grad
   initial=np.zeros(1537*20);initial_loss,_=fun(initial);res=minimize(fun,initial,jac=True,method='L-BFGS-B',options={k:cfg[k] for k in ['maxiter','maxfun','maxls','maxcor','ftol','gtol']})
   audit=dict(outer=outer,arm=arm,success=bool(res.success),status=int(res.status),message=str(res.message),iterations=int(res.nit),function_calls=int(res.nfev),initial_objective=initial_loss,final_objective=float(res.fun),gradient_max=float(np.abs(res.jac).max()),wall_seconds=time.time()-fitstart)
   audits.append(audit);write(out/f'optimization_{outer}.json',audits);np.save(out/f'last_candidate_{outer}_{arm}.private.npy',res.x)
   assert res.success and np.isfinite(res.x).all() and res.fun<=initial_loss+1e-9,'Optimizer failed fixed acceptance; no retry'
   models[arm]=pre[:,None]*torch.from_numpy(res.x.reshape(1537,20));cost['correction_completed']+=1
   print(json.dumps(dict(event='correction_complete',**audit)),flush=True)
  torch.save(models,out/f'outer_{outer}.private.pt');write(out/'BINDINGS.json',bindings);status('outer_models_frozen',outer=outer)
  del xp,base,ty,tw,tr,tk,tt,tp;torch.cuda.empty_cache()
 assert cost['ridge_completed']==9 and cost['correction_completed']==6
 write(out/'FIT_SEAL.json',dict(frozen_at=time.time(),outer_models=3,correction_heads=6,validation_access=False,new_encodes=0))
 arms=['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction'];allscore={a:np.empty((len(y),20)) for a in arms};outermetrics=[]
 for outer in range(3):
  status('outer_evaluation',outer=outer);ids=np.flatnonzero(f==outer);h=torch.load(out/f'outer_{outer}.private.pt',weights_only=True);scores={'cls768_ridge':score(ids,h,'CLS'),'scale_joint':score(ids,h),'df_matched_joint':score(ids,h,'df_joint')}
  for arm in arms[3:]:scores[arm]=score(ids,dict(h,joint=h['joint']/h['temperature']+h[arm]))
  for arm,v in scores.items():allscore[arm][ids]=v
  cs,gs=contrasts(scores,y[ids],g[ids]);outermetrics.append(dict(outer=outer,metrics={a:metrics(v,y[ids],g[ids],cfg['order'],cfg['frequency_bins']) for a,v in scores.items()},contrasts=cs,groups=gs))
 np.savez(out/'outer_predictions.private.npz',rows=np.arange(len(y)),labels=y,groups=g,folds=f,**allscore)
 cs,gs=contrasts(allscore,y,g);write(out/'results.json',dict(scope='training_internal_nested_final20_only',metrics={a:metrics(v,y,g,cfg['order'],cfg['frequency_bins']) for a,v in allscore.items()},contrasts=cs,groups=gs,outer=outermetrics))
 status('final_budget_check');cost.update(wall_seconds=time.time()-start,historical_seconds=cfg['source_research_measured_seconds'],engineering_seconds=cfg['engineering_seconds'],cumulative_measured_seconds=cfg['source_research_measured_seconds']+cfg['engineering_seconds']+time.time()-start,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
 write(out/'costs.json',cost);write(out/'status.private.json',dict(state='COMPLETE',started=start,finished=time.time(),costs=cost,independent_verification='PENDING'));print('COMPLETE',flush=True)


if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
 try:main(cfg)
 except BaseException:
  root=Path(cfg['output']);root.mkdir(exist_ok=True);write(root/'failure.private.json',dict(error=traceback.format_exc(),updated=time.time(),costs=RUNTIME.get('cost',{})));write(root/'status.private.json',dict(state='FAILED',updated=time.time(),wall_seconds=time.time()-RUNTIME.get('start',time.time()),costs=RUNTIME.get('cost',{}),error=traceback.format_exc()));raise
