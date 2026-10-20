"""Frozen nested utility predictability probe. No image/validation loading."""
import time
START=time.time()
import os,sys,json,resource,traceback
from pathlib import Path
import numpy as np
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
from core import weights,loss
from solver import solve
from mechanism import ba_weights,features,utility_labels,fit,choose
from runtime import Data,write,rms,summary
COST=dict(attempts=1,new_encodes=0,ridge_attempts=0,ridge_completed=0,correction_attempts=0,correction_completed=0,selector_attempts=0,selector_completed=0,closure_evaluations=0)


def main():
 torch.set_num_threads(cfg['CPU_threads']);out=Path(cfg['output']);out.mkdir(exist_ok=False);parent=Path(cfg['parent_run']);assert json.loads((parent/'VERIFICATION.json').read_text())['state']=='PASS';write(out/'config.private.json',cfg);last_status=[0.]
 def status(phase,force=False,**kw):
  if force or time.time()-last_status[0]>2:
   peaks=dict(RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=0);write(out/'resource_peaks.private.json',peaks);write(out/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),costs=COST,**peaks,**kw));last_status[0]=time.time()
 data=Data(cfg);y,g,fold=data.y,data.g,data.fold;bindings=[];optim=[];coverage=[]
 write(out/'FOLD_AUDIT.json',dict(images=len(y),patients=len(np.unique(g)),class_fold_images=[[int(((y==c)&(fold==h)).sum()) for h in range(3)] for c in range(20)],class_fold_patients=[[len(np.unique(g[(y==c)&(fold==h)])) for h in range(3)] for c in range(20)],no_validation=True))
 for h in range(3):
  ids=np.flatnonzero(fold==h);status('singleton_moments',True,fold=h);raw,b,fac,proto,present=data.moment(ids);old=torch.load(parent/f'inner_head_{h}.private.pt',weights_only=True)
  err=max(float((fac-old['factors']).abs().max()),float(((raw[data.ci][:,data.ci]+torch.eye(769))@old['CLS']-b[data.ci]).abs().max()/max(1.,float(b.abs().max()))),float(((raw*fac[:,None]*fac[None,:]+torch.eye(1537))@old['joint']-b*fac[:,None]).abs().max()/max(1.,float(b.abs().max()))));assert err<1e-9
  COST['ridge_attempts']+=1;hd,alpha,res=data.df(raw,b,fac);COST['ridge_completed']+=1
  head=dict(old,df_joint=hd,matched_alpha=alpha,prototypes=proto,present=present);sub=data.subfold(ids);assert len(np.unique(sub))==2
  tscore=np.empty((len(ids),20))
  for half in range(2):
   train=ids[sub!=half];test=ids[sub==half];status('temperature_teacher',True,fold=h,half=half);rr,bb,ff,pp,pr=data.moment(train);COST['ridge_attempts']+=1;hj,er=data.ridge(rr*ff[:,None]*ff[None,:]+torch.eye(1537),bb*ff[:,None]);COST['ridge_completed']+=1;teacher=dict(joint=hj,factors=ff);torch.save(teacher,out/f'temp_{h}_{half}.private.pt');tscore[sub==half]=data.scores(test,teacher);coverage.append(dict(fold=h,held_half=half,training_images=len(train),held_images=len(test),missing_classes=np.flatnonzero(~pr).tolist(),ridge_residual=er));del rr,bb,ff,pp,pr
  temp=rms(tscore,y[ids],g[ids]);head['temperature']=temp;head['temperature_half']=sub;np.savez(out/f'temperature_{h}.private.npz',rows=ids,scores=tscore,half=sub);write(out/'TEMPERATURE_COVERAGE.json',coverage)
  ww=weights(y[ids],g[ids]);ww/=ww.sum();z=data.x(ids)*fac;gram=z.T@(z*torch.from_numpy(ww)[:,None]);pre=1/torch.sqrt(torch.diag(gram)+2*cfg['regularizer']);xp=z*pre;anchor=z@old['joint']/temp;ty=torch.from_numpy(y[ids]);tw=torch.from_numpy(ww);empty=torch.zeros(len(ids),dtype=torch.float64);proxy=gram/20*pre[:,None]*pre[None,:]+torch.diag(2*cfg['regularizer']*pre**2);COST['correction_attempts']+=1
  def fun(v):
   COST['closure_evaluations']+=1;status('singleton_plain_fit',fold=h);u=torch.tensor(v.reshape(1537,20),dtype=torch.float64,requires_grad=True);val=loss(u,xp,anchor,ty,tw,empty,empty,empty,pre,False,cfg['regularizer']);val.backward();value=float(val.detach());gradient=u.grad.numpy().ravel().copy();assert np.isfinite(value) and np.isfinite(gradient).all();return value,gradient
  def checkpoint(v):
   p=out/f'candidate_{h}.private.npy';tmp=p.with_suffix('.tmp')
   with tmp.open('wb') as handle:np.save(handle,v)
   tmp.replace(p)
  candidate,audit=solve(fun,np.zeros(1537*20),proxy,cfg,checkpoint);checkpoint(candidate);audit.update(fold=h,arm='singleton_plain');optim.append(audit);write(out/'OPTIMIZATION.json',optim);assert audit['success'],'Singleton correction failed fixed numerical acceptance';COST['correction_completed']+=1;head['plain_correction']=pre[:,None]*torch.from_numpy(candidate.reshape(1537,20));head['precondition']=pre;torch.save(head,out/f'singleton_{h}.private.pt');bindings.append(dict(kind='singleton_B_F',training_folds=[h],reused_CLS_joint=True,teacher_check=err,df_residual=res,temperature=temp,missing_classes=np.flatnonzero(~present).tolist()));write(out/'BINDINGS.json',bindings);print(json.dumps(dict(event='singleton_frozen',fold=h,audit=audit)),flush=True);del z,xp,anchor,gram,proxy,raw,b
 assert COST['ridge_completed']==9 and COST['correction_completed']==3
 for outer in range(3):
  status('selector_training_signals',True,outer=outer);ids=np.flatnonzero(fold!=outer);bf=np.empty((len(ids),20));ff=bf.copy();cos=bf.copy();teacher=np.empty(len(ids),dtype=int)
  for hold in [h for h in range(3) if h!=outer]:
   train=next(h for h in range(3) if h not in [outer,hold]);mask=fold[ids]==hold;rows=ids[mask];head=torch.load(out/f'singleton_{train}.private.pt',weights_only=False);bf[mask]=data.scores(rows,head,'df_joint')/head['temperature'];ff[mask]=data.scores(rows,dict(head,joint=head['joint']/head['temperature']+head['plain_correction']));cos[mask]=data.cosine(rows,head);teacher[mask]=train
  x,disagree=features(bf,ff,cos);target=utility_labels(bf,ff,y[ids]);w=ba_weights(y[ids]);np.savez(out/f'signals_{outer}.private.npz',rows=ids,teacher=teacher,B=bf,F=ff,cosine=cos,features=x,disagree=disagree,target=target,weights=w)
  models={}
  for arm,dim in [('score_selector',4),('support_selector',5)]:
   COST['selector_attempts']+=1;status('selector_fit',True,outer=outer,arm=arm);model,audit=fit(x[disagree,:dim],target[disagree],w[disagree],cfg['selector_regularizer'],cfg);optim.append(dict(outer=outer,arm=arm,**audit));write(out/'OPTIMIZATION.json',optim);assert audit['success'],'Selector failed fixed acceptance';models[arm]=model;COST['selector_completed']+=1
  # Prototypes only use this outer training set. No outer predictions yet.
  raw,b,fac,proto,present=data.moment(ids);old=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);assert float((fac-old['factors']).abs().max())<1e-10;models.update(prototypes=proto,present=present,factors=fac);torch.save(models,out/f'selector_{outer}.private.pt');bindings.append(dict(kind='selector',outer=outer,training_folds=[h for h in range(3) if h!=outer],disagreement_images=int(disagree.sum()),utility_counts=np.bincount(target[disagree],minlength=3).tolist(),train_label_weight='class_image_BA',dimensions=[4,5]));write(out/'BINDINGS.json',bindings);del raw,b
 assert COST['selector_completed']==6;write(out/'FIT_SEAL.json',dict(frozen_at=time.time(),selectors=6,singleton_corrections=3,new_ridge_heads=9,new_encodes=0,outer_prediction_read_before_seal=False))
 previous=np.load(parent/'outer_predictions.private.npz');arms=['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction'];scores={a:previous[a] for a in arms};scores.update({a:np.empty((len(y),20)) for a in ['score_selector','support_selector']});choices={};evaluation=[];maxerr=0.
 for outer in range(3):
  status('sealed_outer_evaluation',True,outer=outer);ids=np.flatnonzero(fold==outer);head=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);models=torch.load(out/f'selector_{outer}.private.pt',weights_only=False);b=data.scores(ids,head,'df_joint');f=data.scores(ids,dict(head,joint=head['joint']/head['temperature']+head['plain_correction']));maxerr=max(maxerr,float(np.max(np.abs(b-scores['df_matched_joint'][ids]))),float(np.max(np.abs(f-scores['plain_correction'][ids]))));assert maxerr<1e-8
  x,disagree=features(b/head['temperature'],f,data.cosine(ids,dict(head,prototypes=models['prototypes'],present=models['present'])));saved=dict(rows=ids,features=x,disagree=disagree)
  for arm,dim in [('score_selector',4),('support_selector',5)]:
   select,gain=choose(models[arm],x[:,:dim],disagree);scores[arm][ids]=np.where(select[:,None],f,b/head['temperature']);saved[arm]=select;saved[arm+'_gain']=gain
  np.savez(out/f'decisions_{outer}.private.npz',**saved)
 np.savez(out/'predictions.private.npz',labels=y,groups=g,folds=fold,**{a:scores[a] for a in ['score_selector','support_selector']});write(out/'results.json',summary(scores,y,g,fold,cfg));write(out/'HISTORY_CHECK.json',dict(max_score_error=maxerr));status('worker_complete',True);COST.update(process_wall_seconds=time.time()-START,cumulative_recorded_seconds=cfg['prior_recorded_seconds']+cfg['engineering_seconds']+time.time()-START,prior_cost_record_incomplete=True,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=0);write(out/'costs.json',COST);write(out/'status.private.json',dict(state='COMPLETE',finished=time.time(),costs=COST,independent_verification='PENDING'));print('COMPLETE',flush=True)


if __name__=='__main__':
 try:main()
 except BaseException:
  out=Path(cfg['output']);out.mkdir(exist_ok=True);write(out/'failure.private.json',dict(error=traceback.format_exc(),costs=COST,process_wall_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=0));write(out/'status.private.json',dict(state='FAILED',updated=time.time(),costs=COST));raise
