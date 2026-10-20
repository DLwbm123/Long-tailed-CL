"""Recompute block means/cosines, fixed selector objectives, decisions and all metrics."""
import time
START=time.time()
import os,sys,json,resource,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code']);sys.path.insert(0,cfg['selector_code'])
from runtime import Data,write,summary
from core import weights
ARMS=['count_CLS','count_patch','count_joint']


def diff(a,b):
 if isinstance(a,dict):assert a.keys()==b.keys();return max([diff(a[k],b[k]) for k in a] or [0.])
 if isinstance(a,list):assert len(a)==len(b);return max([diff(x,y) for x,y in zip(a,b)] or [0.])
 if isinstance(a,(int,float)):return abs(a-b)
 assert a==b;return 0.


def independent(data,ids,head,B,F):
 x=np.asarray(data.cache[ids],dtype=np.float64);bp=B.argmax(1);fp=F.argmax(1);pr=head['prototypes'].numpy();out=[]
 for sl in [slice(0,768),slice(768,1536)]:
  v=x[:,sl];v=v/np.maximum(np.sqrt(np.sum(v*v,axis=1)),1e-300)[:,None];p=pr[:,sl];p=p/np.maximum(np.sqrt(np.sum(p*p,axis=1)),1e-300)[:,None];out.append(2*np.sum(v*(p[fp]-p[bp]),axis=1))
 return np.column_stack(out)


def means_check(data,ids,head):
 means=np.empty((20,1536));w=weights(data.y[ids],data.g[ids])
 for c in range(20):
  mask=data.y[ids]==c;assert mask.any();means[c]=np.average(np.asarray(data.cache[ids[mask]],dtype=np.float64),axis=0,weights=w[mask])
 stored=head['prototypes'].numpy();maximum=0.
 for sl in [slice(0,768),slice(768,1536)]:
  a=means[:,sl];a=a/np.linalg.norm(a,axis=1)[:,None];b=stored[:,sl];b=b/np.linalg.norm(b,axis=1)[:,None];maximum=max(maximum,float(np.abs(a-b).max()))
 assert maximum<1e-10;return maximum


def verify():
 torch.set_num_threads(cfg['CPU_threads']);out=Path(cfg['output']);old=Path(cfg['selector_run']);policy=Path(cfg['policy_run']);parent=Path(cfg['parent_run']);assert json.loads((out/'status.private.json').read_text())['state']=='COMPLETE';data=Data(cfg);y,g,f=data.y,data.g,data.fold;maxfeat=0.;maxscore=0.;maxproto=0.;audits=json.loads((out/'OPTIMIZATION.json').read_text());assert len(audits)==9 and all(a['success'] for a in audits);seal=json.loads((out/'FIT_SEAL.json').read_text());assert max(a['finished_at'] for a in audits)<=seal['frozen_at']
 for teacher in range(3):
  head=torch.load(old/f'singleton_{teacher}.private.pt',weights_only=False);maxproto=max(maxproto,means_check(data,np.flatnonzero(f==teacher),head))
 for outer in range(3):
  ids=np.flatnonzero(f!=outer);held=np.flatnonzero(f==outer);assert not np.intersect1d(g[ids],g[held]).size;sig=np.load(old/f'signals_{outer}.private.npz');pc=np.load(policy/f'training_{outer}.private.npz');saved=np.load(out/f'training_{outer}.private.npz');assert np.array_equal(saved['rows'],ids);extra=np.empty((len(ids),3));extra[:,2]=sig['features'][:,4]
  for teacher in [h for h in range(3) if h!=outer]:
   mask=sig['teacher']==teacher;assert not np.intersect1d(g[ids[mask]],g[f==teacher]).size;h=torch.load(old/f'singleton_{teacher}.private.pt',weights_only=False);extra[mask,:2]=independent(data,ids[mask],h,sig['B'][mask],sig['F'][mask])
  maxfeat=max(maxfeat,float(np.abs(extra-saved['prototype_features']).max()));models=torch.load(out/f'model_{outer}.private.pt',weights_only=False);d=sig['disagree'];target=sig['target'];ww=sig['weights']
  for j,a in enumerate(ARMS):
   model=models[a];audit=next(z for z in audits if z['outer']==outer and z['arm']==a)
   if model['fallback']:assert not d.any() or not np.isin(target[d],[0,1]).any();continue
   x=np.column_stack([pc['count_features'],extra[:,j]])[d];w=ww[d];w/=w.sum();mean=w@x;scale=np.sqrt(w@((x-mean)**2));scale=np.where(scale>1e-12,scale,1.);assert np.abs(mean-model['mean']).max()<1e-10 and np.abs(scale-model['scale']).max()<1e-10;z=np.column_stack([(x-mean)/scale,np.ones(len(x))]);s=z@model['coef'];lp=s-logsumexp(s,axis=1)[:,None];val=-w@lp[np.arange(len(x)),target[d]]+cfg['selector_regularizer']*np.sum(model['coef'][:-1]**2);p=np.exp(lp);p[np.arange(len(x)),target[d]]-=1;grad=z.T@(w[:,None]*p);grad[:-1]+=2*cfg['selector_regularizer']*model['coef'][:-1];assert abs(val-audit['objective'])<1e-9 and abs(float(np.abs(grad).max())-audit['gradient_max'])<1e-9
 base=np.load(parent/'outer_predictions.private.npz');prior=np.load(policy/'predictions.private.npz');older=np.load(old/'predictions.private.npz');saved=np.load(out/'predictions.private.npz');scores={a:base[a] for a in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']};scores.update({a:older[a] for a in ['score_selector','support_selector']});scores.update({a:prior[a] for a in ['prototype_direct','count_selector','context_P00','context_P10','context_P01','context_P11']});scores.update({a:np.empty((len(y),20)) for a in ARMS});diagnostics=[]
 for outer in range(3):
  ids=np.flatnonzero(f==outer);ph=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);head=torch.load(old/f'selector_{outer}.private.pt',weights_only=False);maxproto=max(maxproto,means_check(data,np.flatnonzero(f!=outer),head));B=base['df_matched_joint'][ids]/ph['temperature'];F=base['plain_correction'][ids];extra=np.column_stack([independent(data,ids,head,B,F),np.load(old/f'decisions_{outer}.private.npz')['features'][:,4]]);dec=np.load(out/f'decisions_{outer}.private.npz');maxfeat=max(maxfeat,float(np.abs(extra-dec['prototype_features']).max()));pc=np.load(policy/f'decisions_{outer}.private.npz');models=torch.load(out/f'model_{outer}.private.pt',weights_only=False);d=B.argmax(1)!=F.argmax(1);iw=1/(20*np.bincount(y[ids],minlength=20)[y[ids]])
  for j,a in enumerate(ARMS):
   m=models[a];x=np.column_stack([pc['count_features'],extra[:,j]])
   if m['fallback']:sel=np.zeros(len(ids),dtype=bool)
   else:
    z=np.column_stack([(x-m['mean'])/m['scale'],np.ones(len(x))]);s=z@m['coef'];p=np.exp(s-logsumexp(s,axis=1)[:,None]);sel=d&(p[:,0]>p[:,1])
   assert np.array_equal(sel,dec[a]);ss=np.where(sel[:,None],F,B);scores[a][ids]=ss;maxscore=max(maxscore,float(np.abs(ss-saved[a][ids]).max()));correct=ss.argmax(1)==y[ids];ref=scores['count_selector'][ids].argmax(1)==y[ids];rescue=float(iw@(correct&~ref));harm=float(iw@(~correct&ref));assert abs(float(iw@(correct.astype(int)-ref))-(rescue-harm))<1e-12;diagnostics.append(dict(outer=outer,arm=a,BA_rescue_vs_count=rescue,BA_harm_vs_count=harm,BA_delta_vs_count=rescue-harm))
 metricerr=diff(summary(scores,y,g,f,cfg),json.loads((out/'results.json').read_text()));assert maxfeat<1e-9 and maxscore<1e-8 and metricerr<1e-12;cost=json.loads((out/'costs.json').read_text());assert cost['selector_attempts']==cost['selector_completed']==9 and cost['new_encodes']==cost['new_ridge_heads']==cost['new_base_correction_heads']==0;size=sum(p.stat().st_size for rr in [out.parent,*map(Path,cfg['prior_roots'])] for p in rr.rglob('*') if p.is_file());assert size<cfg['new_storage_cap_bytes']-100000000;write(out/'CONDITIONAL_DIAGNOSTICS.json',diagnostics);v=dict(state='PASS',new_predictions_recomputed=3*len(y),metric_records=64,max_prototype_error=maxproto,max_feature_error=maxfeat,max_score_error=maxscore,max_metric_error=metricerr,block_means_independently_reconstructed=True,patient_isolation=True,all_objectives_and_gradients_checked=True,old_history_consistent=True,combined_remote_bytes=size,worker_costs=cost,verification_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_reserved_bytes=0);write(out/'VERIFICATION.json',v);print(json.dumps(v),flush=True)


if __name__=='__main__':
 try:verify()
 except BaseException:
  write(Path(cfg['output'])/'VERIFICATION_FAILED.private.json',dict(error=traceback.format_exc(),verification_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));raise
