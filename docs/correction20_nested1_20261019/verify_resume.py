"""Independent cache-based prediction, objective, scope and metric recount; no fitting."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp,expit
from core import weights,targets,metrics
from run import write,contrasts


def main():
 start=time.time();cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);src=Path(cfg['cache_run']);torch.set_num_threads(4)
 assert json.loads((out/'status.private.json').read_text())['state']=='COMPLETE';seal=json.loads((out/'FIT_SEAL.json').read_text());assert seal['correction_heads']==6 and seal['new_encodes']==0
 assert seal['frozen_at']<=(out/'outer_predictions.private.npz').stat().st_mtime
 ix=np.load(src/'train_index.private.npz');y=ix['labels'].astype(int);g=ix['groups'];f=ix['folds'];cache=np.load(src/'train768.private.npy',mmap_mode='r');pred=np.load(out/'outer_predictions.private.npz');res=json.loads((out/'results.json').read_text());maxerr=0.;objective_error=0.;normal_error=0.;checks=[]
 arms=['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction'];assert np.array_equal(pred['labels'],y) and np.array_equal(pred['folds'],f) and np.array_equal(pred['groups'],g)
 def chunks(ids):
  for off in range(0,len(ids),1024):
   rows=ids[off:off+1024];yield off,rows,np.column_stack([np.array(cache[rows],dtype=np.float64),np.ones(len(rows))])
 for outer in range(3):
  train=np.flatnonzero(f!=outer);test=np.flatnonzero(f==outer);allowed=[h for h in range(3) if h!=outer];h=torch.load(out/f'outer_{outer}.private.pt',weights_only=True);h={k:(v.numpy() if isinstance(v,torch.Tensor) else v) for k,v in h.items()};old=torch.load(src/f'fold_head_5_{outer}.private.pt',weights_only=True)
  for k in ['CLS','joint','factors']:assert np.array_equal(h[k],old[k].numpy())
  inner=np.load(out/f'inner_{outer}.private.npz');assert np.array_equal(inner['rows'],train)
  for hold in allowed:
   teacher=next(t for t in allowed if t!=hold);assert teacher!=outer and teacher!=hold
   teacherhead=torch.load(out/f'inner_head_{teacher}.private.pt',weights_only=True);teach={k:v.numpy() for k,v in teacherhead.items()};mask=f[train]==hold;cc=[];jj=[]
   for off,rows,x in chunks(train[mask]):cc.append(x[:,list(range(768))+[1536]]@teach['CLS']);jj.append((x*teach['factors'])@teach['joint'])
   for key,v in [('CLS',np.concatenate(cc)),('joint',np.concatenate(jj))]:assert np.max(np.abs(inner[key][mask]-v))<1e-10
  w=weights(y[train],g[train]);repair,keep,target,temp,cap=targets(inner['CLS'],inner['joint'],y[train],w);w/=w.sum()
  assert np.array_equal(repair,inner['repair']) and np.array_equal(keep,inner['keep']);assert np.max(np.abs(target-inner['target']))<1e-12 and abs(temp-h['temperature'])<1e-12
  objectives={a:0. for a in arms[3:]};grad={a:np.zeros((1537,20)) for a in arms[3:]}
  for off,rows,x in chunks(train):
   n=len(rows);yy=y[rows];ww=w[off:off+n];rr=repair[off:off+n];kk=keep[off:off+n];tt=target[off:off+n];z=x*h['factors'];base=z@h['joint']/temp
   for a in arms[3:]:
    s=base+z@h[a];logz=logsumexp(s,axis=1);prob=np.exp(s-logz[:,None]);ce=logz-s[np.arange(n),yy];mul=1+rr if a=='utility_correction' else np.ones(n)
    objectives[a]+=float(ww@(ce*mul));ds=prob;ds[np.arange(n),yy]-=1;ds*=ww[:,None]*mul[:,None]
    if a=='utility_correction':
     arg=tt[:,None]-s[np.arange(n),yy,None]+s;pen=np.logaddexp(0,arg);pen[np.arange(n),yy]=0;objectives[a]+=float((ww*kk)@(pen.sum(1)/19))
     gp=expit(arg);gp[np.arange(n),yy]=0;gp*=((ww*kk)/19)[:,None];gp[np.arange(n),yy]=-gp.sum(1);ds+=gp
    grad[a]+=z.T@ds
  audits=json.loads((out/f'optimization_{outer}.json').read_text())
  for a,audit in zip(arms[3:],audits):
   objectives[a]+=cfg['regularizer']*float((h[a]**2).sum());err=abs(objectives[a]-audit['final_objective']);objective_error=max(objective_error,err);assert err<1e-8
   gr=(grad[a]+2*cfg['regularizer']*h[a])*h['precondition'][:,None];assert abs(float(np.abs(gr).max())-audit['gradient_max'])<1e-7
   assert audit['success'] and audit['iterations']<=cfg['maxiter'];checks.append(dict(outer=outer,arm=a,objective=objectives[a],gradient_max=float(np.abs(gr).max())))
  for off,rows,x in chunks(test):
   z=x*h['factors'];sc={'cls768_ridge':x[:,list(range(768))+[1536]]@h['CLS'],'scale_joint':z@h['joint'],'df_matched_joint':z@h['df_joint']}
   for a in arms[3:]:sc[a]=z@(h['joint']/temp+h[a])
   for a in arms:maxerr=max(maxerr,float(np.abs(sc[a]-pred[a][rows]).max()));assert np.array_equal(sc[a].argmax(1),pred[a][rows].argmax(1))
 # Check every new singleton ridge and all reused outer heads by normal equations without forming or solving a Gram.
 for scope in ['inner','outer']:
  for fold in range(3):
   ids=np.flatnonzero(f==fold if scope=='inner' else f!=fold);w=weights(y[ids],g[ids]);hp=torch.load(out/f'{scope}_{"head_" if scope=="inner" else ""}{fold}.private.pt',weights_only=True);hh={k:(v.numpy() if isinstance(v,torch.Tensor) else v) for k,v in hp.items()}
   for key,reg in [('CLS',1.),('joint',1.)]+([] if scope=='inner' else [('df_joint',hh['matched_alpha'])]):
    v=hh[key];normal=reg*v.copy();rhs=np.zeros_like(v)
    for off,rows,x in chunks(ids):
     xx=x[:,list(range(768))+[1536]] if key=='CLS' else x*hh['factors'];ww=w[off:off+len(rows)];one=np.eye(20)[y[rows]];normal+=xx.T@((xx@v-one)*ww[:,None]);rhs+=xx.T@(one*ww[:,None])
    e=float(np.abs(normal).max()/max(1.,float(np.abs(rhs).max())));normal_error=max(normal_error,e);assert e<1e-9
 def close(a,b):
  if isinstance(a,dict):assert set(a)==set(b);return max([close(a[k],b[k]) for k in a] or [0.])
  if isinstance(a,list):assert len(a)==len(b);return max([close(x,z) for x,z in zip(a,b)] or [0.])
  if isinstance(a,(float,int)):assert abs(a-b)<1e-10;return abs(a-b)
  assert a==b;return 0.
 metricerr=0.
 for fold in [None,0,1,2]:
  m=np.ones(len(y),dtype=bool) if fold is None else f==fold;ss={a:pred[a][m] for a in arms};cs,gs=contrasts(ss,y[m],g[m]);r=res if fold is None else res['outer'][fold]
  metricerr=max(metricerr,close({a:metrics(v,y[m],g[m],cfg['order'],cfg['frequency_bins']) for a,v in ss.items()},r['metrics']),close(cs,r['contrasts']),close(gs,r['groups']))
 cost=json.loads((out/'costs.json').read_text());assert cost['new_encodes']==0 and cost['ridge_completed']==10 and cost['ridge_unique_completed']==9 and cost['artifact_reconstruction_solves']==1 and cost['correction_attempts']==7 and cost['correction_completed']==6 and cost['attempts']==2;assert cost['output_bytes']<cfg['new_storage_cap_bytes']-100000000
 assert maxerr<1e-9
 oldfail=json.loads((out/'failure_attempt1'/'optimization_0.json').read_text())[0];assert oldfail['success'] is False and oldfail['iterations']==500
 peaks=json.loads((out/'resource_peaks.private.json').read_text());assert peaks['RSS_peak_bytes']>0 and peaks['GPU_peak_reserved_bytes']>0
 for outer in range(3):
  for a in json.loads((out/f'optimization_{outer}.json').read_text()):assert a['success'] and (a['gradient_max']<=cfg['gtol'] or (a['relative_accepted_objective_change'] is not None and a['relative_accepted_objective_change']<=cfg['ftol']))
 write(out/'VERIFICATION.json',dict(state='PASS',predictions_recounted=len(y)*5,max_score_error=maxerr,max_objective_error=objective_error,max_normal_equation_error=normal_error,max_metric_error=metricerr,outer_model_seal_verified=True,no_new_fits_or_encodes=True,objectives=checks,wall_seconds=time.time()-start,cumulative_measured_seconds=cost['cumulative_measured_seconds']+time.time()-start))
 print(json.dumps(dict(state='PASS',predictions=len(y)*5,wall_seconds=time.time()-start)))


if __name__=='__main__':main()
