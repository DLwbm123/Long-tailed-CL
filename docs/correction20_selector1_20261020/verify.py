"""No fitting: reconstruct fit provenance, utility labels, features, decisions and metrics."""
import time
START=time.time()
import os,sys,json,resource,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
from core import weights,metrics
from runtime import Data,write,summary


def own_features(b,f,cos):
 bi=b.argmax(1);fi=f.argmax(1);r=np.arange(len(b));bc=b.copy();fc=f.copy();bc[r,bi]=-np.inf;fc[r,fi]=-np.inf
 return np.column_stack([b[r,bi]-bc.max(1),f[r,fi]-fc.max(1),b[r,bi]-b[r,fi],f[r,fi]-f[r,bi],2*(cos[r,fi]-cos[r,bi])]),bi!=fi


def own_choice(m,x,d):
 if m['fallback']:return np.zeros(len(x),dtype=bool),np.zeros(len(x))
 z=np.column_stack([(x-m['mean'])/m['scale'],np.ones(len(x))]);s=z@m['coef'];prob=np.exp(s-logsumexp(s,axis=1)[:,None]);gain=prob[:,0]-prob[:,1];return d&(gain>0),gain


def verify():
 torch.set_num_threads(cfg['CPU_threads']);out=Path(cfg['output']);parent=Path(cfg['parent_run']);assert json.loads((out/'status.private.json').read_text())['state']=='COMPLETE';seal=json.loads((out/'FIT_SEAL.json').read_text());assert seal['selectors']==6 and not seal['outer_prediction_read_before_seal'];data=Data(cfg);y,g,fold=data.y,data.g,data.fold;audit=[];maxscore=0.;maxfeature=0.;maxdecision=0.;all_optimization=json.loads((out/'OPTIMIZATION.json').read_text());assert len(all_optimization)==9 and all(a['success'] for a in all_optimization)
 for h in range(3):
  ids=np.flatnonzero(fold==h);head=torch.load(out/f'singleton_{h}.private.pt',weights_only=False);raw,b,fac,proto,present=data.moment(ids);assert float((fac-head['factors']).abs().max())<1e-10 and torch.allclose(proto,head['prototypes'],rtol=0,atol=1e-10);ng=raw*fac[:,None]*fac[None,:];nb=b*fac[:,None];residuals=[]
  for mat,rhs,key in [(raw[data.ci][:,data.ci]+torch.eye(769),b[data.ci],'CLS'),(ng+torch.eye(1537),nb,'joint'),(ng+head['matched_alpha']*torch.eye(1537),nb,'df_joint')]:residuals.append(float((mat@head[key]-rhs).abs().max()/max(1.,float(rhs.abs().max()))))
  assert max(residuals)<1e-9
  ev=torch.linalg.eigvalsh(raw[data.ci][:,data.ci]).clamp_min(0);ej=torch.linalg.eigvalsh(ng).clamp_min(0);assert abs(float((ev/(ev+1)).sum()-(ej/(ej+head['matched_alpha'])).sum()))<1e-7
  stored=np.load(out/f'temperature_{h}.private.npz');assert np.array_equal(stored['rows'],ids);sub=data.subfold(ids);assert np.array_equal(sub,stored['half']);ts=np.empty((len(ids),20))
  for half in range(2):
   train=ids[sub!=half];test=ids[sub==half];assert not np.intersect1d(g[train],g[test]).size;rr,bb,ff,pp,pr=data.moment(train);th=torch.load(out/f'temp_{h}_{half}.private.pt',weights_only=True);assert float((ff-th['factors']).abs().max())<1e-10;er=float(((rr*ff[:,None]*ff[None,:]+torch.eye(1537))@th['joint']-bb*ff[:,None]).abs().max()/max(1.,float(bb.abs().max())));assert er<1e-9;ts[sub==half]=data.scores(test,th);del rr,bb
  maxscore=max(maxscore,float(np.max(np.abs(ts-stored['scores']))));w=weights(y[ids],g[ids]);center=ts-ts.mean(1,keepdims=True);temp=float(np.sqrt(np.sum(w[:,None]*center**2)/(20*w.sum())));assert abs(temp-head['temperature'])<1e-10;w/=w.sum();z=data.x(ids)*fac;R=head['plain_correction'];s=z@(head['joint']/temp+R);p=torch.softmax(s,dim=1);value=float((torch.from_numpy(w)*(-torch.log_softmax(s,dim=1)[torch.arange(len(ids)),torch.from_numpy(y[ids])])).sum()+cfg['regularizer']*(R**2).sum());p[torch.arange(len(ids)),torch.from_numpy(y[ids])]-=1;grad=z.T@(torch.from_numpy(w)[:,None]*p)+2*cfg['regularizer']*R;gmax=float((head['precondition'][:,None]*grad).abs().max());a=next(a for a in all_optimization if a.get('fold')==h and a['arm']=='singleton_plain');assert abs(value-a['final_objective'])<1e-9 and abs(gmax-a['gradient_max'])<1e-9;audit.append(dict(kind='singleton',fold=h,normal_equation_max=max(residuals),temperature=temp,objective=value,original_U_gradient_max=gmax));del z,raw,ng,b
 for outer in range(3):
  ids=np.flatnonzero(fold!=outer);held=np.flatnonzero(fold==outer);sig=np.load(out/f'signals_{outer}.private.npz');assert np.array_equal(sig['rows'],ids) and not np.intersect1d(g[ids],g[held]).size;b=np.empty((len(ids),20));f=b.copy();cos=b.copy()
  for hold in [j for j in range(3) if j!=outer]:
   train=next(j for j in range(3) if j not in [outer,hold]);mask=fold[ids]==hold;assert np.all(sig['teacher'][mask]==train);assert not np.intersect1d(g[fold==train],g[ids[mask]]).size;h=torch.load(out/f'singleton_{train}.private.pt',weights_only=False);b[mask]=data.scores(ids[mask],h,'df_joint')/h['temperature'];f[mask]=data.scores(ids[mask],dict(h,joint=h['joint']/h['temperature']+h['plain_correction']));cos[mask]=data.cosine(ids[mask],h)
  x,d=own_features(b,f,cos);maxscore=max(maxscore,float(np.max(np.abs(b-sig['B']))),float(np.max(np.abs(f-sig['F']))));maxfeature=max(maxfeature,float(np.max(np.abs(x-sig['features']))));target=np.where((f.argmax(1)==y[ids])&(b.argmax(1)!=y[ids]),0,np.where((b.argmax(1)==y[ids])&(f.argmax(1)!=y[ids]),1,2));assert np.array_equal(target,sig['target']) and np.array_equal(d,sig['disagree']);w=1/np.bincount(y[ids],minlength=20)[y[ids]];w/=w.sum();assert np.max(np.abs(w-sig['weights']))<1e-14;model=torch.load(out/f'selector_{outer}.private.pt',weights_only=False)
  for arm,dim in [('score_selector',4),('support_selector',5)]:
   m=model[arm]
   if m['fallback']:assert not len(x[d]) or not np.isin(target[d],[0,1]).any();continue
   ww=w[d];ww/=ww.sum();xx=x[d,:dim];mean=ww@xx;scale=np.sqrt(ww@((xx-mean)**2));scale=np.where(scale>1e-12,scale,1.);assert np.max(np.abs(mean-m['mean']))<1e-10 and np.max(np.abs(scale-m['scale']))<1e-10;xx=np.column_stack([(xx-mean)/scale,np.ones(len(xx))]);s=xx@m['coef'];lp=s-logsumexp(s,axis=1)[:,None];val=-np.dot(ww,lp[np.arange(len(xx)),target[d]])+cfg['selector_regularizer']*np.sum(m['coef'][:-1]**2);p=np.exp(lp);p[np.arange(len(xx)),target[d]]-=1;grad=xx.T@(ww[:,None]*p);grad[:-1]+=2*cfg['selector_regularizer']*m['coef'][:-1];a=next(a for a in all_optimization if a.get('outer')==outer and a['arm']==arm);assert abs(val-a['objective'])<1e-9 and abs(np.max(np.abs(grad))-a['gradient_max'])<1e-9
  raw,bb,fac,proto,present=data.moment(ids);assert float((fac-model['factors']).abs().max())<1e-10 and float((proto-model['prototypes']).abs().max())<1e-10;assert np.array_equal(present,model['present']);del raw,bb
 previous=np.load(parent/'outer_predictions.private.npz');saved=np.load(out/'predictions.private.npz');arms=['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction'];scores={a:previous[a] for a in arms};scores.update({a:np.empty((len(y),20)) for a in ['score_selector','support_selector']});selection_diagnostics=[]
 for outer in range(3):
  ids=np.flatnonzero(fold==outer);h=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);m=torch.load(out/f'selector_{outer}.private.pt',weights_only=False);bs=data.scores(ids,h,'df_joint');fs=data.scores(ids,dict(h,joint=h['joint']/h['temperature']+h['plain_correction']));maxscore=max(maxscore,float(np.max(np.abs(bs-previous['df_matched_joint'][ids]))),float(np.max(np.abs(fs-previous['plain_correction'][ids]))));x,d=own_features(bs/h['temperature'],fs,data.cosine(ids,dict(h,prototypes=m['prototypes'],present=m['present'])));dec=np.load(out/f'decisions_{outer}.private.npz');maxfeature=max(maxfeature,float(np.max(np.abs(x-dec['features']))));assert np.array_equal(d,dec['disagree']);iw=1/(20*np.bincount(y[ids],minlength=20)[y[ids]]);bc=bs.argmax(1)==y[ids];fc=fs.argmax(1)==y[ids]
  for arm,dim in [('score_selector',4),('support_selector',5)]:
   choose,gain=own_choice(m[arm],x[:,:dim],d);assert np.array_equal(choose,dec[arm]);maxdecision=max(maxdecision,float(np.max(np.abs(gain-dec[arm+'_gain']))));ss=np.where(choose[:,None],fs,bs/h['temperature']);scores[arm][ids]=ss;maxscore=max(maxscore,float(np.max(np.abs(ss-saved[arm][ids]))));retained_harm=float(iw@(choose&bc&~fc));lost_rescue=float(iw@(~choose&fc&~bc));rejected_harm=float(iw@(~choose&bc&~fc));actual=float(iw@((ss.argmax(1)==y[ids]).astype(float)-fc));assert abs(actual-(rejected_harm-lost_rescue))<1e-14;selection_diagnostics.append(dict(outer=outer,arm=arm,disagreement_images=int(d.sum()),adopt_F_images=int(choose.sum()),BA_rejected_harm=rejected_harm,BA_lost_rescue=lost_rescue,BA_retained_harm=retained_harm,BA_delta_vs_F=actual,BA_oracle=float(iw@(bc|fc))))
 recomputed=summary(scores,y,g,fold,cfg);stored=json.loads((out/'results.json').read_text())
 def compare(a,b):
  if isinstance(a,dict):assert a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a] or [0.])
  if isinstance(a,list):assert len(a)==len(b);return max([compare(x,y) for x,y in zip(a,b)] or [0.])
  if isinstance(a,(float,int)):return abs(a-b)
  assert a==b;return 0.
 metric_error=compare(recomputed,stored);assert metric_error<1e-12 and maxscore<1e-8 and maxfeature<1e-9 and maxdecision<1e-9
 cost=json.loads((out/'costs.json').read_text());assert cost['new_encodes']==0 and cost['ridge_attempts']==cost['ridge_completed']==9 and cost['correction_attempts']==cost['correction_completed']==3 and cost['selector_attempts']==cost['selector_completed']==6
 size=sum(p.stat().st_size for rr in [out.parent,*map(Path,cfg['prior_roots'])] for p in rr.rglob('*') if p.is_file());assert size<cfg['new_storage_cap_bytes']-100000000
 write(out/'SELECTION_DIAGNOSTICS.json',selection_diagnostics);result=dict(state='PASS',predictions_recomputed=2*len(y),metric_records=28,max_score_error=maxscore,max_feature_error=maxfeature,max_probability_error=maxdecision,max_metric_error=metric_error,fit_and_label_patient_isolation=True,temperature_and_base_recomputed=True,all_selector_objectives_checked=True,history_consistent=True,output_bytes_combined=size,worker_costs=cost,singleton_audit=audit,verification_seconds=time.time()-START,verification_RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_reserved_bytes=0);write(out/'VERIFICATION.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
 try:verify()
 except BaseException:
  write(Path(cfg['output'])/'VERIFICATION_FAILED.private.json',dict(error=traceback.format_exc(),verification_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));raise
