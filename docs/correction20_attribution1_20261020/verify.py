"""Independent NumPy loss/gradient/prediction recount of six frozen control heads."""
import time
START=time.time()
import json,os,sys,resource
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp,expit
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
from core import weights,metrics
from control import transport
from run_helpers import write,contrasts


def main():
 torch.set_num_threads(4);out=Path(cfg['output']);parent=Path(cfg['parent_run']);assert json.loads((out/'status.private.json').read_text())['state']=='COMPLETE';seal=json.loads((out/'FIT_SEAL.json').read_text());assert seal['frozen_at']<=(out/'predictions.private.npz').stat().st_mtime and seal['new_control_heads']==6
 idx=np.load(Path(cfg['cache_run'])/'train_index.private.npz');y=idx['labels'].astype(int);g=idx['groups'];f=idx['folds'];cache=np.load(Path(cfg['cache_run'])/'train768.private.npy',mmap_mode='r');pred=np.load(out/'predictions.private.npz');assert np.array_equal(pred['labels'],y) and np.array_equal(pred['groups'],g) and np.array_equal(pred['folds'],f);old=np.load(parent/'outer_predictions.private.npz');opt=json.loads((out/'OPTIMIZATION.json').read_text());errscore=errobj=errgrad=0.;ci=list(range(768))+[1536];records=[]
 for outer in range(3):
  h=torch.load(out/f'outer_{outer}.private.pt',weights_only=True);h={k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in h.items()};ph=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);inner=np.load(parent/f'inner_{outer}.private.npz');ids=np.flatnonzero(f!=outer);w=weights(y[ids],g[ids]);w/=w.sum();ys=y[ids];a=np.load(out/f'shuffle_{outer}.private.npz');assert np.array_equal(a['rows'],ids);e,d,m=a['recipients'],a['donors'],a['mass'];re=transport(ys,w,cfg['seed'],outer)
  for aa,bb in zip((e,d,m),re):assert np.array_equal(aa,bb)
  assert np.all(ys[e]==ys[d]) and np.max(np.abs(np.bincount(e,weights=m,minlength=len(ids))-w))<1e-12 and np.max(np.abs(np.bincount(d,weights=m,minlength=len(ids))-w))<1e-12
  assert np.max(np.abs(h['CLS_factors']-np.r_[np.ones(768)*2**-.5,1.]))<1e-12;assert np.max(np.abs(h['CLS_anchor']*h['CLS_factors'][:,None]-ph['CLS'].numpy()))<1e-12
  centered=inner['CLS']-inner['CLS'].mean(1,keepdims=True);tc=float(np.sqrt((w[:,None]*centered**2).sum()/20));assert abs(tc-h['CLS_temperature'])<1e-12
  for arm in cfg['new_arms']:
   iscls=arm=='cls_plain_correction';fac=h['CLS_factors'] if iscls else h['joint_factors'];anchor=h['CLS_anchor'] if iscls else h['joint_anchor'];temp=h['CLS_temperature'] if iscls else h['joint_temperature'];r=h[arm];pre=h[arm+'_precondition'];x=np.column_stack([np.array(cache[ids],dtype=np.float64),np.ones(len(ids))]);z=x[:,ci]*fac if iscls else x*fac;del x;s=z@(anchor/temp+r);lz=logsumexp(s,axis=1);ce=lz-s[np.arange(len(ids)),ys];p=np.exp(s-lz[:,None]);p[np.arange(len(ids)),ys]-=1
   if iscls:obj=float(w@ce);ds=p*w[:,None]
   else:
    repair=inner['repair'].astype(float);keep=inner['keep'].astype(float);target=inner['target'];rw=np.bincount(e,weights=m*repair[d],minlength=len(ids));obj=float((w+rw)@ce);ds=p*(w+rw)[:,None];live=keep[d]>0;ee=e[live];dd=d[live];mw=m[live]*keep[dd];arg=target[dd,None]-s[ee,ys[ee],None]+s[ee];pen=np.logaddexp(0,arg);pen[np.arange(len(ee)),ys[ee]]=0;obj+=float(mw@(pen.sum(1)/19));gp=expit(arg);gp[np.arange(len(ee)),ys[ee]]=0;gp*=mw[:,None]/19;gp[np.arange(len(ee)),ys[ee]]=-gp.sum(1);np.add.at(ds,ee,gp)
    for c in range(20):
     mask=ys==c;edge=ys[e]==c
     for term in [repair,keep,target]:assert abs(w[mask]@term[mask]-m[edge]@term[d[edge]])<1e-12
   obj+=cfg['regularizer']*float((r*r).sum());gu=(z.T@ds+2*cfg['regularizer']*r)*pre[:,None];a=next(a for a in opt if a['outer']==outer and a['arm']==arm);eo=abs(obj-a['final_objective']);eg=abs(float(np.abs(gu).max())-a['gradient_max']);errobj=max(errobj,eo);errgrad=max(errgrad,eg);assert eo<1e-8 and eg<1e-7 and a['success'] and (a['gradient_max']<=cfg['gtol'] or (a['relative_accepted_objective_change'] is not None and a['relative_accepted_objective_change']<=cfg['ftol']))
   records.append(dict(outer=outer,arm=arm,objective=obj,gradient_max=float(np.abs(gu).max())));del z,s,ds
   ids2=np.flatnonzero(f==outer)
   for off in range(0,len(ids2),2048):
    rows=ids2[off:off+2048];x=np.column_stack([np.array(cache[rows],dtype=np.float64),np.ones(len(rows))]);z=x[:,ci]*fac if iscls else x*fac;score=z@(anchor/temp+r);errscore=max(errscore,float(np.abs(score-pred[arm][rows]).max()));assert np.array_equal(score.argmax(1),pred[arm][rows].argmax(1))
 assert errscore<1e-9
 results=json.loads((out/'results.json').read_text());allscore={a:old[a] for a in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']};allscore.update({a:pred[a] for a in cfg['new_arms']})
 def close(a,b):
  if isinstance(a,dict):assert set(a)==set(b);return max([close(a[k],b[k]) for k in a] or [0])
  if isinstance(a,list):assert len(a)==len(b);return max([close(aa,bb) for aa,bb in zip(a,b)] or [0])
  if isinstance(a,(int,float)):assert abs(a-b)<1e-10;return abs(a-b)
  assert a==b;return 0
 metricerr=0.
 for fold,row in zip([None,0,1,2],results):
  mask=np.ones(len(y),dtype=bool) if fold is None else f==fold;ss={a:v[mask] for a,v in allscore.items()};mm={a:metrics(v,y[mask],g[mask],cfg['order'],cfg['frequency_bins']) for a,v in ss.items()};cs,gs=contrasts(ss,y[mask],g[mask]);metricerr=max(metricerr,close(mm,row['metrics']),close(cs,row['contrasts']),close(gs,row['groups']))
  for arm,v in ss.items():
   wc=1/(20*np.bincount(y[mask],minlength=20)[y[mask]]);assert abs(float(wc@(v.argmax(1)==y[mask]))-row['metrics'][arm]['BA'])<1e-12
 cost=json.loads((out/'costs.json').read_text());assert cost['new_encodes']==0 and cost['new_ridge_heads']==0 and cost['correction_attempts']==6 and cost['correction_completed']==6;size=sum(p.stat().st_size for root in [out.parent,Path(cfg['prior_root'])] for p in root.rglob('*') if p.is_file());assert size<cfg['new_storage_cap_bytes']-100000000
 write(out/'VERIFICATION.json',dict(state='PASS',new_predictions_recounted=len(y)*2,parent_arms_recounted=5,metric_records_recounted=28,max_score_error=errscore,max_objective_error=errobj,max_gradient_error=errgrad,max_metric_error=metricerr,model_seal_before_evaluation=True,weighted_transport_conserved=True,CLS_anchor_and_temperature_patch_free=True,objectives=records,new_encodes=0,new_fits=0,combined_remote_bytes=size,process_wall_seconds=time.time()-START,verifier_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));print('PASS',flush=True)


if __name__=='__main__':main()
