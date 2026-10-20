"""Independent NumPy projected objectives, PCA geometry, diagnostics and predictions."""
import time
START=time.time()
import os,sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
from core import weights,metrics
from run_helpers import write,contrasts


def main():
 torch.set_num_threads(4);out=Path(cfg['output']);parent=Path(cfg['parent_run']);attr=Path(cfg['attribution_run']);assert json.loads((out/'status.private.json').read_text())['state']=='COMPLETE';seal=json.loads((out/'FIT_SEAL.json').read_text());assert seal['new_heads']==6 and seal['frozen_at']<=(out/'predictions.private.npz').stat().st_mtime
 idx=np.load(Path(cfg['cache_run'])/'train_index.private.npz');y=idx['labels'].astype(int);g=idx['groups'];fold=idx['folds'];cache=np.load(Path(cfg['cache_run'])/'train768.private.npy',mmap_mode='r');pred=np.load(out/'predictions.private.npz');assert np.array_equal(y,pred['labels']) and np.array_equal(g,pred['groups']) and np.array_equal(fold,pred['folds']);audits=json.loads((out/'OPTIMIZATION.json').read_text());diag=json.loads((out/'FROZEN_DIRECTION_AUDIT.json').read_text());scoreerr=objerr=graderr=diagnosticerr=0.;records=[]
 for outer in range(3):
  h=torch.load(out/f'outer_{outer}.private.pt',weights_only=True);h={k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in h.items()};ph=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);j0=ph['joint'].numpy()/ph['temperature'];fac=ph['factors'].numpy();assert np.max(np.abs(h['joint_anchor']-j0))<1e-12 and np.array_equal(h['factors'],fac);ids=np.flatnonzero(fold!=outer);w=weights(y[ids],g[ids]);w/=w.sum();x=np.column_stack([np.array(cache[ids],dtype=np.float64),np.ones(len(ids))]);z=x*fac;del x;gram=z.T@(z*w[:,None]);mean=(z*w[:,None]).sum(0);qb,qp=h['score_basis'],h['PCA_basis'];r=qb.shape[1];assert r==qp.shape[1] and r<=39 and r>1
  for q in [qb,qp]:assert np.max(np.abs(q.T@q-np.eye(r)))<1e-10 and q[-1,0]==1 and np.max(np.abs(q[:-1,0]))==0 and np.max(np.abs(q[-1,1:]))==0
  ch=np.zeros_like(j0);ch[:768]=ph['CLS'].numpy()[:768]/fac[:768,None];ch[-1]=ph['CLS'].numpy()[-1]/fac[-1];center=np.eye(20)-np.ones((20,20))/20;heads=np.column_stack([j0@center,ch@center]);assert np.max(np.abs(heads-qb@(qb.T@heads)))<1e-9;fheads=heads[:-1];norm=np.linalg.norm(fheads,axis=0);sv=np.linalg.svd(fheads[:,norm>0]/norm[norm>0],compute_uv=False);assert int((sv>sv.max()*cfg['basis_rank_relative_tolerance']).sum())==r-1
  cov=gram[:-1,:-1]-mean[:-1,None]*mean[None,:-1];ev=np.linalg.eigvalsh((cov+cov.T)/2);pv=qp[:-1,1:];selected=ev[-(r-1):];assert np.max(np.abs(cov@pv-pv*selected[None,:]))/max(1.,np.abs(cov).max())<1e-8
  for arm,q in [('score_subspace',qb),('pca_subspace',qp)]:
   a=h[arm];rr=q@a;assert abs(float((rr*rr).sum()-(a*a).sum()))<1e-10*max(1.,float((a*a).sum()));small=z@q;s=z@j0+small@a;lz=logsumexp(s,axis=1);ce=lz-s[np.arange(len(ids)),y[ids]];obj=float(w@ce+cfg['regularizer']*(a*a).sum());prob=np.exp(s-lz[:,None]);prob[np.arange(len(ids)),y[ids]]-=1;ga=small.T@(prob*w[:,None])+2*cfg['regularizer']*a;record=next(a for a in audits if a['outer']==outer and a['arm']==arm);eo=abs(obj-record['final_objective']);eg=abs(float(np.abs(ga).max())-record['gradient_max']);objerr=max(objerr,eo);graderr=max(graderr,eg);assert eo<1e-8 and eg<1e-7 and record['success'] and (record['gradient_max']<=cfg['gtol'] or (record['relative_accepted_objective_change'] is not None and record['relative_accepted_objective_change']<=cfg['ftol']));records.append(dict(outer=outer,arm=arm,objective=obj,projected_gradient_max=float(np.abs(ga).max())))
  for scope,rows in [('outer_training_all_samples',ids),('outer_heldout_all_samples',np.flatnonzero(fold==outer))]:
   wz=weights(y[rows],g[rows]);wz/=wz.sum();xx=np.column_stack([np.array(cache[rows],dtype=np.float64),np.ones(len(rows))]);zz=xx*fac;del xx;residual=ph['plain_correction'].numpy();parallel=qb@(qb.T@residual);ps=zz@parallel;ns=zz@(residual-parallel);base=zz@j0;full=base+ps+ns;iy=np.arange(len(rows));yy=y[rows];other=full.copy();other[iy,yy]=-np.inf;j=other.argmax(1);dp=ps[iy,yy]-ps[iy,j];dn=ns[iy,yy]-ns[iy,j];old=base.copy();old[iy,yy]=-np.inf;bm=base[iy,yy]-old.max(1);fm=full[iy,yy]-other.max(1);switch=base[iy,yy]-base[iy,j]-bm;assert np.max(np.abs(fm-bm-dp-dn-switch))<1e-9;good0=base.argmax(1)==yy;good1=full.argmax(1)==yy;stored=next(a for a in diag if a['outer']==outer and a['scope']==scope)
   for c,cl in enumerate(stored['classes']):
    mask=yy==c;assert cl['class_index']==c and cl['images']==int(mask.sum()) and cl['anchor_correct']==int(good0[mask].sum()) and cl['plain_correct']==int(good1[mask].sum());diagnosticerr=max(diagnosticerr,float(np.max(np.abs(np.quantile(bm[mask],[0,.25,.5,.75,1])-cl['anchor_margin_quantiles']))),float(np.max(np.abs(np.quantile(fm[mask],[0,.25,.5,.75,1])-cl['plain_margin_quantiles']))))
    for kind,sel in [('rescue',mask&~good0&good1),('harm',mask&good0&~good1),('all',mask)]:
     item=cl['groups'][kind];assert item['images']==int(sel.sum()) and abs(item['PP_mass']-float(wz[sel].sum()))<1e-12
     if sel.any():
      for key,val in [('parallel_mean',dp[sel].mean()),('perpendicular_mean',dn[sel].mean()),('competitor_switch_mean',switch[sel].mean()),('parallel_PP_mean',wz[sel]@dp[sel]/wz[sel].sum()),('perpendicular_PP_mean',wz[sel]@dn[sel]/wz[sel].sum())]:diagnosticerr=max(diagnosticerr,abs(float(val)-item[key]))
   if scope=='outer_heldout_all_samples':
    for arm,q in [('score_subspace',qb),('pca_subspace',qp)]:ss=zz@(j0+q@h[arm]);scoreerr=max(scoreerr,float(np.abs(ss-pred[arm][rows]).max()));assert np.array_equal(ss.argmax(1),pred[arm][rows].argmax(1))
 assert max(scoreerr,diagnosticerr)<1e-8
 p=np.load(parent/'outer_predictions.private.npz');a=np.load(attr/'predictions.private.npz');ss={arm:p[arm] for arm in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']};ss.update({arm:a[arm] for arm in ['cls_plain_correction','mass_shuffled_correction']});ss.update({arm:pred[arm] for arm in cfg['new_arms']});results=json.loads((out/'results.json').read_text())
 def close(a,b):
  if isinstance(a,dict):assert set(a)==set(b);return max([close(a[k],b[k]) for k in a] or [0])
  if isinstance(a,list):assert len(a)==len(b);return max([close(aa,bb) for aa,bb in zip(a,b)] or [0])
  if isinstance(a,(float,int)):assert abs(a-b)<1e-10;return abs(a-b)
  assert a==b;return 0
 metricerr=0.
 for f,row in zip([None,0,1,2],results):
  mask=np.ones(len(y),dtype=bool) if f is None else fold==f;scores={k:v[mask] for k,v in ss.items()};mm={k:metrics(v,y[mask],g[mask],cfg['order'],cfg['frequency_bins']) for k,v in scores.items()};cs,gs=contrasts(scores,y[mask],g[mask]);metricerr=max(metricerr,close(mm,row['metrics']),close(cs,row['contrasts']),close(gs,row['groups']))
  for arm,v in scores.items():bw=1/(20*np.bincount(y[mask],minlength=20)[y[mask]]);assert abs(float(bw@(v.argmax(1)==y[mask]))-row['metrics'][arm]['BA'])<1e-12
 c=json.loads((out/'costs.json').read_text());assert c['new_encodes']==0 and c['new_ridge_heads']==0 and c['correction_completed']==6 and c['correction_attempts']==6;size=sum(p.stat().st_size for rr in [out.parent,*map(Path,cfg['prior_roots'])] for p in rr.rglob('*') if p.is_file());assert size<cfg['new_storage_cap_bytes']-100000000
 write(out/'VERIFICATION.json',dict(state='PASS',new_predictions_recounted=len(y)*2,metric_records_recounted=36,max_score_error=scoreerr,max_objective_error=objerr,max_projected_gradient_error=graderr,max_diagnostic_error=diagnosticerr,max_metric_error=metricerr,original_coordinate_penalty_preserved=True,training_only_subspace_geometry_verified=True,fixed_competitor_decomposition_verified=True,model_seal_verified=True,new_fits=0,new_encodes=0,objectives=records,combined_remote_bytes=size,process_wall_seconds=time.time()-START));print('PASS',flush=True)


if __name__=='__main__':main()
