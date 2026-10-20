"""Two fixed subspaces, six constrained CE fits, with all-sample frozen-head diagnostics."""
import time
START=time.time()
import os,sys,json,traceback,resource
from pathlib import Path
import numpy as np
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
from core import weights,loss,metrics
from solver import solve
from basis import bases,diagnostic
from run_helpers import write,contrasts
RUNTIME={}


def main():
 torch.set_num_threads(cfg['CPU_threads']);torch.manual_seed(cfg['seed']);out=Path(cfg['output']);out.mkdir(exist_ok=False);parent=Path(cfg['parent_run']);attribution=Path(cfg['attribution_run']);assert json.loads((parent/'VERIFICATION.json').read_text())['state']=='PASS' and json.loads((attribution/'VERIFICATION.json').read_text())['state']=='PASS';write(out/'config.private.json',cfg)
 cost=dict(attempts=1,new_encodes=0,new_ridge_heads=0,correction_attempts=0,correction_completed=0,closure_evaluations=0);RUNTIME['cost']=cost
 def status(phase,**kw):
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;write(out/'resource_peaks.private.json',dict(RSS_peak_bytes=rss,GPU_peak_reserved_bytes=0,optimization_device='CPU'));write(out/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),costs=cost,RSS_peak_bytes=rss,GPU_peak_reserved_bytes=0,**kw));assert sum(p.stat().st_size for rr in [out.parent,*map(Path,cfg['prior_roots'])] for p in rr.rglob('*') if p.is_file())<cfg['new_storage_cap_bytes']-100000000
 idx=np.load(Path(cfg['cache_run'])/'train_index.private.npz');y=idx['labels'].astype(int);g=idx['groups'];fold=idx['folds'];cache=np.load(Path(cfg['cache_run'])/'train768.private.npy',mmap_mode='r');audits=[];diagnostics=[]
 for outer in range(3):
  status('training_only_bases',outer=outer);ids=np.flatnonzero(fold!=outer);w=weights(y[ids],g[ids]);w/=w.sum();tw=torch.from_numpy(w);ty=torch.from_numpy(y[ids]);h=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);j0=h['joint']/h['temperature'];x=torch.cat([torch.from_numpy(np.array(cache[ids],copy=True)).double(),torch.ones(len(ids),1,dtype=torch.float64)],1);z=x*h['factors'];del x;gram=z.T@(z*tw[:,None]);mean=(z*tw[:,None]).sum(0);qb,qp,info=bases(j0,h['CLS'],h['factors'],gram,mean,cfg['basis_rank_relative_tolerance']);write(out/f'BASIS_{outer}.json',dict(outer=outer,training_folds=[i for i in range(3) if i!=outer],**info))
  models=dict(score_basis=qb,PCA_basis=qp,joint_anchor=j0,factors=h['factors'],rank=info['rank'],training_mean=mean);torch.save(models,out/f'bases_{outer}.private.pt');status('all_training_sample_diagnostics',outer=outer)
  diag=diagnostic(z,ty,tw,j0,h['plain_correction'],qb);diag.update(outer=outer,scope='outer_training_all_samples');diagnostics.append(diag);write(out/'FROZEN_DIRECTION_AUDIT.json',diagnostics)
  base=z@j0
  for arm,q in [('score_subspace',qb),('pca_subspace',qp)]:
   cost['correction_attempts']+=1;status('fit_subspace',outer=outer,arm=arm);projected=z@q;smallgram=q.T@gram@q;proxy=smallgram/20+2*cfg['regularizer']*torch.eye(q.shape[1],dtype=q.dtype);ones=torch.ones(q.shape[1],dtype=q.dtype);empty=torch.zeros(len(ids),dtype=q.dtype)
   def fun(v):
    cost['closure_evaluations']+=1;status('fit_subspace',outer=outer,arm=arm);a=torch.tensor(v.reshape(q.shape[1],20),dtype=torch.float64,requires_grad=True);val=loss(a,projected,base,ty,tw,empty,empty,empty,ones,False,cfg['regularizer']);val.backward();value=float(val.detach());gradient=a.grad.numpy().ravel().copy();assert np.isfinite(value) and np.isfinite(gradient).all();return value,gradient
   checkpoint_path=out/f'candidate_{outer}_{arm}.private.npy'
   def checkpoint(v):
    tmp=checkpoint_path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,v)
    tmp.replace(checkpoint_path)
   candidate,infofit=solve(fun,np.zeros(q.shape[1]*20),proxy,cfg,checkpoint);infofit.update(outer=outer,arm=arm,gradient_coordinate='orthonormal_subspace_A');audits.append(infofit);write(out/'OPTIMIZATION.json',audits);checkpoint(candidate);assert infofit['success'],'Subspace solve failed fixed projected acceptance';models[arm]=torch.from_numpy(candidate.reshape(q.shape[1],20));cost['correction_completed']+=1;torch.save(models,out/f'partial_{outer}.private.pt');print(json.dumps(infofit),flush=True)
  torch.save(models,out/f'outer_{outer}.private.pt');del z,projected,gram,base
 assert cost['correction_completed']==6;write(out/'FIT_SEAL.json',dict(frozen_at=time.time(),outer_models=3,new_heads=6,new_encodes=0,original_validation_or_test=False))
 previous=np.load(parent/'outer_predictions.private.npz');attr=np.load(attribution/'predictions.private.npz');scores={a:previous[a] for a in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']};scores.update({a:attr[a] for a in ['cls_plain_correction','mass_shuffled_correction']});scores.update({a:np.empty((len(y),20)) for a in cfg['new_arms']})
 for outer in range(3):
  status('evaluate_sealed_subspaces',outer=outer);h=torch.load(out/f'outer_{outer}.private.pt',weights_only=True);ph=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);ids=np.flatnonzero(fold==outer);tw=torch.from_numpy(weights(y[ids],g[ids]));tw/=tw.sum();zz=[]
  for off in range(0,len(ids),2048):
   rows=ids[off:off+2048];x=torch.cat([torch.from_numpy(np.array(cache[rows],copy=True)).double(),torch.ones(len(rows),1,dtype=torch.float64)],1);z=x*h['factors'];zz.append(z)
   for arm,q in [('score_subspace',h['score_basis']),('pca_subspace',h['PCA_basis'])]:scores[arm][rows]=(z@(h['joint_anchor']+q@h[arm])).numpy()
  diag=diagnostic(torch.cat(zz),torch.from_numpy(y[ids]),tw,h['joint_anchor'],ph['plain_correction'],h['score_basis']);diag.update(outer=outer,scope='outer_heldout_all_samples');diagnostics.append(diag);write(out/'FROZEN_DIRECTION_AUDIT.json',diagnostics)
 np.savez(out/'predictions.private.npz',labels=y,groups=g,folds=fold,**{a:scores[a] for a in cfg['new_arms']});results=[]
 for outer in [None,0,1,2]:
  mask=np.ones(len(y),dtype=bool) if outer is None else fold==outer;ss={a:v[mask] for a,v in scores.items()};cs,gs=contrasts(ss,y[mask],g[mask]);results.append(dict(scope='pooled' if outer is None else f'outer_{outer}',metrics={a:metrics(v,y[mask],g[mask],cfg['order'],cfg['frequency_bins']) for a,v in ss.items()},contrasts=cs,groups=gs))
 write(out/'results.json',results);status('final_resource_check');cost.update(process_wall_seconds=time.time()-START,cumulative_recorded_seconds=cfg['prior_recorded_seconds']+cfg.get('engineering_seconds',0)+time.time()-START,prior_cost_record_incomplete=True,**json.loads((out/'resource_peaks.private.json').read_text()));write(out/'costs.json',cost);write(out/'status.private.json',dict(state='COMPLETE',costs=cost,finished=time.time(),independent_verification='PENDING'));print('COMPLETE',flush=True)


if __name__=='__main__':
 try:main()
 except BaseException:
  out=Path(cfg['output']);out.mkdir(exist_ok=True);peaks=dict(RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=0);write(out/'resource_peaks.private.json',peaks);write(out/'failure.private.json',dict(error=traceback.format_exc(),costs=RUNTIME.get('cost',{}),process_wall_seconds=time.time()-START,**peaks));write(out/'status.private.json',dict(state='FAILED',updated=time.time(),costs=RUNTIME.get('cost',{}),**peaks));raise
