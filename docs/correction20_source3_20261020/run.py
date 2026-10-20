"""Fixed count-controlled prototype source ablation, cached readouts only."""
import time
START=time.time()
import os,sys,json,resource,traceback
from pathlib import Path
import numpy as np
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code']);sys.path.insert(0,cfg['selector_code'])
from runtime import Data,write,summary
from mechanism import fit,choose
from origin import block_features
COST=dict(attempts=1,new_encodes=0,new_ridge_heads=0,new_base_correction_heads=0,selector_attempts=0,selector_completed=0,cached_block_readout_images=0)
ARMS=['count_CLS','count_patch','count_joint']


def main():
 torch.set_num_threads(cfg['CPU_threads']);out=Path(cfg['output']);out.mkdir(exist_ok=False);old=Path(cfg['selector_run']);policy=Path(cfg['policy_run']);parent=Path(cfg['parent_run']);assert json.loads((old/'VERIFICATION.json').read_text())['state']=='PASS' and json.loads((policy/'VERIFICATION.json').read_text())['state']=='PASS';data=Data(cfg);y,g,fold=data.y,data.g,data.fold;write(out/'config.private.json',cfg);audits=[]
 def status(phase,**kw):
  peaks=dict(RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_reserved_bytes=0);write(out/'resource_peaks.private.json',peaks);write(out/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),costs=COST,**peaks,**kw))
 for outer in range(3):
  status('cached_block_features',outer=outer);sig=np.load(old/f'signals_{outer}.private.npz');pc=np.load(policy/f'training_{outer}.private.npz');ids=np.flatnonzero(fold!=outer);assert np.array_equal(ids,sig['rows']) and np.array_equal(ids,pc['rows']);block=np.empty((len(ids),2))
  for teacher in [h for h in range(3) if h!=outer]:
   mask=sig['teacher']==teacher;h=torch.load(old/f'singleton_{teacher}.private.pt',weights_only=False);block[mask]=block_features(data,ids[mask],h,sig['B'][mask],sig['F'][mask]);COST['cached_block_readout_images']+=int(mask.sum())
  extra=np.column_stack([block,sig['features'][:,4]]);np.savez(out/f'training_{outer}.private.npz',rows=ids,prototype_features=extra);models={}
  for j,a in enumerate(ARMS):
   x=np.column_stack([pc['count_features'],extra[:,j]]);d=sig['disagree'];COST['selector_attempts']+=1;status('selector_fit',outer=outer,arm=a);t=time.time();model,audit=fit(x[d],sig['target'][d],sig['weights'][d],cfg['selector_regularizer'],cfg);audit.update(outer=outer,arm=a,started_at=t,finished_at=time.time(),wall_seconds=time.time()-t);audits.append(audit);write(out/'OPTIMIZATION.json',audits);models[a]=model;torch.save(models,out/f'candidate_{outer}.private.pt');assert audit['success'],'Fixed acceptance failed; preserve candidate';COST['selector_completed']+=1
  torch.save(models,out/f'model_{outer}.private.pt')
 assert COST['selector_attempts']==COST['selector_completed']==9;write(out/'FIT_SEAL.json',dict(frozen_at=time.time(),selectors=9,primary='count_joint',new_encodes=0))
 oldscores=np.load(old/'predictions.private.npz');base=np.load(parent/'outer_predictions.private.npz');prior=np.load(policy/'predictions.private.npz');scores={a:base[a] for a in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']};scores.update({a:oldscores[a] for a in ['score_selector','support_selector']});scores.update({a:prior[a] for a in ['prototype_direct','count_selector','context_P00','context_P10','context_P01','context_P11']});scores.update({a:np.empty((len(y),20)) for a in ARMS})
 for outer in range(3):
  status('sealed_decisions',outer=outer);ids=np.flatnonzero(fold==outer);ph=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);head=torch.load(old/f'selector_{outer}.private.pt',weights_only=False);head['factors']=ph['factors'];B=scores['df_matched_joint'][ids]/ph['temperature'];F=scores['plain_correction'][ids];block=block_features(data,ids,head,B,F);COST['cached_block_readout_images']+=len(ids);dec=np.load(old/f'decisions_{outer}.private.npz');pc=np.load(policy/f'decisions_{outer}.private.npz');extra=np.column_stack([block,dec['features'][:,4]]);m=torch.load(out/f'model_{outer}.private.pt',weights_only=False);saved=dict(rows=ids,prototype_features=extra)
  for j,a in enumerate(ARMS):
   x=np.column_stack([pc['count_features'],extra[:,j]]);sel,gain=choose(m[a],x,dec['disagree']);scores[a][ids]=np.where(sel[:,None],F,B);saved[a]=sel;saved[a+'_gain']=gain
  np.savez(out/f'decisions_{outer}.private.npz',**saved)
 np.savez(out/'predictions.private.npz',labels=y,groups=g,folds=fold,**{a:scores[a] for a in ARMS});write(out/'results.json',summary(scores,y,g,fold,cfg));status('worker_complete');COST.update(process_wall_seconds=time.time()-START,cumulative_recorded_seconds=cfg['prior_recorded_seconds']+cfg['engineering_seconds']+time.time()-START,prior_cost_record_incomplete=True,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_reserved_bytes=0);write(out/'costs.json',COST);write(out/'status.private.json',dict(state='COMPLETE',costs=COST,finished=time.time(),independent_verification='PENDING'));print('COMPLETE',flush=True)


if __name__=='__main__':
 try:main()
 except BaseException:
  out=Path(cfg['output']);out.mkdir(exist_ok=True);write(out/'failure.private.json',dict(error=traceback.format_exc(),costs=COST,process_wall_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));write(out/'status.private.json',dict(state='FAILED',updated=time.time(),costs=COST));raise
