"""Zero-fit audit followed by two cached ablations. Frozen rules, no base training."""
import time
START=time.time()
import os,sys,json,resource,traceback
from pathlib import Path
import numpy as np
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code']);sys.path.insert(0,cfg['selector_code'])
from runtime import Data,write,summary
from mechanism import fit,choose
from control import patient_counts,inputs,policy,oracle_diagnostics
COST=dict(attempts=1,new_encodes=0,new_ridge_heads=0,new_base_correction_heads=0,selector_attempts=0,selector_completed=0,cached_CLS_readout_images=0)
RULES=[('context_P00',0,0),('context_P10',1,0),('context_P01',0,1),('context_P11',1,1)]


def main():
 torch.set_num_threads(cfg['CPU_threads']);out=Path(cfg['output']);out.mkdir(exist_ok=False);old=Path(cfg['selector_run']);parent=Path(cfg['parent_run']);assert json.loads((old/'VERIFICATION.json').read_text())['state']=='PASS';write(out/'config.private.json',cfg)
 def status(phase,**kw):
  peaks=dict(RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=0);write(out/'resource_peaks.private.json',peaks);write(out/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),costs=COST,**peaks,**kw))
 data=Data(cfg);y,g,fold=data.y,data.g,data.fold;tail=[cfg['order'].index(c) for c in cfg['frequency_bins']['tail']];oldpred=np.load(old/'predictions.private.npz');basepred=np.load(parent/'outer_predictions.private.npz');arms=['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction'];scores={a:basepred[a] for a in arms};scores.update({a:oldpred[a] for a in ['score_selector','support_selector']});disagree=np.zeros(len(y),dtype=bool);direct=np.zeros(len(y),dtype=bool);old_choices={a:np.zeros(len(y),dtype=bool) for a in ['score_selector','support_selector']}
 status('E0_zero_fit_diagnostics');t0=time.time()
 for outer in range(3):
  d=np.load(old/f'decisions_{outer}.private.npz');ids=np.flatnonzero(fold==outer);assert np.array_equal(ids,d['rows']);disagree[ids]=d['disagree'];direct[ids]=d['disagree']&(d['features'][:,4]>0)
  for a in old_choices:old_choices[a][ids]=d[a]
 choices=dict(old_choices,prototype_direct=direct);diag=oracle_diagnostics(scores,y,g,fold,cfg['order'],choices);write(out/'E0_DIAGNOSTICS.json',diag);write(out/'E0_SEAL.json',dict(completed_at=time.time(),elapsed_seconds=time.time()-t0,selector_attempts=COST['selector_attempts'],new_encodes=0,new_base_fits=0,stage2_config_precommitted=True));status('E0_complete');print(json.dumps(dict(event='E0_complete',pooled=diag[0])),flush=True)
 counts={h:patient_counts(y,g,np.flatnonzero(fold==h)) for h in range(3)};optim=[];bindings=[];supportaudit=[]
 for outer in range(3):
  status('cached_training_features',outer=outer);sig=np.load(old/f'signals_{outer}.private.npz');ids=np.flatnonzero(fold!=outer);assert np.array_equal(ids,sig['rows']);c=np.empty((len(ids),20));cnt=np.empty((len(ids),20),dtype=int)
  for teacher in [h for h in range(3) if h!=outer]:
   mask=sig['teacher']==teacher;rows=ids[mask];assert not np.intersect1d(g[rows],g[fold==teacher]).size;head=torch.load(old/f'singleton_{teacher}.private.pt',weights_only=False);c[mask]=data.scores(rows,head,'CLS');cnt[mask]=counts[teacher];COST['cached_CLS_readout_images']+=len(rows)
  xc,xctx=inputs(sig['features'],sig['B'],sig['F'],c,cnt,tail);d=sig['disagree'];target=sig['target'];w=sig['weights'];np.savez(out/f'training_{outer}.private.npz',rows=ids,CLS=c,count_features=xc,context_features=xctx,teacher=sig['teacher']);models={}
  for arm,x in [('count_selector',xc),('context_selector',xctx)]:
   COST['selector_attempts']+=1;status('selector_fit',outer=outer,arm=arm);start=time.time();model,audit=fit(x[d],target[d],w[d],cfg['selector_regularizer'],cfg);audit.update(outer=outer,arm=arm,started_at=start,finished_at=time.time(),wall_seconds=time.time()-start);optim.append(audit);write(out/'OPTIMIZATION.json',optim);models[arm]=model;torch.save(models,out/f'candidate_{outer}.private.pt');assert audit['success'],'Fixed selector acceptance failed; no automatic retry';COST['selector_completed']+=1
  models['outer_counts']=patient_counts(y,g,ids);torch.save(models,out/f'model_{outer}.private.pt');bindings.append(dict(outer=outer,training_folds=[h for h in range(3) if h!=outer],training_images=len(ids),disagreement_images=int(d.sum()),dimensions=dict(count=5,context=9),no_new_base_fits=True));write(out/'BINDINGS.json',bindings)
  for teacher in [h for h in range(3) if h!=outer]:
   mask=sig['teacher']==teacher
   supportaudit.append(dict(outer=outer,teacher=teacher,training_class_patients=counts[teacher].tolist(),held_feature_rows=int(mask.sum()),prototype_feature_quantiles=np.quantile(sig['features'][mask,4],[0,.25,.5,.75,1]).tolist(),count_feature_quantiles=np.quantile(xc[mask,4],[0,.25,.5,.75,1]).tolist()))
 assert COST['selector_attempts']==COST['selector_completed']==6;write(out/'FIT_SEAL.json',dict(frozen_at=time.time(),selectors=6,base_fits=0,new_encodes=0,E0_read_outer_predictions=True,primary_candidate='context_P11'));write(out/'TRAIN_SUPPORT.json',supportaudit)
 newarms=['prototype_direct','count_selector',*[a for a,_,_ in RULES]];scores.update({a:np.empty((len(y),20)) for a in newarms});allchoices={a:np.zeros(len(y),dtype=bool) for a in newarms}
 for outer in range(3):
  status('sealed_outer_decisions',outer=outer);ids=np.flatnonzero(fold==outer);d=np.load(old/f'decisions_{outer}.private.npz');h=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);m=torch.load(out/f'model_{outer}.private.pt',weights_only=False);B=scores['df_matched_joint'][ids]/h['temperature'];F=scores['plain_correction'][ids];C=scores['cls768_ridge'][ids];xc,xctx=inputs(d['features'],B,F,C,m['outer_counts'],tail);select,_=choose(m['count_selector'],xc,d['disagree']);local=dict(prototype_direct=direct[ids],count_selector=select);saved=dict(rows=ids,count_features=xc,context_features=xctx)
  for arm,lt,lc in RULES:local[arm],saved[arm+'_log_odds']=policy(m['context_selector'],xctx,d['disagree'],lt,lc,20/len(tail))
  for arm,select in local.items():allchoices[arm][ids]=select;saved[arm]=select;scores[arm][ids]=np.where(select[:,None],F,B)
  np.savez(out/f'decisions_{outer}.private.npz',**saved)
 np.savez(out/'predictions.private.npz',labels=y,groups=g,folds=fold,**{a:scores[a] for a in newarms});write(out/'results.json',summary(scores,y,g,fold,cfg));write(out/'DECISION_DIAGNOSTICS.json',oracle_diagnostics(scores,y,g,fold,cfg['order'],dict(old_choices,**allchoices)));status('worker_complete');COST.update(process_wall_seconds=time.time()-START,cumulative_recorded_seconds=cfg['prior_recorded_seconds']+cfg['engineering_seconds']+time.time()-START,prior_cost_record_incomplete=True,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=0);write(out/'costs.json',COST);write(out/'status.private.json',dict(state='COMPLETE',finished=time.time(),costs=COST,independent_verification='PENDING'));print('COMPLETE',flush=True)


if __name__=='__main__':
 try:main()
 except BaseException:
  out=Path(cfg['output']);out.mkdir(exist_ok=True);write(out/'failure.private.json',dict(error=traceback.format_exc(),costs=COST,process_wall_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_reserved_bytes=0));write(out/'status.private.json',dict(state='FAILED',updated=time.time(),costs=COST));raise
