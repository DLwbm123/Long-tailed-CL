"""Closeout of saved responses, curves and receipts only; zero model forwards."""
import os,json,math
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);curves=[];records=[];response_checks=0;steps=0;summary={}
assert json.loads((root/'CONTROL_EXIT.json').read_text())['exit_code']==0
for ds in ['NIH','ISIC','HK']:
 d=root/ds;dc=json.loads((d/'config.private.json').read_text());a=json.loads((d/'aggregate.json').read_text());rec=json.loads((d/'TRAIN_RECORDS.json').read_text());assert len(rec)==3*len(dc['stages']);steps+=a['optimizer_steps'];assert json.loads((d/'ALL_HEADS_SEALED.json').read_text())['VAL_reads']==0
 for r in rec:
  assert r['strict_restore_predictions_exact'] and r['raw_old_means_exact'] and r['class_specific_learned_parameters']==0 and r['VAL_reads']==0 and r['correction_F_norm']<=.250001
  assert r['all_seen_real_rows']==sum(r['train_counts']) and r['optimizer_steps']==10*math.ceil(r['all_seen_real_rows']/256) and r['class_total_risk_mass']==[1.]*r['seen'];records.append(dict(dataset=ds,**r))
 for seed in [74002,74003,74004]:
  for stage in range(len(dc['stages'])):
   c=json.loads((d/str(seed)/f'curves{stage}.json').read_text());r=next(z for z in rec if z['seed']==seed and z['stage']==stage);assert [z['epoch'] for z in c]==list(range(11));assert c[-1]['optimizer_steps']==r['optimizer_steps'];assert all(z['full_TRAIN_counts']==r['train_counts'] and z['correction_F_norm']<=.250001 for z in c);curves.append(dict(dataset=ds,seed=seed,stage=stage,curves=c))
   f=torch.load(d/str(seed)/f'responses{stage}.private.pt',map_location='cpu',weights_only=False);row=next(z for z in a['results'] if z['seed']==seed and z['stage']==stage);y=f['labels'];assert len(f['ids'])==len(y) and len(set(f['ids']))==len(y)
   for arm,z in f['scores'].items():
    assert torch.isfinite(z).all() and z.shape==(len(y),row['seen']);pred=z.argmax(1);counts=torch.bincount(y,minlength=row['seen']).tolist();correct=torch.bincount(y[y==pred],minlength=row['seen']).tolist();assert counts==row['metrics'][arm]['counts'] and correct==row['metrics'][arm]['correct'];response_checks+=1
 # Curves are full TRAIN resubstitution, NOT independent holdout trajectories.
 final=[z for z in curves if z['dataset']==ds and z['stage']==len(dc['stages'])-1];summary[ds]=dict(final_TRAIN_macro_BA_pct=sum(z['curves'][-1]['macro_BA'] for z in final)/3*100,final_TRAIN_tail_BA_pct=sum(z['curves'][-1]['tail_BA'] for z in final)/3*100,last3_objective_relative_range_under_1pct=sum((max(q['objective_mean'] for q in z['curves'][-3:])-min(q['objective_mean'] for q in z['curves'][-3:]))/max(abs(z['curves'][-1]['objective_mean']),1e-15)<.01 for z in curves if z['dataset']==ds),trained_stages=len(rec),bound_active_final_stages=sum(z['correction_F_norm']>.24999 for z in rec))
assert steps==52110 and response_checks==90 and len(curves)==45
for name,x in [('FULL_TRAIN_CURVES.json',curves),('TRAINING_RECORDS.json',records),('CLOSEOUT_VERIFICATION.json',dict(all_domains_seeds=True,optimizer_steps=steps,trained_stages=45,curve_records=495,saved_response_checks=response_checks,new_model_forwards=0,new_optimizer_steps=0,summary=summary,no_TEST=True,independent_confirmation=False))]:(root/'results'/name).write_text(json.dumps(x,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))
