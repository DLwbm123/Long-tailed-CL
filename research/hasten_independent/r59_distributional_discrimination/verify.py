"""Saved complete curves, receipts and responses, no model forward or training."""
import os,json,math
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);curves=[];records=[];checks=0;steps=0
assert json.loads((root/'CONTROL_EXIT.json').read_text())['exit_code']==0
for ds in ['NIH','ISIC','HK']:
 d=root/ds;dc=json.loads((d/'config.private.json').read_text());a=json.loads((d/'aggregate.json').read_text());rec=json.loads((d/'TRAIN_RECORDS.json').read_text());assert len(rec)==3*len(dc['stages']);steps+=a['optimizer_steps'];assert json.loads((d/'ALL_HEADS_SEALED.json').read_text())['VAL_reads']==0
 for r in rec:
  assert r['strict_restore_predictions_exact'] and r['class_specific_learned_parameters']==0 and r['learned_shared_parameters']==512 and r['VAL_reads']==0 and r['all_query_self_entries_excluded'] and r['no_support_gradient_detach']
  assert r['all_seen_real_rows']==sum(r['train_counts']) and r['optimizer_steps']==10*math.ceil(r['all_seen_real_rows']/256) and r['class_total_risk_mass']==[1.]*r['seen'] and r['own_class_support_denominator']==[z-1 for z in r['train_counts']];records.append(dict(dataset=ds,**r))
 for seed in [74002,74003,74004]:
  for stage in range(len(dc['stages'])):
   c=json.loads((d/str(seed)/f'curves{stage}.json').read_text());r=next(z for z in rec if z['seed']==seed and z['stage']==stage);assert [z['epoch'] for z in c]==list(range(11)) and c[-1]['optimizer_steps']==r['optimizer_steps'];assert all(z['full_TRAIN_counts']==r['train_counts'] and z['metric_weight_min']>0 and z['metric_effective_dimensions']<=512.001 for z in c);curves.append(dict(dataset=ds,seed=seed,stage=stage,curves=c))
   f=torch.load(d/str(seed)/f'responses{stage}.private.pt',map_location='cpu',weights_only=False);row=next(z for z in a['results'] if z['seed']==seed and z['stage']==stage);y=f['labels'];assert len(f['ids'])==len(y) and len(set(f['ids']))==len(y) and set(f['scores'])=={'distribution_shared','distribution_fixed','centroid_shared'}
   for arm,z in f['scores'].items():
    assert torch.isfinite(z).all() and z.shape==(len(y),row['seen']);pred=z.argmax(1);counts=torch.bincount(y,minlength=row['seen']).tolist();correct=torch.bincount(y[y==pred],minlength=row['seen']).tolist();assert counts==row['metrics'][arm]['counts'] and correct==row['metrics'][arm]['correct'];checks+=1
assert steps==52110 and checks==135 and len(curves)==45
for name,x in [('FULL_TRAIN_LOO_CURVES.json',curves),('TRAINING_RECORDS.json',records),('CLOSEOUT_VERIFICATION.json',dict(optimizer_steps=steps,trained_stages=45,curve_records=495,saved_response_checks=checks,new_model_forwards=0,new_optimizer_steps=0,all_domains_seeds=True,full_TRAIN_LOO_not_independent_holdout=True,no_TEST=True,independent_confirmation=False))]:(root/'results'/name).write_text(json.dumps(x,indent=2,allow_nan=False))
print('Verified45stages/52110steps/495curves/135savedreadouts;0newtraining-or-model-forwards')
