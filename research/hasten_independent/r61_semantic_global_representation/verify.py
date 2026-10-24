"""Verify all saved trained arms, curves and response counts; no new forwards."""
import os,json,math
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);curves=[];records=[];steps=0;checks=0
assert json.loads((root/'CONTROL_EXIT.json').read_text())['exit_code']==0
for ds in ['NIH','ISIC','HK']:
 d=root/ds;dc=json.loads((d/'config.private.json').read_text());agg=json.loads((d/'aggregate.json').read_text());rec=json.loads((d/'TRAIN_RECORDS.json').read_text());assert len(rec)==9*len(dc['stages']);steps+=agg['optimizer_steps'];assert json.loads((d/'ALL_HEADS_SEALED.json').read_text())['VAL_reads']==0
 for r in rec:
  assert r['strict_restore_predictions_exact'] and r['fixed_anchors_exact'] and r['old_semantic_prefix_preserved'] and r['initial_core_paired_exact'] and r['same_prespecified_batch_generator'] and r['class_specific_learned_parameters']==0 and r['shared_parameters']==66112 and r['VAL_reads']==0
  assert r['all_seen_real_rows']==sum(r['train_counts']) and r['optimizer_steps']==10*math.ceil(r['all_seen_real_rows']/256) and r['class_total_risk_mass']==[1.]*r['seen'];c=json.loads((d/r['arm']/str(r['seed'])/f"curves{r['stage']}.json").read_text());assert [z['epoch'] for z in c]==list(range(11)) and c[-1]['optimizer_steps']==r['optimizer_steps'] and all(z['full_TRAIN_counts']==r['train_counts'] for z in c);curves.append(dict(dataset=ds,arm=r['arm'],seed=r['seed'],stage=r['stage'],curves=c));records.append(dict(dataset=ds,**r))
 for seed in [74002,74003,74004]:
  for stage in range(len(dc['stages'])):
   f=torch.load(d/'responses'/str(seed)/f'responses{stage}.private.pt',map_location='cpu',weights_only=False);r=next(z for z in agg['results'] if z['seed']==seed and z['stage']==stage);y=f['labels'];assert len(f['ids'])==len(y) and len(set(f['ids']))==len(y);assert set(f['scores'])=={'semantic_global','paired_control','permuted_semantic','frozen_semantic'}
   for arm,z in f['scores'].items():
    assert torch.isfinite(z).all() and z.shape==(len(y),r['seen']);pred=z.argmax(1);n=torch.bincount(y,minlength=r['seen']).tolist();c=torch.bincount(y[pred==y],minlength=r['seen']).tolist();assert n==r['metrics'][arm]['counts'] and c==r['metrics'][arm]['correct'];checks+=1
assert steps==156330 and checks==180 and len(curves)==135
for name,x in [('FULL_TRAIN_CURVES.json',curves),('TRAINING_RECORDS.json',records),('CLOSEOUT_VERIFICATION.json',dict(optimizer_steps=steps,trained_stages=135,curve_records=1485,saved_response_checks=checks,new_model_forwards=0,new_optimizer_steps=0,all_domains_seeds_arms=True,no_TEST=True,independent_confirmation=False))]:(root/'results'/name).write_text(json.dumps(x,indent=2,allow_nan=False))
print('Verified135stages/156330steps/1485curves/180savedreadouts')
