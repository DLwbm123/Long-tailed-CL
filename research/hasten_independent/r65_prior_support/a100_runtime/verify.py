"""Full saved-output closeout, no model/data rescoring or new fitting."""
import os,json,math
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);assert json.loads((root/'CONTROL_EXIT.json').read_text())['exit_code']==0;records=[];curves=[];steps=0;micro=0;checks=0;cache=[]
for ds in ['NIH','ISIC','HK']:
 dc=cfg['domains'][ds];d=Path(dc['output']);agg=json.loads((d/'aggregate.json').read_text());rr=json.loads((d/'TRAIN_RECORDS.json').read_text());assert len(rr)==3*len(dc['stages']);seal=json.loads((d/'ALL_HEADS_SEALED.json').read_text());assert seal['VAL_reads']==0;steps+=agg['optimizer_steps'];micro+=agg['micro_backwards'];assert agg['optimizer_steps']==dc['expected_steps']
 for r in rr:
  assert r['all_original_tail_parameters_exact'] and r['strict_restore_predictions_exact'] and r['old_fold_assignment_exact'] and r['metadata_groups_disjoint'] and r['support_detached'] and r['full_support_reencoded_for_head'];assert r['LoRA_trainable_parameters']==49152 and r['class_specific_parameters']==0 and r['VAL_reads']==0 and r['training_support_refreshes']==20
  assert r['prior_counts_applied_in_training_only'] and not r['inverse_class_weight'];assert all(abs(a-b/sum(r['train_counts']))<1e-6 for a,b in zip(r['class_total_risk_mass'],r['train_counts']));N=sum(r['train_counts']);assert N==r['all_real_rows']==sum(r['fold_rows']) and r['training_support_rows']==r['training_query_rows']==10*N;assert r['optimizer_steps']==10*sum(math.ceil(v/256) for v in r['fold_rows']) and r['micro_backwards']==10*sum(math.ceil(v/64) for v in r['fold_rows']);assert all(all(c>0 for c in z) for z in r['fold_class_counts']);assert [sum(z) for z in zip(*r['fold_class_counts'])]==r['train_counts']
  c=json.loads((d/str(r['seed'])/f"curves{r['stage']}.json").read_text());assert [z['epoch'] for z in c]==list(range(11)) and c[-1]['optimizer_steps']==r['optimizer_steps'];assert all(z['query_counts']==r['train_counts'] and z['is_independent_holdout']==False for z in c);records.append(dict(dataset=ds,**r));curves.append(dict(dataset=ds,seed=r['seed'],stage=r['stage'],curves=c))
  h=torch.load(d/str(r['seed'])/f"head{r['stage']}.private.pt",map_location='cpu',weights_only=False);assert all(torch.isfinite(t).all() for t in h['delta'].values());assert all(h[k].shape==(r['seen'],512) and torch.isfinite(h[k]).all() for k in ['prototypes','raw_prototypes']);assert h['train_counts'].tolist()==r['train_counts'] and h['inference_prior_adjustment']==False
 for seed in [74002,74003,74004]:
  for stage in range(len(dc['stages'])):
   f=torch.load(d/'responses'/str(seed)/f'responses{stage}.private.pt',map_location='cpu',weights_only=False);r=next(z for z in agg['results'] if z['seed']==seed and z['stage']==stage);y=f['labels'];assert len(f['ids'])==len(y) and len(set(f['ids']))==len(y) and set(f['scores'])=={'prior_support','frozen_support'}
   for arm,z in f['scores'].items():
    assert torch.isfinite(z).all() and z.shape==(len(y),r['seen']);pr=z.argmax(1);nn=torch.bincount(y,minlength=r['seen']).tolist();cc=torch.bincount(y[pr==y],minlength=r['seen']).tolist();assert nn==r['metrics'][arm]['counts'] and cc==r['metrics'][arm]['correct'];checks+=1
 for p in (root/'cache').iterdir():
  if not (p/'META.json').exists():continue
  m=json.loads((p/'META.json').read_text())
  if m['dataset']!=ds or int(p.name)==0:continue
  assert m['reference_max_abs']<=2e-5 and m['actual_encoded_rows']==len(m['ids'])<=16000;cache.append(dict(dataset=ds,split=m['split'],stage=m['stage'],block=m.get('block'),rows=m['actual_encoded_rows'],dtype=m['dtype'],token_shape=m['token_shape'],reference_max_abs=m['reference_max_abs'],token_bytes=sum(z['bytes'] for z in m['chunks'])))
 cleanup=json.loads((d/'REGENERABLE_CACHE_CLEANUP.json').read_text());assert cleanup['all3seeds_TRAIN_and_VAL_complete'] and cleanup['scientific_artifacts_preserved']
 assert cleanup['backend']=='bounded_FP32_single_fold_cache' and cleanup['reference_max_abs']<=2e-5 and cleanup['encoded_rows']>=sum(agg['train_counts'])
 cache.append(dict(dataset=ds,backend='A100_original_pixels_bounded_FP32_prefix_cache',reference_max_abs=cleanup['reference_max_abs'],encoded_rows=cleanup['encoded_rows'],no_lossy_quantization=True))
assert steps==52260 and micro==208230 and checks==90 and len(curves)==45
out=root/'results';out.mkdir(exist_ok=True)
for n,x in [('FULL_TRAIN_CURVES.json',curves),('TRAINING_RECORDS.json',records),('ENCODING_SUMMARY.json',cache),('CLOSEOUT_VERIFICATION.json',dict(effective_optimizer_steps=steps,micro_backwards=micro,trained_stages=45,curve_records=495,saved_response_checks=90,new_model_forwards=0,new_optimization=0,fp32_native_tokens=True,all_domains_seeds=True,support_query_metadata_components_disjoint=True,support_detached_semigradient=True,no_TEST=True,independent_patient_confirmation=False))]:(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
print('Verified complete R65 saved receipts,45stages/52260updates/90responses')
