"""Saved outputs/receipts only, after all complete domains, no pixels/forwards."""
import os,json,math
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);assert json.loads((root/'CONTROL_EXIT.json').read_text())['exit_code']==0;records=[];curves=[];steps=0;micro=0;checks=0;cache=[]
for ds in ['NIH','ISIC','HK']:
 dc=cfg['domains'][ds];d=Path(dc['output']);agg=json.loads((d/'aggregate.json').read_text());rr=json.loads((d/'TRAIN_RECORDS.json').read_text());assert len(rr)==3*len(dc['stages']);seal=json.loads((d/'ALL_HEADS_SEALED.json').read_text());assert seal['VAL_reads']==0;steps+=agg['optimizer_steps'];micro+=agg['micro_backwards']
 for r in rr:
  assert r['all_original_tail_parameters_exact'] and r['strict_restore_predictions_exact'] and r['fixed_text_exact'] and r['LoRA_trainable_parameters']==49152 and r['class_specific_parameters']==0 and r['VAL_reads']==0
  N=sum(r['train_counts']);assert N==r['all_real_rows'] and r['optimizer_steps']==10*math.ceil(N/256) and r['micro_backwards']==10*math.ceil(N/64);c=json.loads((d/str(r['seed'])/f"curves{r['stage']}.json").read_text());assert [z['epoch'] for z in c]==list(range(11)) and c[-1]['optimizer_steps']==r['optimizer_steps'];assert all(z['full_TRAIN_counts']==r['train_counts'] for z in c);records.append(dict(dataset=ds,**r));curves.append(dict(dataset=ds,seed=r['seed'],stage=r['stage'],curves=c))
 for seed in [74002,74003,74004]:
  for stage in range(len(dc['stages'])):
   f=torch.load(d/'responses'/str(seed)/f'responses{stage}.private.pt',map_location='cpu',weights_only=False);r=next(z for z in agg['results'] if z['seed']==seed and z['stage']==stage);y=f['labels'];assert len(f['ids'])==len(y) and len(set(f['ids']))==len(y)
   for arm,z in f['scores'].items():
    assert torch.isfinite(z).all() and z.shape==(len(y),r['seen']);p=z.argmax(1);nn=torch.bincount(y,minlength=r['seen']).tolist();cc=torch.bincount(y[p==y],minlength=r['seen']).tolist();assert nn==r['metrics'][arm]['counts'] and cc==r['metrics'][arm]['correct'];checks+=1
 for j in range(dc['job_base'],dc['job_base']+len(dc['stages'])+1):
  p=root/'cache'/str(j);m=json.loads((p/'META.json').read_text());assert m['reference_max_abs']<=2e-5 and m['actual_encoded_rows']==len(m['ids']);cache.append(dict(dataset=ds,split=m['split'],stage=m['stage'],rows=m['actual_encoded_rows'],dtype=m['dtype'],token_shape=m['token_shape'],reference_max_abs=m['reference_max_abs'],token_bytes=sum(z['bytes'] for z in m['chunks'])))
 cleanup=json.loads((d/'REGENERABLE_CACHE_CLEANUP.json').read_text());assert cleanup['all3seeds_TRAIN_and_VAL_complete'] and cleanup['scientific_artifacts_preserved']
assert steps==52110 and checks==90 and len(curves)==45
out=root/'results';out.mkdir(exist_ok=True)
for n,x in [('FULL_TRAIN_CURVES.json',curves),('TRAINING_RECORDS.json',records),('ENCODING_SUMMARY.json',cache),('CLOSEOUT_VERIFICATION.json',dict(effective_optimizer_steps=steps,micro_backwards=micro,trained_stages=45,curve_records=495,saved_response_checks=90,new_model_forwards=0,new_optimization=0,fp32_native_tokens=True,all_domains_seeds=True,all_inputs_match_frozen_reference=True,no_TEST=True,independent_patient_confirmation=False))]:(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
print('Verified complete R62 saved receipts,45stages/52110updates/90responses')
