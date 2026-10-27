"""Saved-only transition/count verification; no models or image reads."""
from pathlib import Path
import os,json,math
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);out=r/'HK';assert json.loads((r/'CONTROL_EXIT.json').read_text())['exit_code']==0
records=json.loads((out/'CENSUS_RECORDS.json').read_text());summaries=json.loads((out/'CLASS_SUMMARIES.json').read_text());seal=json.loads((out/'CENSUS_SEAL.json').read_text());assert len(records)==len(summaries)==18 and seal['flipped_endpoint_forward_rows']==seal['saved_original_embedding_rows']==87651 and seal['view_scoring_routes']==144 and seal['optimizer_steps']==seal['backwards']==seal['VAL_TEST_reads']==0
assert {(x['stage'],x['model']) for x in records}=={(s,m) for s in range(6) for m in ['sft_reference','prototype_grpo','distribution_grpo']}
for rec,s in zip(records,summaries):
 assert (rec['stage'],rec['model'])==(s['stage'],s['model']) and rec['optimizer_steps']==rec['backwards']==rec['VAL_TEST_reads']==0
 for k in ['old_folds_preserved','original_supports_only','original_weights_unchanged','LoRA_delta_unchanged','gradients_absent','opposite_component_fold_support']:assert rec[k]
 assert rec['full_original_prototype_parity_maxabs']<=2e-5 and sum(rec['counts'])==rec['TRAIN_rows']
 assert set(s['modes'])=={'prototype_raw','prototype_natural','distribution_raw','distribution_natural'}
 for mode,classes in s['modes'].items():
  assert len(classes)==rec['seen']
  for c,v in enumerate(classes):
   n=rec['counts'][c];assert v['class_index']==c and v['count']==n
   assert v['both_correct']+v['original_only_correct']+v['flipped_only_correct']+v['neither_correct']==n
   assert v['original_correct']==v['both_correct']+v['original_only_correct'] and v['flipped_correct']==v['both_correct']+v['flipped_only_correct']
   assert v['original_only_correct']+v['flipped_only_correct']<=v['prediction_disagreements']<=n
   for key,value in v.items():assert math.isfinite(value)
   for key in ['original_true_probability_mean','flipped_true_probability_mean']:assert 0<=v[key]<=1
   assert -1.00001<=v['paired_embedding_cosine_mean']<=1.00001
assert sum(v['TRAIN_rows'] for v in records)==87651
c=json.loads((out/'REGENERABLE_CACHE_CLEANUP.json').read_text());assert c['flipped_full_visual_parity_maxabs']<=2e-5 and c['encoded_unique_flipped_rows']==6821 and c['TRAIN_only'] and c['science_preserved'] and c['temporary_swap_bytes']==0
p=r/'results';p.mkdir(exist_ok=True);x=dict(passed=True,saved_only=True,new_model_scores=0,records=18,flipped_endpoint_forward_rows=87651,saved_original_embedding_rows=87651,paired_classifier_policy_routes=72,view_scoring_routes=144,optimizer_steps=0,backwards=0,VAL_TEST_reads=0);(p/'VERIFICATION.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
