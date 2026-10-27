"""Verify only saved outputs; never run a model or read images."""
from pathlib import Path
import os,json,math
import torch
from decision import ARMS,GROUP_SIZE,tailness,guide_weights
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);assert json.loads((root/'CONTROL_EXIT.json').read_text())['exit_code']==0
assert json.loads((root/'CONTROL_EXIT.json').read_text())['peak_aggregate_RSS_bytes']<30000000000
steps=micro=branches=readouts=curves=0
for ds in ['HK']:
 d=root/ds;dc=cfg['domains'][ds];records=json.loads((d/'TRAIN_RECORDS.json').read_text());assert len(records)==len(dc['seeds'])*4*len(dc['stages']);seal=json.loads((d/'ALL_HEADS_SEALED.json').read_text());assert seal['records']==len(records) and seal['VAL_reads']==0
 agg=json.loads((d/'aggregate.json').read_text());assert len(agg['results'])==len(dc['seeds'])*len(dc['stages'])
 for r in records:
  assert r['original_parameters_frozen'] and r['strict_restore_exact'] and r['old_fold_assignment_exact'] and r['opposite_fold_support'] and r['raw_inference'] and r['tailness_frozen'] and r['VAL_reads']==0 and r['LoRA_trainable_parameters']==49152
  assert r['supervised_direction_guard'] and r['guide_class_balanced']==r['arm'].startswith('balanced_') and r['matched_guide_backward_exposure'] and r['optimizer_moments_unprojected'] and len(r['direction_diagnostics'])==r['optimizer_steps']
  for dd in r['direction_diagnostics']:
   if r['supervised_direction_guard']:assert dd['actual_applied_guide_dot']<=dd['applied_rounding_abs_error']+dd['projection_tolerance']
   else:assert not dd['projected']
  assert r['guide_micro_backwards']==r['micro_backwards']
  assert torch.allclose(torch.tensor(r['guide_class_weights'],dtype=torch.double),guide_weights(torch.tensor(r['train_counts'],dtype=torch.double),r['arm']))
  n=sum(r['train_counts']);assert r['epochs']==2 and r['query_rows']==r['support_rows']==2*n and r['optimizer_steps']==2*sum(math.ceil(v/256) for v in r['fold_rows']) and r['micro_backwards']==2*sum(math.ceil(v/64) for v in r['fold_rows'])
  f=d/r['arm']/str(r['seed']);cc=json.loads((f/f"curves{r['stage']}.json").read_text());assert [v['epoch'] for v in cc]==[0,1,2] and cc[-1]['optimizer_steps']==r['optimizer_steps'] and all(v['query_counts']==r['train_counts'] for v in cc)
  h=torch.load(f/f"head{r['stage']}.private.pt",map_location='cpu',weights_only=False);assert h['train_counts'].tolist()==r['train_counts'] and h['raw_inference'] and h['matched_SFT_stage_initialization'] and all(torch.isfinite(v).all() for v in h['delta'].values()) and h['prototypes'].shape==(r['seen'],512)
  t=json.loads((d/str(r['seed'])/f"TAILNESS{r['stage']}.json").read_text());assert torch.allclose(tailness(torch.tensor(t['counts']),torch.tensor(t['SFT_correct'])),torch.tensor(t['weights'],dtype=torch.double))
  s=r['statistics']
  if r['arm'].endswith('grpo'):assert s['groups']==2*n and s['actions']==2*n*GROUP_SIZE and 0<=s['zero_advantage_groups']<=s['groups'] and 0<=s['clipped_actions']<=s['actions'] and 0<=s['reward_sum']<=s['actions']
  else:assert s['groups']==s['actions']==0
  z=torch.load(f/f"response{r['stage']}.private.pt",map_location='cpu',weights_only=False);y=z['labels'];assert z['scores'].shape==(len(y),r['seen']) and torch.isfinite(z['scores']).all() and len(z['ids'])==len(y)==len(set(z['ids']));metric=next(v for v in agg['results'] if v['seed']==r['seed'] and v['stage']==r['stage'])['metrics'][r['arm']];assert torch.bincount(y,minlength=r['seen']).tolist()==metric['counts'] and torch.bincount(y[z['scores'].argmax(1)==y],minlength=r['seen']).tolist()==metric['correct']
  branches+=1;readouts+=1;curves+=len(cc);steps+=r['optimizer_steps'];micro+=r['micro_backwards']
 cleanup=json.loads((d/'REGENERABLE_CACHE_CLEANUP.json').read_text());assert cleanup['all_requested_seeds_TRAIN_and_VAL_complete'] and cleanup['scientific_artifacts_preserved'] and cleanup['reference_max_abs']<=2e-5 and cleanup['retained_prefix_bytes']<=23000000000
assert (steps,micro,branches,curves,readouts)==(944,3696,24,72,24)
sft=Path(cfg['matched_reference_run'])/'HK';sr=json.loads((sft/'TRAIN_RECORDS.json').read_text());assert len(sr)==6 and sum(v['optimizer_steps'] for v in sr)==1180
for v in sr:
 z=torch.load(root/'HK/sft_reference/74002'/f"response{v['stage']}.private.pt",map_location='cpu',weights_only=False);m=next(row for row in agg['results'] if row['stage']==v['stage'])['metrics']['sft_reference'];yy=z['labels'];assert torch.bincount(yy[z['scores'].argmax(1)==yy],minlength=v['seen']).tolist()==m['correct']
for r in records:assert r['statistics']['KL_sum']>=-1e-4 and r['statistics']['mean_weight_sum']>=r['query_rows']
result=dict(passed=True,scope='HK seed74002 natural-versus-macro risk direction by estimator factorial',effective_optimizer_steps=steps,micro_backwards=micro,additional_guide_micro_backwards=sum(r['guide_micro_backwards'] for r in records),stage_arm_branches=branches,curve_records=curves,saved_response_checks=readouts+6,reused_supervised_reference_no_new_SFT=True,new_model_forwards=0,new_optimization=0,full_three_domain_confirmation=False)
p=root/'results';p.mkdir(exist_ok=True);(p/'CLOSEOUT_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
