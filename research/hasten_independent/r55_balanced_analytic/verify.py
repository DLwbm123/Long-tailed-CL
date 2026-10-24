"""Verify saved responses and training receipts; no model/feature forward."""
import os,json
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);assert json.loads((r/'CONTROL_EXIT.json').read_text())['exit_code']==0;checks=0;records=0;steps=0
for ds in ['NIH','ISIC','HK']:
 p=r/ds;a=json.loads((p/'aggregate.json').read_text());t=json.loads((p/'TRAIN_RECORDS.json').read_text());assert len(t)==len(a['results']);steps+=a['optimizer_steps']
 for q in t:
  assert all(q[k] for k in ['old_Q_exact','frozen_map_exact','frozen_R_exact','strict_restore_predictions_exact']);assert q['VAL_reads']==0 and q['class_mass']==[1.]*len(q['class_mass'])
  assert all(v['normal_equation_relative_residual']<1e-6 for v in q['audit'].values());records+=1
 for q in a['results']:
  x=torch.load(p/str(q['seed'])/f"responses{q['stage']}.private.pt",map_location='cpu',weights_only=False);y=x['labels']
  for k,s in x['scores'].items():
   assert torch.isfinite(s).all();pred=s.argmax(1);correct=torch.bincount(y[pred==y],minlength=q['seen']).tolist();counts=torch.bincount(y,minlength=q['seen']).tolist();assert correct==q['metrics'][k]['correct'] and counts==q['metrics'][k]['counts'];checks+=1
 for seed in [74002,74003,74004]:
  curves=json.loads((p/str(seed)/'BASE_CURVES.json').read_text());assert [z['epoch'] for z in curves]==list(range(21))
assert checks==90 and records==45 and steps==63900
(r/'results/RESPONSE_VERIFICATION.json').write_text(json.dumps(dict(saved_response_checks=checks,TRAIN_stage_receipts=records,base_models=9,epochs_each=20,optimizer_steps=steps,new_model_forwards=0,all_finite=True,all_classcounts_exact=True),indent=2));print('ALL_90_SAVED_RESPONSES_AND45_TRAIN_RECEIPTS_VERIFIED')
