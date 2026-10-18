"""Independently recount saved scores; does not fit models or access images."""
import json
import os
from pathlib import Path
import time
import torch

start=time.time()
root=Path(os.environ['RUN_OUTPUT'])
cfg=json.loads((root/'config.private.json').read_text())
assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE'
records=json.loads((root/'results.json').read_text())
assert len(records)==6
contrasts=[('cls128_ridge','cls768_ridge'),('patchmean128_ridge','patchmean768_ridge'),
 ('patchmean768_ridge','cls768_ridge'),('patchmean128_ridge','cls128_ridge'),
 ('patchmean128_gda','patchmean128_ridge'),('cls128_gda','cls128_ridge')]
expected_arms={x for pair in contrasts for x in pair}|{'patchgaussian128_historical'}
count=0;max_error=0;contrast_summary={}
for stage,r in enumerate(records):
 s=torch.load(root/f'stage_{stage}.private.pt',weights_only=True)
 seen=s['seen'];labels=s['labels'].tolist()
 assert seen==cfg['order'][:sum(cfg['task_sizes'][:stage+1])]==r['seen']
 assert set(s['scores'])==expected_arms
 old=sum(cfg['task_sizes'][:stage]);pred={};ba={}
 for arm,scores in s['scores'].items():
  assert scores.shape==(15*len(seen),len(seen)) and torch.isfinite(scores).all()
  p=[seen[j] for j in scores.argmax(1).tolist()];pred[arm]=p
  counts=[labels.count(c) for c in seen]
  assert counts==[15]*len(seen)
  correct=[sum(y==c and q==c for y,q in zip(labels,p)) for c in seen]
  confusion=[[sum(y==c and q==d for y,q in zip(labels,p)) for d in seen] for c in seen]
  recalls=[a/b for a,b in zip(correct,counts)];m=r['metrics'][arm]
  assert counts==m['counts'] and correct==m['correct'] and confusion==m['confusion']
  ba[arm]=sum(recalls)/len(seen)
  for key,value in [('macro',ba[arm]),('micro',sum(correct)/len(labels)),('new_macro',sum(recalls[old:])/(len(seen)-old))]:
   assert abs(value-m[key])<1e-12
  assert (m['old_macro'] is None) if old==0 else abs(sum(recalls[:old])/old-m['old_macro'])<1e-12
  count+=len(labels)
 for a,b in contrasts:
  name=a+'-'+b;p,q=pred[a],pred[b];t=r['contrasts'][name]
  assert abs(t['delta_BA']-(ba[a]-ba[b]))<1e-12
  assert t['rescue']==sum(x==y and z!=y for x,z,y in zip(p,q,labels))
  assert t['harm']==sum(x!=y and z==y for x,z,y in zip(p,q,labels))
  assert t['changed']==sum(x!=z for x,z in zip(p,q))
  contrast_summary.setdefault(name,[]).append(100*(ba[a]-ba[b]))
 h=torch.load(Path(cfg['prior_run'])/f'stage_{stage}.private.pt',weights_only=True)
 assert h['seen']==seen and h['labels'].tolist()==labels
 error=float((s['scores']['cls768_ridge']-h['scores']['ridge']).abs().max());max_error=max(error,max_error)
 assert error<cfg['score_atol']
 assert s['scores']['cls768_ridge'].argmax(1).tolist()==h['scores']['ridge'].argmax(1).tolist()
 assert torch.equal(s['scores']['patchgaussian128_historical'],h['scores']['uniform'])
cost=json.loads((root/'costs.json').read_text())
terminal=json.loads((root.parent/'terminal.private.json').read_text())
assert terminal['returncode']==0 and terminal['elapsed_seconds']<=cfg['max_wall_seconds']
assert cost['new_training_images']==cfg['train_count'] and cost['new_validation_images']==cfg['val_count']
assert cost['GPU_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes']
assert cost['optimizer_updates']==0 and cost['attempts']==1
output_bytes=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
assert output_bytes<cfg['output_budget_bytes']
seal=json.loads((root/'FIT_SEAL.json').read_text());parity=json.loads((root/'PARITY.json').read_text())
assert seal['query_opened'] is False and seal['fitted_classes']==20
assert max(seal['train_CLS_max_abs_error'].values())<=cfg['feature_atol']
assert max(v[k] for v in parity['validation'].values() for k in ['CLS_max_abs','patch_mean128_max_abs'])<=cfg['feature_atol']
result=dict(status='PASS',predictions_recounted=count,stages=6,arms=7,
 metrics_confusions_old_new_BA_and_six_contrasts_verified=True,historical_ridge_max_score_error=max_error,
 historical_patch_scores_reproduced=True,budget_checks_passed=True,output_bytes=output_bytes,
 guardian_wall_seconds=terminal['elapsed_seconds'],contrasts_stage_delta_pp=contrast_summary,
 contrasts_mean_stage_delta_pp={k:sum(v)/len(v) for k,v in contrast_summary.items()},
 wall_seconds=time.time()-start)
(root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
