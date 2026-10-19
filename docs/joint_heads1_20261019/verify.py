"""Recount saved scores independently; no new fitting or image extraction."""
import json,os,sys,time
from pathlib import Path
import torch
start=time.time();root=Path(os.environ['RUN_OUTPUT'])
cfg=json.loads((root/'config.private.json').read_text())
sys.path.insert(0,str(root.parents[1]/'case_hierarchical1_20261018'))
from data import read_rows
val=read_rows(cfg,'val')
assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE'
records=json.loads((root/'results.json').read_text());assert len(records)==6
arms=set(cfg['arms']);total=0;max_error=0;coverage=[]
pairs=[('joint768_ridge','cls768_ridge'),('joint768_ridge','duplicate_raw'),('duplicate_raw','cls768_ridge'),('duplicate_calibrated','cls768_ridge')]
for stage,r in enumerate(records):
 s=torch.load(root/f'stage_{stage}.private.pt',weights_only=True)
 seen=s['seen'];labels=s['labels'].tolist();old=sum(cfg['task_sizes'][:stage])
 assert seen==cfg['order'][:old+cfg['task_sizes'][stage]]==r['seen']
 rows=[x for x in val if x['label'] in seen];assert [x['label'] for x in rows]==labels
 assert set(s['scores'])==arms
 pred={};bas={}
 for arm,scores in s['scores'].items():
  assert torch.isfinite(scores).all() and scores.shape==(len(labels),len(seen))
  p=[seen[j] for j in scores.argmax(1).tolist()];pred[arm]=p
  counts=[labels.count(c) for c in seen];assert counts==[15]*len(seen)
  correct=[sum(y==c and q==c for y,q in zip(labels,p)) for c in seen]
  recalls=[a/b for a,b in zip(correct,counts)]
  confusion=[[sum(y==c and q==d for y,q in zip(labels,p)) for d in seen] for c in seen]
  m=r['metrics'][arm]
  assert counts==m['counts'] and correct==m['correct'] and confusion==m['confusion']
  bas[arm]=sum(recalls)/len(seen)
  for k,v in [('macro',bas[arm]),('micro',sum(correct)/len(labels)),('new_macro',sum(recalls[old:])/(len(seen)-old))]:assert abs(v-m[k])<1e-12
  if old:assert abs(sum(recalls[:old])/old-m['old_macro'])<1e-12
  else:assert m['old_macro'] is None
  total+=len(labels)
 for a,b in pairs:
  p,q=pred[a],pred[b];expected=r['contrasts'][a+'-'+b]
  rescue=[i for i,(x,z,y) in enumerate(zip(p,q,labels)) if x==y and z!=y]
  harm=[i for i,(x,z,y) in enumerate(zip(p,q,labels)) if x!=y and z==y]
  assert len(rescue)==expected['rescue'] and len(harm)==expected['harm']
  assert sum(x!=z for x,z in zip(p,q))==expected['changed']
  assert abs(bas[a]-bas[b]-expected['delta_BA'])<1e-12
  if a=='joint768_ridge':
   rc=[sum(labels[i]==c for i in rescue) for c in seen];hc=[sum(labels[i]==c for i in harm) for c in seen]
   rg={rows[i]['group'] for i in rescue};hg={rows[i]['group'] for i in harm}
   groups={x['group'] for x in rows}
   net={g:sum(rows[i]['group']==g for i in rescue)-sum(rows[i]['group']==g for i in harm) for g in groups}
   coverage.append(dict(stage=stage,comparison=a+'-'+b,patients=len(groups),rescue_patients=len(rg),harm_patients=len(hg),patients_with_both=len(rg&hg),net_positive_patients=sum(v>0 for v in net.values()),net_negative_patients=sum(v<0 for v in net.values()),net_zero_patients=sum(v==0 for v in net.values()),rescue_classes=sum(v>0 for v in rc),harm_classes=sum(v>0 for v in hc),net_positive_classes=sum(x>y for x,y in zip(rc,hc)),net_negative_classes=sum(x<y for x,y in zip(rc,hc)),per_class=[dict(class_id=c,rescue=x,harm=y,net=x-y) for c,x,y in zip(seen,rc,hc)]))
 previous=torch.load(Path(cfg['matched_run'])/f'stage_{stage}.private.pt',weights_only=True)
 assert previous['seen']==seen and previous['labels'].tolist()==labels
 for arm in ['cls768_ridge','patchmean768_ridge']:
  e=float((s['scores'][arm]-previous['scores'][arm]).abs().max());max_error=max(max_error,e)
  assert e<cfg['score_atol'] and torch.equal(s['scores'][arm].argmax(1),previous['scores'][arm].argmax(1))
 assert torch.equal(s['scores']['duplicate_calibrated'],s['scores']['cls768_ridge'])
cost=json.loads((root/'costs.json').read_text());term=json.loads((root.parent/'terminal.private.json').read_text())
assert term['returncode']==0 and term['elapsed_seconds']<=cfg['max_wall_seconds']
assert cost['new_training_images']==cfg['train_count'] and cost['new_validation_images']==cfg['val_count']
assert cost['GPU_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes'] and cost['attempts']==1 and cost['optimizer_updates']==0
size=sum(p.stat().st_size for p in root.iterdir() if p.is_file());assert size<cfg['output_budget_bytes']
seal=json.loads((root/'FIT_SEAL.json').read_text());parity=json.loads((root/'PARITY.json').read_text())
assert seal['query_opened'] is False and seal['fitted_classes']==20 and seal['stages']==6
assert max(v for d in seal['marginal_parity'].values() for v in d.values())<cfg['feature_atol']
assert max(parity['validation_CLS'].values())<=cfg['feature_atol']
(root/'COVERAGE.json').write_text(json.dumps(coverage,indent=2)+'\n')
result=dict(status='PASS',stages=6,arms=5,predictions_recounted=total,metrics_counts_confusion_old_new_and_four_contrasts_verified=True,historical_max_score_error=max_error,calibrated_duplicate_exact_baseline_scores=True,calibrated_duplicate_is_analytic_not_independent_fit=True,budget_checks_passed=True,guardian_wall_seconds=term['elapsed_seconds'],output_bytes=size,wall_seconds=time.time()-start)
(root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));print(json.dumps(coverage[-2:]))
