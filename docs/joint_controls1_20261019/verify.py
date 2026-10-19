"""Independent score recount and Cholesky-trace df verification."""
import json,os,time
from pathlib import Path
import torch
start=time.time();torch.set_num_threads(4)
root=Path(os.environ['RUN_OUTPUT']);cfg=json.loads((root/'config.private.json').read_text())
assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE'
r=json.loads((root/'results.json').read_text());seal=json.loads((root/'FIT_SEAL.json').read_text())
assert seal['query_opened'] is False and len(seal['stages'])==6
prior=Path(cfg['prior_run']);matched=Path(cfg['matched_run']);joint=Path(cfg['joint_run'])
gc=torch.zeros(769,769,dtype=torch.float64);gp=gc.clone();cross=torch.zeros(768,768,dtype=torch.float64)
base_c=base_p=0;ends=[sum(cfg['task_sizes'][:j+1]) for j in range(6)];dferrors=[]
for i,c in enumerate(cfg['order']):
 cgram=torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True)['ridge_xx'];pgram=torch.load(matched/f'class_{c:02d}.private.pt',weights_only=True)['patch_mean_gram']
 gc+=cgram;gp+=pgram;cross+=torch.load(joint/f'cross_{c:02d}.private.pt',weights_only=True)['cross']
 if i<10:base_c+=float(cgram[:-1,:-1].trace())/10;base_p+=float(pgram[:-1,:-1].trace())/10
 if i+1 not in ends:continue
 stage=ends.index(i+1);audit=seal['stages'][stage];s=seal['patch_scale']
 assert abs(s-(base_c/base_p)**.5)<1e-12
 g=torch.zeros(1537,1537,dtype=torch.float64)
 g[:768,:768]=gc[:-1,:-1]/2;g[768:1536,768:1536]=gp[:-1,:-1]*s*s/2
 g[:768,768:1536]=cross*s/2;g[768:1536,:768]=cross.T*s/2
 g[:768,-1]=gc[:-1,-1]/2**.5;g[-1,:768]=gc[-1,:-1]/2**.5
 g[768:1536,-1]=gp[:-1,-1]*s/2**.5;g[-1,768:1536]=gp[-1,:-1]*s/2**.5;g[-1,-1]=gc[-1,-1]
 def trace_df(a,alpha):
  chol=torch.linalg.cholesky(a+alpha*torch.eye(len(a),dtype=a.dtype))
  return len(a)-alpha*float(torch.cholesky_inverse(chol).trace())
 target=trace_df(gc,1.);actual=trace_df(g,audit['matched_alpha']);scale_df=trace_df(g,1.)
 assert abs(target-audit['CLS_df'])<1e-6 and abs(actual-target)<1e-6 and abs(scale_df-audit['scale_joint_df'])<1e-6
 dferrors.append(abs(actual-target))
 if stage==0:assert abs(float(g[:-1,:-1].trace())-float(gc[:-1,:-1].trace()))<1e-7
count=0
for stage,record in enumerate(r):
 s=torch.load(root/f'stage_{stage}.private.pt',weights_only=True);old=sum(cfg['task_sizes'][:stage]);seen=s['seen'];labels=s['labels'].tolist()
 assert seen==cfg['order'][:old+cfg['task_sizes'][stage]]==record['seen']
 previous=torch.load(joint/f'stage_{stage}.private.pt',weights_only=True)
 assert torch.equal(s['labels'],previous['labels']) and len(s['scores'])==7
 pred={};ba={}
 for arm,score in s['scores'].items():
  assert torch.isfinite(score).all() and score.shape==(len(labels),len(seen))
  p=[seen[j] for j in score.argmax(1).tolist()];pred[arm]=p
  counts=[labels.count(c) for c in seen];assert counts==[15]*len(seen)
  correct=[sum(y==c and z==c for y,z in zip(labels,p)) for c in seen];recalls=[a/b for a,b in zip(correct,counts)]
  confusion=[[sum(y==c and z==d for y,z in zip(labels,p)) for d in seen] for c in seen]
  m=record['metrics'][arm];assert counts==m['counts'] and correct==m['correct'] and confusion==m['confusion']
  ba[arm]=sum(recalls)/len(seen)
  assert abs(ba[arm]-m['macro'])<1e-12 and abs(sum(correct)/len(labels)-m['micro'])<1e-12
  assert abs(sum(recalls[old:])/(len(seen)-old)-m['new_macro'])<1e-12
  if old:assert abs(sum(recalls[:old])/old-m['old_macro'])<1e-12
  else:assert m['old_macro'] is None
  if arm in previous['scores']:assert torch.equal(score,previous['scores'][arm])
  count+=len(labels)
 for name,d in record['contrasts'].items():
  a,b=name.split('-');p,q=pred[a],pred[b]
  assert abs(ba[a]-ba[b]-d['delta_BA'])<1e-12
  assert d['rescue']==sum(x==y and z!=y for x,z,y in zip(p,q,labels))
  assert d['harm']==sum(x!=y and z==y for x,z,y in zip(p,q,labels))
  assert d['changed']==sum(x!=z for x,z in zip(p,q))
 for name,d in record['class_sensitivity'].items():
  a,b=name.split('-');delta=[(x-y)/15 for x,y in zip(record['metrics'][a]['correct'],record['metrics'][b]['correct'])]
  assert list(d['per_class_delta'].values())==delta
  loo=[(sum(delta)-x)/(len(delta)-1) for x in delta]
  assert abs(min(loo)-d['leave_one_class_out_min'])<1e-12 and abs(max(loo)-d['leave_one_class_out_max'])<1e-12
cost=json.loads((root/'costs.json').read_text());term=json.loads((root.parent/'terminal.private.json').read_text())
assert term['returncode']==0 and term['elapsed_seconds']<cfg['max_wall_seconds']
assert cost['new_training_images']==cost['new_validation_images']==cost['optimizer_updates']==0 and cost['attempts']==1
size=sum(p.stat().st_size for p in root.iterdir() if p.is_file());assert size<cfg['output_budget_bytes']
result=dict(status='PASS',predictions_recounted=count,all_metrics_contrasts_and_class_sensitivity_verified=True,previous_five_scores_exactly_reused=True,df_verified_by_independent_Cholesky_inverse_trace=True,max_df_error=max(dferrors),base_energy_match_verified=True,budget_checks_passed=True,output_bytes=size,guardian_wall_seconds=term['elapsed_seconds'],wall_seconds=time.time()-start)
(root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
