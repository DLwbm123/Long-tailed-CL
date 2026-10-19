"""Independent OOF risk and evaluation recount; no image encoding or fitting."""
import json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
start=time.time();torch.set_num_threads(4);root=Path(os.environ['RUN_OUTPUT']);cfg=json.loads((root/'config.private.json').read_text());assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE'
records=json.loads((root/'results.json').read_text());audit=json.loads((root/'TRAINING_AUDIT.json').read_text());seal=json.loads((root/'FIT_SEAL.json').read_text());assert seal['query_opened'] is False and len(records)==len(audit)==6
index=np.load(root/'train_index.private.npz');groups=index['groups'];folds=index['folds'];trainlabels=index['labels'];masses=index['weights'];bounds=index['bounds'];assert len(groups)==len(folds)==68058
fold_map={}
for g,f in zip(groups.tolist(),folds.tolist()):assert fold_map.setdefault(g,f)==f
for ci in range(20):
 ids=np.flatnonzero(trainlabels==ci);assert abs(masses[ids].sum()-1)<1e-10
 counts=Counter(groups[ids].tolist());expected=np.array([1/(len(counts)*counts[g]) for g in groups[ids].tolist()]);assert np.max(np.abs(masses[ids]-expected))<1e-12
max_risk_error=0.;max_constraint=0.;solver_iterations=[]
for stage,a in enumerate(audit):
 k=sum(cfg['task_sizes'][:stage+1]);n=bounds[k];o=np.load(root/f'OOF_{stage}.private.npz');c=o['CLS'];j=o['joint'];ys=o['labels'];w=o['weights'];fs=o['folds'];gs=o['groups']
 assert np.array_equal(ys,trainlabels[:n]) and np.array_equal(fs,folds[:n]) and np.array_equal(gs,groups[:n]) and np.allclose(w,masses[:n],atol=1e-15)
 assert c.shape==j.shape==(n,k) and np.isfinite(c).all() and np.isfinite(j).all()
 c=c-c.mean(1,keepdims=True);d=j-o['CLS'];d-=d.mean(1,keepdims=True);target=np.eye(k)[ys];target-=target.mean(1,keepdims=True)
 def risk(alpha,mask=None):
  if mask is None:mask=np.ones(n,dtype=bool)
  f=c[mask]+d[mask]*alpha;f-=f.mean(1,keepdims=True)
  return float(((f-target[mask])**2*w[mask,None]).sum()/(w[mask].sum()*k))
 zero=risk(0.);scalar=a['uniform_alpha'];alpha=np.array(a['risk_limited_alpha']);assert alpha.shape==(k,) and alpha.min()>=-1e-10 and alpha.max()<=1+1e-10 and 0<=scalar<=1
 e=c-target;den=float((d*d*w[:,None]).sum());u=float(np.clip(-(e*d*w[:,None]).sum()/den,0,1)) if den>1e-14*w.sum()*k else 0.
 assert abs(u-scalar)<1e-9
 for value,key in [(zero,'OOF_CLS_risk'),(risk(scalar),'OOF_uniform_risk'),(risk(alpha),'OOF_limited_risk')]:
  er=abs(value-a[key]);max_risk_error=max(er,max_risk_error);assert er<1e-10
 assert risk(alpha)<=zero+1e-7
 correct=o['CLS'].argmax(1)==ys;expected_constraints=[];expected_skips=[]
 for fold in range(3):
  for ci in range(k):
   for kind,mask in [('class',(fs==fold)&(ys==ci)),('CLS_correct',(fs==fold)&(ys==ci)&correct)]:
    npat=len(np.unique(gs[mask]));images=int(mask.sum());entry=dict(fold=fold,class_id=cfg['order'][ci],kind=kind,images=images,patients=npat)
    if images==0 or (kind=='CLS_correct' and npat<2):expected_skips.append(entry)
    else:
     r0=risk(0.,mask);delta=risk(alpha,mask)-r0;normalized=delta/max(r0,1e-8);max_constraint=max(normalized,max_constraint);assert normalized<=cfg['feasibility_tolerance']+1e-9
     expected_constraints.append((entry,r0,delta,normalized))
 assert expected_skips==a['skipped_constraints'] and len(expected_constraints)==len(a['constraints'])
 for (meta,r0,delta,norm),saved in zip(expected_constraints,a['constraints']):
  assert all(saved[key]==val for key,val in meta.items());assert abs(r0-saved['baseline_risk'])<1e-10 and abs(delta-saved['risk_delta'])<1e-10 and abs(norm-saved['normalized_delta'])<1e-7
 assert a['solver']['success'];solver_iterations.append(a['solver']['iterations'])
 for ci in [cfg['order'].index(x) for x in a['fixed_zero_classes']]:assert abs(alpha[ci])<1e-10
count=0;group_diag=json.loads((root/'GROUP_DIAGNOSTICS.json').read_text())
for stage,record in enumerate(records):
 s=torch.load(root/f'stage_{stage}.private.pt',weights_only=True);before=torch.load(Path(cfg['score_run'])/f'stage_{stage}.private.pt',weights_only=True);seen=s['seen'];y=s['labels'].tolist();old=sum(cfg['task_sizes'][:stage]);k=old+cfg['task_sizes'][stage]
 assert seen==cfg['order'][:k]==record['seen'] and torch.equal(s['labels'],before['labels']) and len(s['scores'])==13
 c=before['scores']['cls768_ridge'].numpy();c=c-c.mean(1,keepdims=True);d=before['scores']['scale_joint'].numpy()-before['scores']['cls768_ridge'].numpy();d=d-d.mean(1,keepdims=True)
 for arm,alpha in [('uniform_shrink',audit[stage]['uniform_alpha']),('risk_limited_shrink',np.array(audit[stage]['risk_limited_alpha']))]:
  expected=c+d*alpha;expected-=expected.mean(1,keepdims=True);assert np.max(np.abs(s['scores'][arm].numpy()-expected))<cfg['score_atol']
 assert np.array_equal(c.argmax(1),before['scores']['cls768_ridge'].argmax(1).numpy()) and np.array_equal((c+d).argmax(1),before['scores']['scale_joint'].argmax(1).numpy())
 pred={};bas={}
 for arm,score in s['scores'].items():
  assert torch.isfinite(score).all() and score.shape==(len(y),k)
  if arm in before['scores']:assert torch.equal(score,before['scores'][arm])
  p=[seen[j] for j in score.argmax(1).tolist()];pred[arm]=p;counts=[y.count(c) for c in seen];assert counts==[15]*k
  correct=[sum(t==c and z==c for t,z in zip(y,p)) for c in seen];rec=[a/b for a,b in zip(correct,counts)];m=record['metrics'][arm];conf=[[sum(t==c and z==d for t,z in zip(y,p)) for d in seen] for c in seen]
  assert counts==m['counts'] and correct==m['correct'] and conf==m['confusion'];bas[arm]=sum(rec)/k
  assert abs(bas[arm]-m['macro'])<1e-12 and abs(sum(correct)/len(y)-m['micro'])<1e-12 and abs(sum(rec[old:])/(k-old)-m['new_macro'])<1e-12
  if old:assert abs(sum(rec[:old])/old-m['old_macro'])<1e-12
  else:assert m['old_macro'] is None
  count+=len(y)
 for name,a in record['contrasts'].items():
  ca,cb=name.split('-');p,q=pred[ca],pred[cb];assert abs(bas[ca]-bas[cb]-a['delta_BA'])<1e-12
  assert a['rescue']==sum(z==t and v!=t for z,v,t in zip(p,q,y)) and a['harm']==sum(z!=t and v==t for z,v,t in zip(p,q,y)) and a['changed']==sum(z!=v for z,v in zip(p,q))
 for name,di in group_diag[stage]['groups'].items():
  ids=[]
  for i,t in enumerate(y):
   a=pred['cls768_ridge'][i]==t;b=pred['patchmean768_ridge'][i]==t;group='both_correct' if a and b else 'CLS_only' if a else 'patch_only' if b else 'both_wrong'
   if group==name:ids.append(i)
  assert len(ids)==di['images']
  for arm,p in pred.items():assert sum(p[i]==y[i] for i in ids)==di['correct'][arm]
freq=json.loads((root/'FREQUENCY.json').read_text());coh=json.loads((root/'COHORTS.json').read_text())
for data in freq.values():
 for arm,values in data['BA'].items():
  for name,cs in data['bins'].items():assert abs(sum(records[-1]['metrics'][arm]['recall'][cfg['order'].index(c)] for c in cs)/len(cs)-values[name])<1e-12
for arm,x in coh.items():assert x==dict(base_correct=sum(records[-1]['metrics'][arm]['correct'][:10]),later_correct=sum(records[-1]['metrics'][arm]['correct'][10:]))
cost=json.loads((root/'costs.json').read_text());terminal=json.loads((root.parent/'resume_terminal.private.json').read_text());assert terminal['returncode']==0 and terminal['elapsed_seconds']+cost['prior_and_repair_wall_seconds']<=cfg['max_wall_seconds'];assert cost['new_training_images']==68058 and cost['new_validation_images']==cost['backbone_optimizer_updates']==0 and cost['fold_head_completed']==cost['fold_head_attempts']==36 and cost['shrink_solver_calls']==7 and cost['attempts']==2
assert cost['GPU_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes'] and cost['RSS_peak_bytes']<=cfg['host_RSS_budget_bytes'];size=sum(p.stat().st_size for directory in [root,root.parent/'solver_deps'] for p in directory.rglob('*') if p.is_file());assert size<=cfg['output_budget_bytes']
assert cost['continuation_new_training_images']==cost['continuation_new_validation_images']==0
assert json.loads((root/'failure_attempt1/status.private.json').read_text())['state']=='FAILED'
assert cost['wall_seconds']<=cfg['max_wall_seconds']
for stage,a in enumerate(audit):
 if stage>=2:
  s=a['solver'];assert s['solver']=='Clarabel' and s['status']=='Solved' and s['settings']['tol_gap_abs']==cfg['solver_ftol'] and s['settings']['tol_feas']==cfg['solver_ftol']
  assert s['PSD_factor_max_error']<=1e-12 and s['raw_bound_polish_max']<=1e-10
result=dict(previous_failure_retained=True,no_continuation_encodes_verified=True,status='PASS',predictions_recounted=count,OOF_risks_constraints_and_coefficients_verified=True,patient_fold_and_weights_verified=True,max_risk_recount_error=max_risk_error,max_normalized_constraint_delta=max_constraint,solver_iterations=solver_iterations,previous_eleven_scores_exactly_reused=True,all_metrics_groups_frequency_and_cohorts_verified=True,budget_checks_passed=True,output_bytes=size,guardian_wall_seconds=terminal['elapsed_seconds'],wall_seconds=time.time()-start)
(root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
