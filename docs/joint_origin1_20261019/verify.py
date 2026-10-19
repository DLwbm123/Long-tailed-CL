"""Independent recount, normal-equation, Schur and trace checks; no fitting."""
import json,os,time
from pathlib import Path
import torch
start=time.time();torch.set_num_threads(4);root=Path(os.environ['RUN_OUTPUT']);cfg=json.loads((root/'config.private.json').read_text())
assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE'
prior=Path(cfg['prior_run']);matched=Path(cfg['matched_run']);joint=Path(cfg['joint_run']);control=Path(cfg['control_run'])
records=json.loads((root/'results.json').read_text());diag=json.loads((root/'DIAGNOSTICS.json').read_text());traj=json.loads((root/'TRAJECTORIES.json').read_text());freq=json.loads((root/'FREQUENCY.json').read_text());seal=json.loads((root/'FIT_SEAL.json').read_text())
assert seal['query_opened'] is False and len(records)==len(diag)==6
sur=torch.load(root/'surrogate.private.pt',weights_only=True);mapping=sur['mapping'];transform=sur['transform']
gc=torch.zeros(769,769,dtype=torch.float64);um=torch.zeros(769,768,dtype=torch.float64);targets=[];classgs=[];patchgs=[];ends=[sum(cfg['task_sizes'][:i+1]) for i in range(6)];dferror=[];equation_error=[]
for i,c in enumerate(cfg['order']):
 old=torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True);pm=torch.load(matched/f'class_{c:02d}.private.pt',weights_only=True);cross=torch.load(joint/f'cross_{c:02d}.private.pt',weights_only=True)['cross']
 classgs.append(old['ridge_xx']);patchgs.append(pm['patch_mean_gram']);gc+=old['ridge_xx'];um+=torch.cat([cross,pm['patch_mean_target'][:-1][None,:]],0);targets.append(old['ridge_xy'])
 if i==9:
  error=float(((gc+torch.eye(769,dtype=torch.float64)*cfg['surrogate_rho'])@mapping-um).abs().max());assert error<1e-7
  ec=float(gc[:-1,:-1].trace())/10;eh=float((mapping.T@gc@mapping).trace())/10
  assert abs(seal['surrogate_scale']-(ec/eh)**.5)<1e-10
  t=torch.zeros(769,1537,dtype=torch.float64);t[:768,:768]=torch.eye(768,dtype=torch.float64)/2**.5;t[:,768:1536]=mapping*seal['surrogate_scale']/2**.5;t[-1,-1]=1
  assert torch.equal(t,transform)
 if i+1 not in ends:continue
 stage=ends.index(i+1);a=seal['stages'][stage];h=torch.load(root/f'heads_{stage}.private.pt',weights_only=True);b=torch.stack(targets,1)
 penalties=torch.ones(769,dtype=torch.float64)*2*a['joint_alpha'];penalties[-1]=a['joint_alpha'];q=gc+torch.diag(penalties)
 e=float((q@h['cls_same_block']-b).abs().max());assert e<1e-7;equation_error.append(e)
 assert float((q@h['P']-um).abs().max())<1e-7
 sg=transform.T@gc@transform;alpha=a['surrogate_alpha'];full=h['surrogate_full_head']
 assert float(((sg+alpha*torch.eye(1537,dtype=torch.float64))@full-transform.T@b).abs().max())<1e-7
 assert float((transform@full-h['predicted_patch_joint']).abs().max())<1e-10
 def df(g,alpha):return len(g)-alpha*float(torch.cholesky_inverse(torch.linalg.cholesky(g+alpha*torch.eye(len(g),dtype=torch.float64))).trace())
 target=df(gc,1.);actual=df(sg,alpha);assert abs(target-a['CLS_df'])<1e-6 and abs(actual-target)<1e-6;dferror.append(abs(actual-target))
 for name,gs in [('CLS_energy',classgs),('patch_energy',patchgs)]:
  means=torch.stack([g[:-1,-1] for g in gs]);total=sum(float(g[:-1,:-1].trace()) for g in gs)/len(gs);globalmu=float(means.mean(0).square().sum());classmu=float(means.square().sum(1).mean());expected=dict(total=total,global_mean_energy=globalmu,between_class_mean_energy=classmu-globalmu,within_class_energy=total-classmu)
  for k,v in expected.items():assert abs(a[name][k]-v)<1e-7
query=torch.load(joint/'query768.private.pt',weights_only=True);count=0;schurmax=0;previous=None
for stage,r in enumerate(records):
 s=torch.load(root/f'stage_{stage}.private.pt',weights_only=True);seen=s['seen'];labels=s['labels'].tolist();old=sum(cfg['task_sizes'][:stage]);assert seen==cfg['order'][:old+cfg['task_sizes'][stage]]==r['seen']
 earlier=torch.load(control/f'stage_{stage}.private.pt',weights_only=True);assert torch.equal(s['labels'],earlier['labels']) and len(s['scores'])==9
 pred={};bas={}
 for arm,score in s['scores'].items():
  assert torch.isfinite(score).all() and score.shape==(len(labels),len(seen))
  p=[seen[j] for j in score.argmax(1).tolist()];pred[arm]=p;counts=[labels.count(c) for c in seen];assert counts==[15]*len(seen)
  correct=[sum(y==c and z==c for y,z in zip(labels,p)) for c in seen];rec=[a/b for a,b in zip(correct,counts)];m=r['metrics'][arm]
  confusion=[[sum(y==c and z==d for y,z in zip(labels,p)) for d in seen] for c in seen]
  assert counts==m['counts'] and correct==m['correct'] and confusion==m['confusion'];bas[arm]=sum(rec)/len(seen)
  assert abs(bas[arm]-m['macro'])<1e-12 and abs(sum(correct)/len(labels)-m['micro'])<1e-12
  assert abs(sum(rec[old:])/(len(seen)-old)-m['new_macro'])<1e-12
  if old:assert abs(sum(rec[:old])/old-m['old_macro'])<1e-12
  else:assert m['old_macro'] is None
  if arm in earlier['scores']:assert torch.equal(score,earlier['scores'][arm])
  count+=len(labels)
 for name,d in r['contrasts'].items():
  a,b=name.split('-');p,q=pred[a],pred[b];assert abs(bas[a]-bas[b]-d['delta_BA'])<1e-12
  for k,value in [('rescue',sum(x==y and z!=y for x,z,y in zip(p,q,labels))),('harm',sum(x!=y and z==y for x,z,y in zip(p,q,labels))),('changed',sum(x!=z for x,z in zip(p,q))),('both_wrong',sum(x!=y and z!=y for x,z,y in zip(p,q,labels)))]:assert value==d[k]
 groups={name:[] for name in ['both_correct','CLS_only','patch_only','both_wrong']}
 for i,y in enumerate(labels):
  a=pred['cls768_ridge'][i]==y;b=pred['patchmean768_ridge'][i]==y;name='both_correct' if a and b else 'CLS_only' if a else 'patch_only' if b else 'both_wrong';groups[name].append(i)
 for name,idx in groups.items():
  assert len(idx)==diag[stage]['groups'][name]['count']
  for arm,p in pred.items():assert sum(p[i]==labels[i] for i in idx)==diag[stage]['groups'][name]['correct'][arm]
 if previous is not None:
  idx=[i for i,y in enumerate(labels) if y in previous['seen']];ys=[labels[i] for i in idx];assert ys==previous['labels'].tolist()
  for arm,score in s['scores'].items():
   pold=[seen[j] for j in score[idx,:old].argmax(1).tolist()];pall=[pred[arm][i] for i in idx];pp=[seen[j] for j in previous['scores'][arm].argmax(1).tolist()]
   def macro(p):return sum(sum(y==c and z==c for y,z in zip(ys,p))/ys.count(c) for c in seen[:old])/old
   before,internal,allba=macro(pp),macro(pold),macro(pall);d=diag[stage]['competition'][arm]
   expected=dict(previous_old_BA=before,current_restricted_old_BA=internal,current_all_old_BA=allba,internal_change=internal-before,competition_loss=internal-allba,total_change=allba-before,new_class_stolen_correct=sum(x==y and z!=y for x,z,y in zip(pold,pall,ys)))
   for k,v in expected.items():assert abs(v-d[k])<1e-12
 positions={c:0 for c in seen};cs=[];ms=[]
 for c in labels:
  j=positions[c];positions[c]+=1;cs.append(query[c][0][j]);ms.append(query[c][1][j])
 u=torch.cat([torch.stack(cs),torch.ones(len(cs),1,dtype=torch.float64)],1);m=torch.stack(ms);h=torch.load(root/f'heads_{stage}.private.pt',weights_only=True)
 for arm in cfg['new_arms']:assert float((u@h[arm]-s['scores'][arm]).abs().max())<cfg['score_atol']
 e=float((u@h['cls_same_block']+(m-u@h['P'])@h['WM']-s['scores']['df_matched_joint']).abs().max());assert e<cfg['score_atol'];schurmax=max(e,schurmax);previous=s
for row in traj:
 c=row['class_id'];born=row['birth_stage'];assert born==next(i for i,r in enumerate(records) if c in r['seen'])
 for arm,d in row['arms'].items():
  values=[r['metrics'][arm]['recall'][r['seen'].index(c)] for r in records[born:]];refs=[r['metrics']['cls768_ridge']['recall'][r['seen'].index(c)] for r in records[born:]];adv=[a-b for a,b in zip(values,refs)]
  assert values==d['recalls'] and adv==d['advantage_over_CLS'] and abs(adv[-1]-adv[0]-d['advantage_change'])<1e-12
for criterion,data in freq['groups'].items():
 ordered=sorted(cfg['order'],key=lambda c:(freq['support'][str(c)][criterion],c));assert data['bins']==dict(tail=ordered[:7],middle=ordered[7:13],head=ordered[13:])
 for arm,values in data['BA'].items():
  for name,members in data['bins'].items():assert abs(sum(records[-1]['metrics'][arm]['recall'][cfg['order'].index(c)] for c in members)/len(members)-values[name])<1e-12
cost=json.loads((root/'costs.json').read_text());terminal=json.loads((root.parent/'terminal.private.json').read_text());assert terminal['returncode']==0 and terminal['elapsed_seconds']<cfg['max_wall_seconds'];assert cost['new_training_images']==cost['new_validation_images']==cost['optimizer_updates']==0 and cost['attempts']==1
size=sum(p.stat().st_size for p in root.iterdir() if p.is_file());assert size<cfg['output_budget_bytes']
result=dict(status='PASS',predictions_recounted=count,normal_equations_verified=True,Schur_max_error=schurmax,surrogate_df_max_error=max(dferror),groups_competition_trajectories_frequency_energy_verified=True,previous_seven_scores_exactly_reused=True,budget_checks_passed=True,output_bytes=size,guardian_wall_seconds=terminal['elapsed_seconds'],wall_seconds=time.time()-start)
(root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
