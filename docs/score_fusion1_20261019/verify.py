"""Independent training-scale, fusion, metric and overlap recount."""
import json,os,time
from pathlib import Path
import torch
start=time.time();torch.set_num_threads(4);root=Path(os.environ['RUN_OUTPUT']);cfg=json.loads((root/'config.private.json').read_text());assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE'
prior=Path(cfg['prior_run']);matched=Path(cfg['matched_run']);joint=Path(cfg['joint_run']);origin=Path(cfg['origin_run'])
r=json.loads((root/'results.json').read_text());seal=json.loads((root/'FIT_SEAL.json').read_text());overs=json.loads((root/'OVERLAP.json').read_text());frequency=json.loads((root/'FREQUENCY.json').read_text());assert seal['query_opened'] is False and seal['head_fits']==0 and len(r)==6
cg=torch.zeros(769,769,dtype=torch.float64);mg=cg.clone();ends=[sum(cfg['task_sizes'][:i+1]) for i in range(6)];max_scale_error=0
for i,c in enumerate(cfg['order']):
 cg+=torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True)['ridge_xx'];mg+=torch.load(matched/f'class_{c:02d}.private.pt',weights_only=True)['patch_mean_gram']
 if i+1 not in ends:continue
 stage=ends.index(i+1);h=torch.load(joint/f'heads_{stage}.private.pt',weights_only=True);a=seal['training_scales'][stage];assert a['classes']==i+1
 for k,g,key in [('cls',cg,'sigma_CLS'),('patch',mg,'sigma_patch')]:
  w=h[k]-h[k].mean(1,keepdim=True);actual=float(((g@w)*w).sum()/(i+1)**2)**.5;e=abs(actual-a[key]);max_scale_error=max(e,max_scale_error);assert e<1e-9
count=0;max_score_error=0
for stage,record in enumerate(r):
 s=torch.load(root/f'stage_{stage}.private.pt',weights_only=True);before=torch.load(origin/f'stage_{stage}.private.pt',weights_only=True);labels=s['labels'].tolist();seen=s['seen'];old=sum(cfg['task_sizes'][:stage]);assert seen==cfg['order'][:old+cfg['task_sizes'][stage]]==record['seen'] and torch.equal(s['labels'],before['labels']);assert len(s['scores'])==11
 c=before['scores']['cls768_ridge'];m=before['scores']['patchmean768_ridge'];a=seal['training_scales'][stage]
 expected=dict(score_equal=(c+m)*.5,score_train_rms=(c/a['sigma_CLS']+m/a['sigma_patch'])*.5)
 # The RMS implementation removes a per-image common score; compare centered outputs.
 pred={};ba={}
 for arm,score in s['scores'].items():
  assert torch.isfinite(score).all() and score.shape==(len(labels),len(seen))
  if arm in before['scores']:assert torch.equal(score,before['scores'][arm])
  else:
   target=expected[arm]
   if arm=='score_train_rms':target=target-target.mean(1,keepdim=True)
   e=float((score-target).abs().max());max_score_error=max(e,max_score_error);assert e<cfg['score_atol']
  p=[seen[j] for j in score.argmax(1).tolist()];pred[arm]=p;counts=[labels.count(c) for c in seen];assert counts==[15]*len(seen)
  correct=[sum(y==c and z==c for y,z in zip(labels,p)) for c in seen];rec=[x/y for x,y in zip(correct,counts)];met=record['metrics'][arm]
  conf=[[sum(y==c and z==d for y,z in zip(labels,p)) for d in seen] for c in seen]
  assert counts==met['counts'] and correct==met['correct'] and conf==met['confusion'];ba[arm]=sum(rec)/len(seen)
  assert abs(ba[arm]-met['macro'])<1e-12 and abs(sum(correct)/len(labels)-met['micro'])<1e-12
  assert abs(sum(rec[old:])/(len(seen)-old)-met['new_macro'])<1e-12
  if old:assert abs(sum(rec[:old])/old-met['old_macro'])<1e-12
  else:assert met['old_macro'] is None
  count+=len(labels)
 for name,d in record['contrasts'].items():
  x,y=name.split('-');p,q=pred[x],pred[y];assert abs(ba[x]-ba[y]-d['delta_BA'])<1e-12
  assert d['rescue']==sum(a==c and b!=c for a,b,c in zip(p,q,labels)) and d['harm']==sum(a!=c and b==c for a,b,c in zip(p,q,labels)) and d['changed']==sum(a!=b for a,b in zip(p,q))
 def correct(arm):return {i for i,y in enumerate(labels) if pred[arm][i]==y}
 base=correct('cls768_ridge');patch=correct('patchmean768_ridge');allidx=set(range(len(labels)));groups=dict(both_correct=base&patch,CLS_only=base-patch,patch_only=patch-base,both_wrong=allidx-(base|patch))
 for entry in [x for x in overs if x['stage']==stage]:
  ca=correct('cls_same_block');cj=correct(entry['joint_arm'])
  for kind,a,j in [('rescue',ca-base,cj-base),('harm',base-ca,base-cj)]:
   partitions={'all':(a,j)}
   for key,idx in groups.items():partitions['group:'+key]=(a&idx,j&idx)
   for c in seen:
    idx={i for i,y in enumerate(labels) if y==c};partitions['class:'+str(c)]=(a&idx,j&idx)
   for key,(aa,jj) in partitions.items():
    target=entry[kind]['all'] if key=='all' else entry[kind]['original_groups'][key[6:]] if key.startswith('group:') else entry[kind]['per_class'][key[6:]]
    values=dict(A_count=len(aa),J_count=len(jj),shared=len(aa&jj),A_only=len(aa-jj),J_only=len(jj-aa),union=len(aa|jj),Jaccard=len(aa&jj)/len(aa|jj) if aa|jj else None)
    assert values==target
for data in frequency.values():
 for arm,m in data['BA'].items():
  for name,members in data['bins'].items():assert abs(sum(r[-1]['metrics'][arm]['recall'][cfg['order'].index(c)] for c in members)/len(members)-m[name])<1e-12
cost=json.loads((root/'costs.json').read_text());term=json.loads((root.parent/'terminal.private.json').read_text());assert term['returncode']==0 and term['elapsed_seconds']<cfg['max_wall_seconds'];assert cost['new_training_images']==cost['new_validation_images']==cost['optimizer_updates']==cost['head_fits']==0 and cost['attempts']==1
size=sum(p.stat().st_size for p in root.iterdir() if p.is_file());assert size<cfg['output_budget_bytes']
result=dict(status='PASS',predictions_recounted=count,metrics_contrasts_frequency_and_all_overlap_sets_verified=True,training_RMS_independently_verified=True,max_training_scale_error=max_scale_error,max_fusion_score_error=max_score_error,previous_nine_scores_exactly_reused=True,budget_checks_passed=True,output_bytes=size,guardian_wall_seconds=term['elapsed_seconds'],wall_seconds=time.time()-start)
(root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
