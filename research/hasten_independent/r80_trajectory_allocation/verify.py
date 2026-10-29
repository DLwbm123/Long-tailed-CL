"""Saved-only count, carry, reward and score audit; no images or model calls."""
import os,json,math
from pathlib import Path
import torch
from common import write
from decision import ARMS,LEARNED,utility
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);d=root/'HK';exit=json.loads((root/'CONTROL_EXIT.json').read_text());assert exit['exit_code']==0 and exit['peak_aggregate_RSS_bytes']<30000000000
for tag in ['SOFTWARE_CHECK','NATIVE_CHECK']:assert json.loads((root/(tag+'.json')).read_text())['passed']
records=json.loads((d/'TRAIN_RECORDS.json').read_text());contexts=json.loads((d/'CONTEXT_RECORDS.json').read_text());seal=json.loads((d/'ALL_HEADS_SEALED.json').read_text());agg=json.loads((d/'aggregate.json').read_text());assert len(records)==30 and len(contexts)==240 and len(agg['results'])==6
assert seal['records']==36 and seal['VAL_reads']==0 and seal['actual_optimizer_steps']==6480 and seal['committed_optimizer_steps']==1872 and seal['micro_backwards']==25920

def equal_tree(x,y):
 if torch.is_tensor(x):assert torch.equal(x,y)
 elif isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in x:equal_tree(x[k],y[k])
 elif isinstance(x,(tuple,list)):
  assert len(x)==len(y)
  for a,b in zip(x,y):equal_tree(a,b)
 else:assert x==y
for r in records:
 arm=r['arm'];stage=r['stage'];folder=d/arm/'74002';head=torch.load(folder/f'head{stage}.private.pt',map_location='cpu',weights_only=False);before=torch.load(folder/f'before{stage}.private.pt',map_location='cpu',weights_only=False)
 assert r['actual_optimizer_steps']==216 and r['committed_optimizer_steps']==(216 if arm=='balanced_long' else 24) and r['micro_backwards']==864 and r['LoRA_parameters']==49152 and r['original_parameters_frozen'] and r['strict_restore_exact'] and r['VAL_reads']==0
 if stage:
  prev=torch.load(folder/f'head{stage-1}.private.pt',map_location='cpu',weights_only=False)
  for k,j in [('delta','delta'),('optimizer','optimizer_state'),('controller','controller_state'),('controller_optimizer','controller_optimizer')]:equal_tree(before[k],prev[j])
 else:
  original=torch.load(Path(cfg['matched_reference_run'])/'HK/74002/head0.private.pt',map_location='cpu',weights_only=False);equal_tree(before['delta'],original['delta']);assert before['optimizer']['state']=={}
 assert r['controller_updates']==(32*(stage+1) if arm in LEARNED else 0)
for c in contexts:
 arm=c['arm'];n=len(c['support_counts']);assert c['context'] in range(8) and c['actual_updates']==27 and c['committed_updates']==(27 if arm=='balanced_long' else 3) and c['unique_batch'] and c['no_query_in_inner_or_support'] and not c['reward_used_for_commit_selection']
 brier=torch.tensor([v['class_brier'] for v in c['branches']]);anchor=torch.tensor(c['anchor_brier']);protected=torch.tensor(c['protected_classes']);us=torch.stack([utility(v,anchor,protected,arm!='macro_controller')[0] for v in brier]);ds=torch.stack([utility(v,anchor,protected,True)[1] for v in brier]);rr=us[:8]-us[8];aa=(rr-rr.mean())/(rr.std(unbiased=False)+1e-8)
 assert torch.allclose(us,torch.tensor(c['utilities']),atol=1e-6) and torch.allclose(ds,torch.tensor(c['damage']),atol=1e-6) and torch.allclose(rr,torch.tensor(c['absolute_returns']),atol=1e-6) and torch.allclose(aa,torch.tensor(c['advantages']),atol=2e-4)
 selected=8 if arm=='balanced_long' else (c['stage']*8+c['context'])%8;assert c['retained_branch']==selected
 assert sum(c['all_exposure'])==9*3*256 and sum(c['committed_exposure'])==c['committed_updates']*256
 assert torch.tensor(c['all_exposure']).equal(torch.tensor([v['exposure_counts'] for v in c['branches']]).sum(0))
 for branch in c['branches']:
  assert sum(branch['exposure_counts'])==768 and all(sum(v['controlled_counts'])==64 for v in branch['allocation'])
  assert all(v['priority_allocated']<=min(64,v['priority_available']) for v in branch['allocation'])
 if arm=='balanced_short':
  assert torch.allclose(brier,brier[:1].expand_as(brier),atol=1e-6,rtol=0)
  assert all(v['exposure_counts']==c['branches'][0]['exposure_counts'] and v['allocation']==c['branches'][0]['allocation'] and v['inner_loss']==c['branches'][0]['inner_loss'] and v['query_correct']==c['branches'][0]['query_correct'] for v in c['branches'])
 assert torch.allclose(torch.tensor(c['assigned_returns']),rr.roll(1) if arm=='shuffled_controller' else rr,atol=1e-6)
 if arm in LEARNED:assert len(c['policy_stats'])==4
for arm in (*ARMS,'frozen_initial'):
 for stage in range(6):
  folder=d/arm/'74002';h=torch.load(folder/f'head{stage}.private.pt',map_location='cpu',weights_only=False);z=torch.load(folder/f'response{stage}.private.pt',map_location='cpu',weights_only=False);n=h['seen'];N=sum(h['train_counts']).item();f=h['features'];y=h['labels'];nn=h['support_counts'];assert f.shape==(N,512) and torch.isfinite(f).all() and torch.allclose(f.norm(dim=1),torch.ones(N),atol=2e-6)
  sums=torch.zeros(n,512,dtype=torch.double);sums.index_add_(0,y,f.double());rebuilt=torch.nn.functional.normalize((sums/nn[:,None]).float(),dim=1);assert torch.allclose(rebuilt,h['prototypes'],atol=2e-5,rtol=0) and torch.bincount(y,minlength=n).equal(h['train_counts'])
  m=agg['results'][stage]['metrics'][arm];yy=z['labels'];assert len(z['ids'])==len(yy)==len(set(z['ids'])) and torch.isfinite(z['scores']).all() and z['scores'].shape==(len(yy),n)
  assert torch.bincount(yy,minlength=n).tolist()==m['counts'] and torch.bincount(yy[z['scores'].argmax(1)==yy],minlength=n).tolist()==m['correct']
  if arm=='frozen_initial':equal_tree(h['delta'],original['delta'])
cleanup=json.loads((d/'REGENERABLE_CACHE_CLEANUP.json').read_text());assert cleanup['scientific_artifacts_preserved'] and cleanup['reference_max_abs']<=2e-5 and cleanup['temporary_swap_bytes']==0 and cleanup['retained_prefix_bytes']<=23000000000
result=dict(passed=True,actual_visual_optimizer_steps=6480,retained_visual_optimizer_steps=1872,micro_backwards=25920,controller_optimizer_steps=576,trained_stage_heads=30,zero_update_stage_heads=6,new_VAL_readouts=36,new_TEST=0,new_model_scores_in_verifier=0,context_records=240,trajectory_endpoints=2160,carry_visual_optimizer_controller_verified=True)
write(root/'results/VERIFICATION.json',result);print(json.dumps(result))
