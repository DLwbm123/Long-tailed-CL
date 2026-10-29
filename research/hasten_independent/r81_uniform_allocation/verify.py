"""Saved-only uniform-control audit. TRAIN metadata is read, never images or models."""
import os,json,copy
from pathlib import Path
import torch
from common import write
from data import train_rows
from folds import assign_folds
from decision import legal_actions,allocation
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);d=r/'HK';dc=cfg['domains']['HK'];ex=json.loads((r/'CONTROL_EXIT.json').read_text());assert ex['exit_code']==0 and ex['peak_aggregate_RSS_bytes']<30000000000
assert json.loads((r/'SOFTWARE_CHECK.json').read_text())['passed'] and json.loads((r/'NATIVE_CHECK.json').read_text())['passed']
records=json.loads((d/'TRAIN_RECORDS.json').read_text());ctx=json.loads((d/'CONTEXT_RECORDS.json').read_text());seal=json.loads((d/'ALL_HEADS_SEALED.json').read_text());agg=json.loads((d/'aggregate.json').read_text());assert len(records)==6 and len(ctx)==48 and len(agg['results'])==6
assert seal['records']==6 and seal['VAL_reads']==0 and (seal['actual_optimizer_steps'],seal['committed_optimizer_steps'],seal['micro_backwards'])==(1296,144,5184)
def equal(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
original=torch.load(Path(cfg['matched_reference_run'])/'HK/74002/head0.private.pt',map_location='cpu',weights_only=False)
rows,labels=train_rows(dc);ids=[];first_actor=None
for stage in range(6):
 ids.extend(i for i,y in enumerate(labels.tolist()) if y in dc['stages'][stage]);yy=labels[ids];n=sum(map(len,dc['stages'][:stage+1]));folds=assign_folds(yy.tolist(),[rows[i]['identity_component'] for i in ids]);parts=[[i for i,f in enumerate(folds) if f==b] for b in [0,1]];folder=d/'uniform_controller/74002';h=torch.load(folder/f'head{stage}.private.pt',map_location='cpu',weights_only=False);before=torch.load(folder/f'before{stage}.private.pt',map_location='cpu',weights_only=False);rec=records[stage]
 assert rec['arm']=='uniform_controller' and rec['stage']==stage and rec['actual_optimizer_steps']==216 and rec['committed_optimizer_steps']==24 and rec['micro_backwards']==864 and rec['controller_updates']==0 and rec['LoRA_parameters']==49152 and rec['original_parameters_frozen'] and rec['strict_restore_exact'] and rec['VAL_reads']==0
 if stage:
  prev=torch.load(folder/f'head{stage-1}.private.pt',map_location='cpu',weights_only=False)
  for k,j in [('delta','delta'),('optimizer','optimizer_state'),('controller','controller_state'),('controller_optimizer','controller_optimizer')]:equal(before[k],prev[j])
 else:equal(before['delta'],original['delta']);assert before['optimizer']['state']=={};first_actor=before['controller']
 equal(first_actor,h['controller_state']);assert h['controller_optimizer']['state']=={}
 for k,v in h['controller_state'].items():
  if k.startswith(('cls.','bal.')):assert torch.count_nonzero(v)==0
 assert all(int(v['step'])==24*(stage+1) for v in h['optimizer_state']['state'].values())
 for context in range(8):
  c=next(v for v in ctx if v['stage']==stage and v['context']==context);support=parts[1-context%2];query=parts[context%2];assert c['support_counts']==torch.bincount(yy[support],minlength=n).tolist() and c['query_counts']==torch.bincount(yy[query],minlength=n).tolist();assert c['controller_updates']==0 and c['policy_stats']==[] and not c['reward_used_for_commit_selection'] and c['actual_updates']==27 and c['committed_updates']==3 and c['retained_branch']==(stage*8+context)%8
  assert c['unique_batch'] and c['no_query_in_inner_or_support'] and sum(c['all_exposure'])==6912 and sum(c['committed_exposure'])==768
  assert torch.equal(torch.tensor(c['all_exposure']),torch.tensor([v['exposure_counts'] for v in c['branches']]).sum(0))
  seed=74002+10000*stage+100*context
  for branch,br in enumerate(c['branches']):
   expo=torch.zeros(n,dtype=torch.long)
   for step in range(3):
    g=torch.Generator().manual_seed(seed+step);common=[support[i] for i in torch.randperm(len(support),generator=g).tolist()[:(192 if step<2 else 256)]]
    if step<2:
     mask=legal_actions(yy,support,common);p=mask.float()/mask.sum();cdf=p.cumsum(0);cdf/=cdf[-1].clone();u=torch.rand((),generator=torch.Generator().manual_seed(seed+700000+branch*10+step));action=int(torch.searchsorted(cdf,u)) if branch<8 else 0;assert br['actions'][step]==action;ii=allocation(yy,support,common,action-1,torch.Generator().manual_seed(seed+800000+step));assert torch.bincount(yy[ii[192:]],minlength=n).tolist()==br['allocation'][step]['controlled_counts']
    else:ii=common
    assert len(ii)==len(set(ii))==256 and not set(ii)&set(query);expo+=torch.bincount(yy[ii],minlength=n)
   assert expo.tolist()==br['exposure_counts']
  assert c['committed_exposure']==c['branches'][c['retained_branch']]['exposure_counts']
 N=len(ids);f=h['features'];y=h['labels'];assert f.shape==(N,512) and torch.isfinite(f).all() and torch.allclose(f.norm(dim=1),torch.ones(N),atol=2e-6);counts=torch.bincount(y,minlength=n);assert counts.tolist()==dc['train_counts_ordered'][:n]
 sums=torch.zeros(n,512,dtype=torch.double);sums.index_add_(0,y,f.double());proto=torch.nn.functional.normalize((sums/counts[:,None]).float(),dim=1);assert torch.allclose(proto,h['prototypes'],atol=2e-5,rtol=0)
 z=torch.load(folder/f'response{stage}.private.pt',map_location='cpu',weights_only=False);y=z['labels'];m=agg['results'][stage]['metrics']['uniform_controller'];assert z['scores'].shape==(len(y),n) and torch.isfinite(z['scores']).all() and len(z['ids'])==len(y)==len(set(z['ids']));assert torch.bincount(y,minlength=n).tolist()==m['counts'] and torch.bincount(y[z['scores'].argmax(1)==y],minlength=n).tolist()==m['correct']
clean=json.loads((d/'REGENERABLE_CACHE_CLEANUP.json').read_text());assert clean['reference_max_abs']<=2e-5 and clean['temporary_swap_bytes']==0 and clean['retained_prefix_bytes']<=23000000000 and clean['scientific_artifacts_preserved']
x=dict(passed=True,actual_visual_updates=1296,retained_visual_updates=144,micro_backwards=5184,controller_updates=0,contexts=48,branch_endpoints=432,heads=6,VAL_readouts=6,TEST_reads=0,uniform_actions_reconstructed=True,actor_unchanged_and_optimizer_empty=True,visual_AdamW_carry_verified=True,new_model_scores=0,new_optimizer_steps=0);write(r/'results/VERIFICATION.json',x);print(json.dumps(x))
