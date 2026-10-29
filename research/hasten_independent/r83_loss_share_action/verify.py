"""Saved-only fixed-batch/share/state reconstruction, no scoring/optimization."""
import os,json,torch
from pathlib import Path
from common import write
from data import train_rows
from folds import assign_folds
from decision import allocation
from loss_share import weights
from train import parameters_diff
from learner import cpu_copy
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]));cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);dc=cfg['domains']['HK'];records=json.loads((r/'HK/RECORDS.private.json').read_text());seal=json.loads((r/'HK/ALL_ENDPOINTS_SEALED.json').read_text());assert len(records)==36;rows,labels=train_rows(dc);fold=assign_folds(labels.tolist(),[z['identity_component'] for z in rows]);sources={};actual=branches=micro=0

def equal(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
for v in records:
 stage,b,rep,n=v['stage'],v['fold'],v['repeat'],v['seen'];ids=[i for classes in dc['stages'][:stage+1] for i,c in enumerate(labels.tolist()) if c in classes];y=labels[ids];S=[i for i,idx in enumerate(ids) if fold[idx]==1-b];Q=[i for i,idx in enumerate(ids) if fold[idx]==b];seed=74002+2000000+10000*stage+100*b+10*rep;assert v['seed']==seed;fixed=[]
 for h in range(3):
  common=[S[i] for i in torch.randperm(len(S),generator=torch.Generator().manual_seed(seed+h)).tolist()[:192 if h<2 else 256]];ii=allocation(y,S,common,-1,torch.Generator().manual_seed(seed+800000+h)) if h<2 else common;assert len(ii)==len(set(ii))==256 and not set(ii)&set(Q);fixed.append(ii)
 assert set(y[fixed[0]].tolist())==set(range(n));assert v['legal_actions']==list(range(n+1)) and [z['action'] for z in v['branches']]==list(range(n+1))+[0,-1]
 folder=r/'HK/endpoints'/f's{stage}_f{b}_r{rep}';st=torch.load(folder/'START.private.pt',map_location='cpu',weights_only=False)
 if stage not in sources:sources[stage]=torch.load(Path(cfg['source_run'])/'HK/uniform_controller/74002'/f'head{stage}.private.pt',map_location='cpu',weights_only=False)
 equal(st['delta'],sources[stage]['delta']);equal(st['optimizer_state'],sources[stage]['optimizer_state']);assert st['optimizer_state']['param_groups'][0]['lr']==.0005;baseline=None;expo=sum((torch.bincount(y[ii],minlength=n) for ii in fixed),torch.zeros(n,dtype=torch.long))
 for j,z in enumerate(v['branches']):
  assert z['exposure']==expo.tolist() and z['exposure_minus_action0']==[0]*n and z['duplicate']==(j==n+1) and z['fixed_mean']==(j==n+2)
  for h,ii in enumerate(fixed):
   assert z['allocation'][h]['counts']==torch.bincount(y[ii],minlength=n).tolist();a=z['action'] if h==0 else 0;w=weights(y[ii],n,a) if h==0 else torch.full((256,),1/256,dtype=torch.float64);mass=[float(w[y[ii]==c].sum()) for c in range(n)];assert max(abs(x-z0) for x,z0 in zip(mass,z['allocation'][h]['CE_class_mass']))<1e-12
  end=torch.load(r/'HK/endpoints'/z['endpoint'],map_location='cpu',weights_only=False);assert end['action']==z['action'] and end['duplicate']==z['duplicate'] and end['fixed_mean']==z['fixed_mean'];assert all(int(q['step'])==24*(stage+1)+3 for q in end['optimizer_state']['state'].values());assert all(torch.isfinite(q).all() for q in end['delta'].values())
  if baseline is None:baseline=cpu_copy(end['delta'])
  assert z['parameter_difference']==parameters_diff(end['delta'],baseline);assert z['actual_updates']==3 and z['micro_backwards']==12;actual+=3;micro+=12;branches+=1
assert (actual,micro,branches)==(2268,9072,756);assert seal['actual_updates']==actual and seal['micro_backwards']==micro and seal['branches']==branches and seal['fixed_mean_controls']==36 and seal['controller_updates']==seal['VAL_reads']==seal['TEST_reads']==0
write(r/'results/VERIFICATION.json',dict(passed=True,blocks=36,branches=756,actual_visual_updates=actual,micro_backwards=micro,fixed_mean_controls=36,controller_updates=0,VAL_reads=0,TEST_reads=0,fixed_unique_batch_reconstructed=True,class_loss_share_masses_verified=True,all_start_states_and_AdamW_exact=True,new_model_scores=0,new_optimizer_steps=0));print('Saved-only R83 verification passed')
