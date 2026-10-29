"""Saved-only reconstruction; no image/model scoring or optimizer calls."""
import os,json,torch
from pathlib import Path
from common import write
from train import parameters_diff
from learner import cpu_copy
from decision import allocation,legal_actions
from data import train_rows
from folds import assign_folds
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]));cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);dc=cfg['domains']['HK'];seal=json.loads((r/'HK/ALL_ENDPOINTS_SEALED.json').read_text());records=json.loads((r/'HK/RECORDS.private.json').read_text());rows,yy=train_rows(dc);folds=assign_folds(yy.tolist(),[v['identity_component'] for v in rows]);assert len(records)==seal['blocks']==36;actual=micro=branches=0
source_cache={}
def equal(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
assert json.loads((r/'SOFTWARE_CHECK.json').read_text())['passed'] and json.loads((r/'NATIVE_CHECK.json').read_text())['passed']
for v in records:
 stage,b,rep,n=v['stage'],v['fold'],v['repeat'],v['seen'];ids=[i for classes in dc['stages'][:stage+1] for i,label in enumerate(yy.tolist()) if label in classes];y=yy[ids];S=[i for i,idx in enumerate(ids) if folds[idx]==1-b];Q=[i for i,idx in enumerate(ids) if folds[idx]==b];seed=74002+2000000+10000*stage+100*b+10*rep;assert v['seed']==seed and v['source_optimizer_step']==24*(stage+1) and v['saved_lr']==.0005;common=[]
 for h in range(3):common.append([S[i] for i in torch.randperm(len(S),generator=torch.Generator().manual_seed(seed+h)).tolist()[:192 if h<2 else 256]])
 legal=torch.where(legal_actions(y,S,common[0]))[0].tolist();assert legal==v['legal_actions'];assert [z['action'] for z in v['branches']]==legal+[0]
 folder=r/'HK/endpoints'/f's{stage}_f{b}_r{rep}';start=torch.load(folder/'START.private.pt',map_location='cpu',weights_only=False)
 if stage not in source_cache:source_cache[stage]=torch.load(Path(cfg['source_run'])/'HK/uniform_controller/74002'/f'head{stage}.private.pt',map_location='cpu',weights_only=False)
 equal(start['delta'],source_cache[stage]['delta']);equal(start['optimizer_state'],source_cache[stage]['optimizer_state']);assert all(int(z['step'])==24*(stage+1) for z in start['optimizer_state']['state'].values());baseline=None
 for j,z in enumerate(v['branches']):
  expo=torch.zeros(n,dtype=torch.long)
  for h in range(3):
   a=z['action'] if h==0 else 0;ii=allocation(y,S,common[h],a-1,torch.Generator().manual_seed(seed+800000+h)) if h<2 else common[h];assert len(ii)==len(set(ii))==256 and not set(ii)&set(Q);cc=torch.bincount(y[ii],minlength=n);assert cc.tolist()==z['allocation'][h]['counts'];expo+=cc
  assert expo.tolist()==z['exposure'];h=torch.load(r/'HK/endpoints'/z['endpoint'],map_location='cpu',weights_only=False);assert h['action']==z['action'] and h['duplicate']==(j==len(v['branches'])-1)==z['duplicate'];assert all(int(x['step'])==24*(stage+1)+3 for x in h['optimizer_state']['state'].values());assert h['features'].shape[0]==len(S) and torch.isfinite(h['features']).all();assert len(h['prototypes'])==n
  if baseline is None:baseline=cpu_copy(h['delta'])
  diff=parameters_diff(h['delta'],baseline)
  assert diff==z['parameter_difference'];assert all(torch.isfinite(t).all() for t in h['delta'].values());assert z['actual_updates']==3 and z['micro_backwards']==12
  for tag,m in z['query'].items():
   assert len(m['counts'])==len(m['brier'])==len(m['correct'])==n and all(0<=k<=den for k,den in zip(m['correct'],m['counts']));covered=[i for i,c in enumerate(m['counts']) if c];ba=100*sum(m['correct'][i]/m['counts'][i] for i in covered)/len(covered);assert abs(ba-z['stats'][tag]['BA'])<1e-10
   if tag=='full':assert sum(m['counts'])==len(Q)
  actual+=3;micro+=12;branches+=1
assert actual==seal['actual_updates']<=2160 and micro==seal['micro_backwards']==4*actual and branches==seal['branches']<=720 and seal['controller_updates']==seal['VAL_reads']==seal['TEST_reads']==0
write(r/'results/VERIFICATION.json',dict(passed=True,blocks=36,branches=branches,actual_visual_updates=actual,micro_backwards=micro,controller_updates=0,VAL_reads=0,TEST_reads=0,unique_batch_and_actions_reconstructed=True,all_saved_optimizer_steps=True,new_model_scores=0,new_optimizer_steps=0));print('saved-only verification passed')
