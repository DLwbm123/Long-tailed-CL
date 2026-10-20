import json,os,sys
from pathlib import Path

def main(cfg):
 root=Path(cfg['root']).resolve();ref=Path(cfg['reference']).resolve();matched=Path(cfg['matched']).resolve();window=Path(cfg['window_root']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());raw=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']];test=Path(bc['output']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  assert not any(p.is_relative_to(v) for v in raw),'RAW_FORBIDDEN'
  assert not p.is_relative_to(test),'TEST_FORBIDDEN'
  if p.is_relative_to(ref/'inputs'):assert p==ref/'inputs/0.private.pt','FUTURE_FORBIDDEN'
  if p.is_relative_to(matched):assert p.is_relative_to(matched/'pixels_0'),'MATCHED_NONINITIAL_FORBIDDEN'
  elif p.is_relative_to(window):assert p.is_relative_to(root),'OLD_EXPERIMENT_ASSETS_FORBIDDEN'
 sys.addaudithook(guard)
 for bad in [ref/'inputs/1.private.pt',matched/'pixels_3/INDEX.json',window/'readout_consistent_ce/train/P.encoder.private.pt',window/'decision_anchor/train/protection.private.pt']:
  try:guard('open',(str(bad),'rb'))
  except AssertionError:pass
  else:raise AssertionError('ACCESS_GUARD_SELFTEST')
 if cfg['mode']=='INJECT_FAILURE':raise RuntimeError('planned injected failure')
 import torch
 from sampling import write
 if cfg['mode']=='SYNTHETIC':
  from geometry import split,norm
  from evaluation import evaluate
  y=torch.arange(128)%4;groups=['toy'+str(int(c))+'g'+str(i//8) for i,c in enumerate(y)]
  f=split(y,groups);s=f==0;g=torch.randn(128,7,dtype=torch.float64,generator=torch.Generator().manual_seed(74291));before=g[s].clone();poison=g.clone();poison[~s]=torch.nan;yy=y.clone();yy[~s]=-99
  assert torch.equal(before,poison[s]) and torch.equal(y[s],yy[s])
  m=norm(before,torch.ones(len(before),dtype=torch.float64));pm=norm(poison[s],torch.ones(len(before),dtype=torch.float64));assert all(torch.equal(a,b) for a,b in zip(m,pm))
  x=torch.cat([before,torch.ones(len(before),1,dtype=torch.float64)],1);hot=torch.nn.functional.one_hot(y[s],4).double();a=x.T@x/len(x);a.diagonal()[:-1].add_(.01);rhs=x.T@hot/len(x);w=torch.linalg.solve(a,rhs);err=float((a@w-rhs).abs().max());assert err<=1e-10
  good=torch.nn.functional.one_hot(y,4).double()*3;bad=torch.roll(good,1,1);tab=dict(T_seen=good,T_out=bad,F_seen=bad,F_out=bad)
  r=evaluate(tab,y,groups);assert r['point_gate_passed'] and r['bootstrap_performed'] and len(r['contrasts'])==2
  tab['T_seen']=bad;r0=evaluate(tab,y,groups);assert not r0['point_gate_passed'] and not r0['bootstrap_performed']
  try:evaluate({},y,groups)
  except AssertionError:pass
  else:raise AssertionError('MISSING_FAMILY')
  write(Path(cfg['output'])/'aggregate.json',dict(pass_all=True,split_exclusion_coverage=True,poison_query_input_invariance=True,teacher_toy_residual=err,two_contrast_inference_success_failure_missing=True))
 elif cfg['mode']=='PREP':
  from geometry import split,teacher
  from data import initial
  obj=initial(cfg);fold=split(obj['y'],obj['groups']);torch.save(dict(fold=fold,y=obj['y'],groups=obj['groups'],row_ids=obj['row_ids'],cached_g=obj['g']),Path(cfg['output'])/'PLAN.private.pt');summary=[]
  for k in [0,1]:
   mask=fold==k;ids=torch.where(mask)[0];sg=[obj['groups'][i] for i in ids];t=teacher(obj['g'][mask],obj['y'][mask],sg);torch.save(t,Path(cfg['output'])/f'{k}.teacher.private.pt')
   summary.append(dict(fold=k,rows=len(ids),class_rows=[int((obj['y'][mask]==c).sum()) for c in range(4)],class_groups=[len({obj['groups'][i] for i in torch.where(mask & (obj['y']==c))[0]}) for c in range(4)],residual=t['residual']))
  write(Path(cfg['output'])/'PLAN_SEAL.json',dict(folds=summary,teacher_fits=2,seed=74290,query_excluded_from_teacher_and_mapper=True,development_used=False,source_commit=cfg['source_commit']))
 elif cfg['mode'] in ['TRAIN','PREFLIGHT']:
  from train import run
  run(cfg)
 else:
  assert cfg['mode']=='EVALUATE'
  from pipeline import run
  run(cfg)
