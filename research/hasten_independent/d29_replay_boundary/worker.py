"""Fixed TRAIN Gaussian draws and deterministic segment-to-NCM-cell feasibility."""
import os,json,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F

def native(x,means):return (F.normalize(x.float(),dim=1)@F.normalize(means.float(),dim=1).T).argmax(1)
def correct_replay(x,label,means):
 assert x.dtype==torch.float32 and means.dtype==torch.float32
 a=F.normalize(means.double(),dim=1);anchor=means[label].double();before=native(x,means);off=before!=label
 assert int(native(means[label:label+1],means)[0])==label
 out=x.clone();alpha=torch.zeros(len(x),dtype=torch.float32);fallback=torch.zeros(len(x),dtype=torch.bool)
 if off.any():
  g=x[off].double();scores=g@a.T;gaps=scores[:,label:label+1]-scores
  ag=anchor@a.T;ag=ag[label]-ag
  bad=gaps<0;den=ag[None,:]-gaps
  assert (den[bad]>0).all()
  t=torch.where(bad,-gaps/den.clamp_min(torch.finfo(torch.float64).tiny),torch.zeros_like(gaps)).max(1).values
  assert ((t>=0)&(t<=1)).all()
  t=torch.nextafter(t.float(),torch.ones_like(t,dtype=torch.float32)).clamp_max(1)
  fixed=((1-t.double()[:,None])*g+t.double()[:,None]*anchor).float()
  missed=native(fixed,means)!=label;fixed[missed]=means[label]
  out[off]=fixed;alpha[off]=t;fallback[off]=missed
 missed=native(out,means)!=label
 assert not bool((missed&~off).any())
 out[missed]=means[label];fallback[missed]=True
 assert torch.equal(out[~off],x[~off]) and bool((native(out,means)==label).all())
 return out,off,alpha,fallback

def run():
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ref=Path(cfg['reference']);torch.set_num_threads(4);torch.set_num_interop_threads(2)
 def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
 mu=torch.tensor([[1.,0.],[0.,1.]]);x=torch.tensor([[1.,0.],[0.,1.]])
 z,off,t,fb=correct_replay(x,0,mu);assert off.tolist()==[False,True] and torch.equal(z[0],x[0]) and (native(z,mu)==0).all()
 write('SYNTHETIC_ACCEPTANCE.json',dict(unchanged_consistent_rows=True,offclass_corrected=True,no_rng_in_projection=True))
 results=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else {}
 total=sum(r['rows'] for r in results.values());offtotal=sum(r['offclass'] for r in results.values())
 for di,ds in enumerate(['NIH','ISIC','HK']):
  if ds in results:continue
  source=ref/'linear64'/ds;seal=json.loads((source/'SPLIT_SEAL.json').read_text());records=[]
  for stage in range(1,len(seal['stages'])):
   previous=torch.load(source/f'trajectories/74002/H-unit/stage{stage-1}.private.pt',map_location='cpu',weights_only=False)
   current=torch.load(source/f'trajectories/74002/H-unit/stage{stage}.private.pt',map_location='cpu',weights_only=False)
   means=current['network']['fc.weight'];old=previous['total'];n=current['total'];assert old==sum(map(len,seal['stages'][:stage])) and n==sum(map(len,seal['stages'][:stage+1]))
   assert torch.equal(native(means,means),torch.arange(n))
   gen=torch.Generator().manual_seed(20261024+100*di+stage)
   for c in range(old):
    cov=previous['cov'][c].double();m=previous['prototype'][c].double();assert torch.isfinite(cov).all()
    x=(torch.randn(128,512,generator=gen,dtype=torch.double)@torch.linalg.cholesky(cov).T+m).float()
    z,off,t,fb=correct_replay(x,c,means);nn=int(off.sum());rel=(z.double()-x.double()).norm(dim=1)/x.double().norm(dim=1).clamp_min(1e-30)
    records.append(dict(stage=stage,class_index=c,seen=n,rows=128,offclass=nn,moved=int((z!=x).any(1).sum()),fallback=int(fb.sum()),alpha_mean=None if not nn else float(t[off].double().mean()),alpha_max=None if not nn else float(t[off].max()),relative_displacement_mean=float(rel.mean()),relative_displacement_max=float(rel.max()),corrected_class_consistent_rows=128,unchanged_consistent_rows=int((~off).sum())))
  rows=sum(r['rows'] for r in records);bad=sum(r['offclass'] for r in records);fallback=sum(r['fallback'] for r in records)
  results[ds]=dict(records=records,rows=rows,offclass=bad,offclass_fraction=bad/rows,fallback=fallback,all_denominators_preserved=True);total+=rows;offtotal+=bad;write('RESULTS.json',results)
 passed=all(v['offclass']>0 for v in results.values()) and offtotal/total>=.01
 write('FEASIBILITY.json',dict(passed=passed,rows=total,offclass=offtotal,offclass_fraction=offtotal/total,requires_nonzero_all_domains=True,minimum_fraction=.01,no_performance_claim=True,optimizer_steps=0,VAL_reads=0,no_test=True))
 write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
if __name__=='__main__':
 try:run()
 except BaseException:
  out=Path(json.loads(Path(os.environ['RUN_CONFIG']).read_text())['output']);(out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
