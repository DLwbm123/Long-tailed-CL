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
 out=x.clone();alpha=torch.zeros(len(x),dtype=torch.float32,device=x.device);fallback=torch.zeros(len(x),dtype=torch.bool,device=x.device)
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
 alpha[fallback]=1
 return out,off,alpha,fallback
