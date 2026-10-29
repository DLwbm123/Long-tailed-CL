"""Known class-loss shares: fixed unique batch; anchor remains separately uniform."""
import torch
import torch.nn.functional as F
from learner import inner_step
from data import store
from common import guard
RHO=.25

def weights(labels,n,action,rho=RHO):
 counts=torch.bincount(labels,minlength=n).double();assert bool((counts>0).all());B=len(labels);w=torch.ones(B,device=labels.device,dtype=torch.float64)/B
 if action>0:
  c=action-1;assert 0<=c<n;w=(1-rho)*w+rho*(labels==c).double()/counts[c]
 elif action==-1:w=(1-rho+rho/(n+1))*w+(rho/(n+1))/counts[labels]
 else:assert action==0
 assert torch.isfinite(w).all() and bool((w>=0).all()) and abs(float(w.sum())-1)<1e-12
 return w

def step(model,opt,data,indices,proto,nc,cfg,action):
 if action==0:return inner_step(model,opt,data,indices,proto,nc,cfg)
 assert len(indices)==len(set(indices))==256
 tokens=store().batch(data,indices,True);yy=data.labels[indices];base=data.base[indices];w=weights(yy,len(nc),action).float();opt.zero_grad(set_to_none=True);total=0.
 for tx,bx,by,ww in zip(tokens.split(64),base.split(64),yy.split(64),w.split(64)):
  guard(cfg);z=model(tx.cuda());logits=20*z@proto.cuda().T+nc.cuda().float().log()[None,:];loss=(F.cross_entropy(logits,by.cuda(),reduction='none')*ww.cuda()).sum()+.01*(z-bx.cuda()).square().sum()/256;assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
 params=[p for p in model.parameters() if p.requires_grad];assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step();assert torch.cuda.max_memory_reserved(0)<12*2**30;return total
