"""Sequential, bounded, real optimizer trajectories; all heads sealed before VAL."""
import os,json,time,copy,random,traceback,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import write,guard
from data import train_rows,get_cache,Tokens,OrderSampler,DataLoader,store
from folds import assign_folds
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
from decision import *

def cpu_copy(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_copy(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
 return copy.deepcopy(x)

def encode(model,data,indices,n,cfg):
 sampler=OrderSampler(len(data));sampler.order=list(indices);loader=DataLoader(data,batch_size=256,sampler=sampler);zs=[];ys=[]
 with torch.no_grad():
  for tx,bx,yy,ix in loader:
   guard(cfg)
   for tt,y in zip(tx.split(64),yy.split(64)):zs.append(model(tt.cuda()).cpu());ys.append(y)
 z=torch.cat(zs);y=torch.cat(ys);counts=torch.bincount(y,minlength=n);assert bool((counts>0).all())
 sums=torch.zeros(n,512,dtype=torch.double);sums.index_add_(0,y,z.double());proto=F.normalize((sums/counts[:,None]).float(),dim=1)
 return dict(prototypes=proto,features=z,labels=y,counts=counts)

def evaluate(model,data,indices,proto,n,cfg):
 sampler=OrderSampler(len(data));sampler.order=list(indices);dl=DataLoader(data,batch_size=256,sampler=sampler);losses=[];margins=[];ces=[];ys=[];correct=[]
 with torch.no_grad():
  for tx,bx,yy,ix in dl:
   guard(cfg)
   for tt,y in zip(tx.split(64),yy.split(64)):
    scores=20*model(tt.cuda())@proto.cuda().T;p=scores.softmax(1);yg=y.cuda();target=F.one_hot(yg,n).float();losses.append((.5*(p-target).square().sum(1)).cpu());ces.append(F.cross_entropy(scores,yg,reduction='none').cpu());v=p.clone();v[torch.arange(len(y),device='cuda'),yg]=-1;margins.append((p[torch.arange(len(y),device='cuda'),yg]-v.max(1).values).cpu());ys.append(y);correct.append((scores.argmax(1)==yg).cpu())
 y=torch.cat(ys);den=torch.bincount(y,minlength=n);assert bool((den>0).all())
 def mean(values):
  sums=torch.zeros(n);sums.index_add_(0,y,torch.cat(values));return sums/den
 cc=torch.bincount(y[torch.cat(correct)],minlength=n)
 return dict(brier=mean(losses),margin=mean(margins),ce=mean(ces),correct=cc,counts=den)

def inner_step(model,opt,data,indices,proto,nc,cfg):
 assert len(indices)==len(set(indices))==256;assert len(proto)==len(nc)
 tokens=store().batch(data,indices,True);yy=data.labels[indices];base=data.base[indices];opt.zero_grad(set_to_none=True);loss_sum=0.
 for tx,bx,by in zip(tokens.split(64),base.split(64),yy.split(64)):
  guard(cfg);z=model(tx.cuda());logits=20*z@proto.cuda().T+nc.cuda().float().log()[None,:];loss=(F.cross_entropy(logits,by.cuda(),reduction='none')+.01*(z-bx.cuda()).square().sum(1)).sum()/256
  assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach())
 params=[p for p in model.parameters() if p.requires_grad];assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step();assert torch.cuda.max_memory_reserved(0)<12*2**30
 return loss_sum

