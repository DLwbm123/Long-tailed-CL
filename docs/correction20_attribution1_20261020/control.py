"""Weighted class-preserving signal randomization; no new observations."""
import numpy as np
import torch
import torch.nn.functional as F


def transport(y,w,seed,outer):
 recipients=[];donors=[];mass=[]
 for c in range(20):
  ids=np.flatnonzero(y==c);order=np.random.default_rng(np.random.SeedSequence([seed,outer,c,1])).permutation(ids);i=j=0;a=float(w[ids[0]]);b=float(w[order[0]])
  while i<len(ids) and j<len(ids):
   v=min(a,b)
   if v>0:recipients.append(ids[i]);donors.append(order[j]);mass.append(v)
   a-=v;b-=v
   if a<=1e-18:
    i+=1
    if i<len(ids):a=float(w[ids[i]])
   if b<=1e-18:
    j+=1
    if j<len(ids):b=float(w[order[j]])
 e=np.array(recipients,dtype=np.int64);d=np.array(donors,dtype=np.int64);m=np.array(mass);assert np.all(y[e]==y[d])
 assert np.max(np.abs(np.bincount(e,weights=m,minlength=len(y))-w))<1e-12 and np.max(np.abs(np.bincount(d,weights=m,minlength=len(y))-w))<1e-12
 return e,d,m


def shuffled_loss(u,x,base,y,w,repair,target,keep,pre,e,d,m,regularizer):
 s=base+x@u;ce=F.cross_entropy(s,y,reduction='none');value=(w*ce).sum()+(m*repair[d]*ce[e]).sum()
 live=keep[d]>0;ee=e[live];dd=d[live];mm=m[live];gap=s[ee].gather(1,y[ee,None])-s[ee];pen=F.softplus(target[dd,None]-gap);pen=pen.scatter(1,y[ee,None],0.)
 return value+(mm*keep[dd]*pen.sum(1)/(s.shape[1]-1)).sum()+regularizer*((pre[:,None]*u)**2).sum()
