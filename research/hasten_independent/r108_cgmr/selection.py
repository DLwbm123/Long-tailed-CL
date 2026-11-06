"""Fixed R107 ZERO/BALANCED construction; no learned value or actor features."""
import torch
import torch.nn.functional as F

def top_capacity(hist,k):
 left=torch.broadcast_to(torch.as_tensor(k,device=hist.device),hist.shape[:-1]).clone();out=torch.zeros_like(left)
 for q in range(5,-1,-1):
  take=torch.minimum(left,hist[...,q]);out+=q*take;left-=take
 return out

class FixedSelect:
 def __init__(self,pool,C,capacity):
  self.p=pool;self.C=C;self.B=min(capacity,len(pool));g=pool.gidx;self.cent=F.normalize(torch.stack([pool.z[g[pool.y[g]==c]].mean(0) for c in range(C)]),dim=-1)
  self.cover=torch.stack([ix[(pool.z[ix]@self.cent[c]).argmax()] for c in range(C) for ix in [g[pool.y[g]==c]]]);self.optional=g[~torch.isin(g,self.cover)];self.caps=torch.bincount(pool.parent_index[pool.view>0],minlength=len(g));self.covercap=int(self.caps[pool.parent_index[self.cover]].sum());hist=torch.bincount(self.caps[pool.parent_index[self.optional]],minlength=6);ng=torch.arange(C,min(self.B,len(g))+1,device=g.device);self.ng=ng[top_capacity(hist.expand(len(ng),-1),ng-C)+self.covercap>=self.B-ng];assert len(self.ng)
 def choose(self,mode,generator):
  p=self.p;sel=self.cover.clone();forced=0;decisions=0
  def pick(ix,score=None):
   nonlocal forced,decisions
   if len(ix)==1:forced+=1;return ix[0]
   decisions+=1
   return ix[int(score.argmax())] if score is not None else ix[int(torch.randint(len(ix),(1,),device=ix.device,generator=generator))]
  ng=int(self.ng[(self.ng-max(self.C,self.B//2)).abs().argmin()]) if mode=='BALANCED' else int(pick(self.ng));cap=self.covercap
  while int((p.view[sel]==0).sum())<ng:
   ix=self.optional[~torch.isin(self.optional,sel)];left=ng-int((p.view[sel]==0).sum())-1;caps=self.caps[p.parent_index[ix]];h=torch.bincount(caps,minlength=6);valid=(len(ix)-1>=left)&(cap+caps+top_capacity(h.expand(len(ix),-1)-F.one_hot(caps,6),left)>=self.B-ng);ix=ix[valid];assert len(ix)
   score=None
   if mode=='BALANCED':score=-100*torch.bincount(p.y[sel[p.view[sel]==0]],minlength=self.C).float()[p.y[ix]]+(p.z[ix]*self.cent[p.y[ix]]).sum(-1)
   j=pick(ix,score);sel=torch.cat([sel,j.reshape(1)]);cap+=int(self.caps[p.parent_index[j]])
  while len(sel)<self.B:
   ix=torch.where((p.view>0)&torch.isin(p.parent_index,p.parent_index[sel[p.view[sel]==0]])&~torch.isin(torch.arange(len(p),device=p.z.device),sel))[0];assert len(ix)>=self.B-len(sel);score=None
   if mode=='BALANCED':score=-100*torch.bincount(p.y[sel],minlength=self.C).float()[p.y[ix]]-torch.bincount(p.parent_index[sel[p.view[sel]>0]],minlength=len(p.parents)).float()[p.parent_index[ix]]-.0001*p.view[ix]
   j=pick(ix,score);sel=torch.cat([sel,j.reshape(1)])
  mem=p.take(sel);mem.assert_subset(p);assert len(mem)==self.B
  return mem,self.cover,dict(global_tokens=ng,local_tokens=self.B-ng,nontrivial_decisions=decisions,forced_transitions=forced)
