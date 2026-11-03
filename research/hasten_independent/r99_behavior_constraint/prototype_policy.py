"""Shared bounded prototype actions. Support geometry is frozen per episode."""
import math,torch
from torch import nn
import torch.nn.functional as F
SIGMA=.25

def unit(x):return F.normalize(x,dim=-1,eps=1e-8)
class Policy(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(1032,64),nn.Tanh(),nn.Linear(64,4));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,s):return 2*torch.tanh(self.net(s)/2)

@torch.no_grad()
def build_state(support,rawcounts,births,stage):
 C=len(support);p0=torch.stack([unit(z.mean(0)) for z in support]);sim=p0@p0.T
 # Vector lexicographic ties avoid introducing class-ID preference.
 keys=[tuple(v) for v in p0.cpu().tolist()];neighbors=[];h=[];states=[];active=[]
 for c,z in enumerate(support):
  ix=sorted([d for d in range(C) if d!=c],key=lambda d:(-float(sim[c,d]),keys[d]))[:3];neighbors.append(ix);cols=[z@p0[c]]+[z@p0[d] for d in ix];cols += [torch.zeros(len(z),device=z.device)]*(4-len(cols));v=torch.stack(cols,1);v=((v-v.mean(0))/v.std(0,correction=0).clamp_min(.05)).clamp(-3,3);m=len(z);h.append(v if m>1 else torch.zeros_like(v));var=z.var(0,correction=1) if m>1 else torch.zeros_like(z[0]);ns=sim[c,ix];ns=F.pad(ns,(0,3-len(ix)));sc=z.new_tensor([math.log1p(rawcounts[c])/math.log(6822),math.log1p(m)/math.log(6822),m>1,(stage-births[c])/5,births[c]==stage]);states.append(torch.cat([p0[c],var,ns,sc]));active.append(m>1)
 state=torch.stack(states);assert state.shape==(C,1032) and torch.isfinite(state).all();return {'support':support,'h':h,'state':state,'p0':p0,'active':torch.tensor(active,device=p0.device),'neighbors':neighbors}

def build_prototypes(actions,ep,details=False):
 assert actions.ndim==3 and actions.shape[1:]==(len(ep['support']),4)
 ps=[];ws=[]
 for c,(z,h) in enumerate(zip(ep['support'],ep['h'])):
  u=2*torch.tanh((actions[:,c]@h.T)/2);w=.75/len(z)+.25*u.softmax(-1);ps.append(unit(w@z));ws.append(w)
 p=torch.stack(ps,1);return (p,ws) if details else p

def gaussian_logp(mu,actions):return (-.5*((actions-mu[None])/SIGMA).square()-math.log(SIGMA)-.5*math.log(2*math.pi)).sum(-1)
def gaussian_KL(mu,ref,active):return ((mu-ref).square().sum(-1)/(2*SIGMA**2))[active].mean() if active.any() else mu.sum()*0
