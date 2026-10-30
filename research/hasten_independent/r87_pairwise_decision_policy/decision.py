"""Frozen two-slot top2 arbitration; labels are absent from policy inputs."""
import torch
import torch.nn.functional as F
from torch import nn

def inputs(scores,proto,support_counts,ages):
 scores=scores.double();n=scores.shape[1];order=torch.argsort(scores,dim=1,descending=True,stable=True);winner,runner=order[:,0],order[:,1];prob=scores.softmax(1)
 actions=[(-1,-1)];features=[[0.]*10+[1.]];groups=[torch.zeros(len(scores),dtype=torch.bool)]
 for target in range(n):
  for source in range(n):
   if target==source:continue
   rows=(winner==source)&(runner==target)&(scores[:,source]>scores[:,target])
   if not rows.any():continue
   gap=scores[rows,source]-scores[rows,target]
   actions.append((target,source));groups.append(rows)
   features.append([float(torch.log1p(support_counts[target]))/torch.log(torch.tensor(1000.)).item(),float(torch.log1p(support_counts[source]))/torch.log(torch.tensor(1000.)).item(),float(ages[target])/5,float(ages[source])/5,float(proto[target]@proto[source]),float(rows.double().mean()),float(gap.mean())/20,float(gap.std(unbiased=False))/20,float(prob[rows,target].mean()),float(prob[rows,source].mean()),0.])
 f=torch.tensor(features,dtype=torch.double);m=len(actions);g=f.mean(0);x1=torch.cat([f,g.expand(m,-1),torch.zeros(m,5,dtype=torch.double)],1)
 x2=torch.cat([f[None].expand(m,-1,-1),g.expand(m,m,-1),torch.ones(m,m,1,dtype=torch.double),torch.zeros(m,m,4,dtype=torch.double)],2)
 mask=torch.ones(m,m,dtype=torch.bool);mask[0]=False;mask[0,0]=True
 for a,(c,d) in enumerate(actions[1:],1):
  for b,(e,h) in enumerate(actions[1:],1):
   if {c,d}=={e,h}:mask[a,b]=False
   x2[a,b,-4:]=torch.tensor([e==c,e==d,h==c,h==d],dtype=torch.double)
 assert x1.shape[1]==x2.shape[2]==27 and torch.isfinite(x1).all() and torch.isfinite(x2).all()
 return dict(actions=actions,groups=groups,x1=x1,x2=x2,mask=mask,winner=winner,runner=runner,scores=scores)

def outcomes(ctx,labels,eligible,protected):
 scores=ctx['scores'];y=labels.long();n=scores.shape[1];m=len(ctx['actions']);count=torch.bincount(y,minlength=n).double();den=count.clamp_min(1)
 base_correct=torch.bincount(y[ctx['winner']==y],minlength=n).double();ce=F.cross_entropy(scores,y,reduction='none');base_ce=torch.zeros(n,dtype=torch.double).index_add_(0,y,ce)
 dc=torch.zeros(m,n,dtype=torch.double);dl=torch.zeros_like(dc)
 for a,((c,d),rows) in enumerate(zip(ctx['actions'][1:],ctx['groups'][1:]),1):
  good=rows&(y==c);bad=rows&(y==d);dc[a,c]=good.sum();dc[a,d]=-bad.sum();gap=scores[:,d]-scores[:,c];dl[a,c]=-gap[good].sum();dl[a,d]=gap[bad].sum()
 acc=(base_correct[None,None]+dc[:,None]+dc[None,:])/den
 nll=(base_ce[None,None]+dl[:,None]+dl[None,:])/den
 available=count>0;eligible=torch.tensor([c for c in eligible if available[c]],dtype=torch.long);prot=torch.tensor([c for c in protected if c in eligible.tolist()],dtype=torch.long)
 assert len(eligible)>0
 reward=acc[:,:,eligible].mean(-1);cost=nll[:,:,eligible].mean(-1)
 if len(prot):
  reward-=torch.relu(base_correct[prot]/den[prot]-acc[:,:,prot]).mean(-1)
  cost+=torch.relu(nll[:,:,prot]-base_ce[prot]/den[prot]).mean(-1)
 def standard(v):
  z=v[ctx['mask']];return (v-z.mean())/z.std(unbiased=False).clamp_min(1e-8)
 return dict(count=count,base_correct=base_correct,base_ce=base_ce,delta_correct=dc,delta_ce=dl,reward=reward,cost=cost,reward_z=standard(reward),cost_z=standard(cost),eligible=eligible,protected=prot)

class Policy(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(27,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1));self.double();nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def distribution(self,ctx):
  m=len(ctx['actions']);p1=self.net(ctx['x1']).squeeze(-1).softmax(0);p2=self.net(ctx['x2'].reshape(-1,27)).reshape(m,m).masked_fill(~ctx['mask'],-torch.inf).softmax(1)
  return p1,p2

def uniform(ctx):
 m=len(ctx['actions']);return torch.full((m,),1/m,dtype=torch.double),ctx['mask'].double()/ctx['mask'].sum(1,keepdim=True)
def joint(p):return p[0][:,None]*p[1]
def kl(p,ctx):
 q=uniform(ctx);tiny=1e-300
 return (p[0]*(p[0].clamp_min(tiny).log()-q[0].log())).sum()+(p[0]*torch.where(ctx['mask'],p[1]*(p[1].clamp_min(tiny).log()-q[1].clamp_min(tiny).log()),0).sum(1)).sum()
def draw(p,u):
 a=torch.searchsorted(p[0].cumsum(0).contiguous(),u[:,0].contiguous()).clamp_max(len(p[0])-1);b=torch.stack([torch.searchsorted(p[1][int(i)].cumsum(0).contiguous(),v).clamp_max(len(p[0])-1) for i,v in zip(a,u[:,1])]);return a,b

def modal(p):
 m=len(p[0]);k=int(joint(p).argmax());return k//m,k%m

def summary(ctx,out,p,tail):
 j=joint(p).detach();count=out['count'];den=count.clamp_min(1);dc=out['delta_correct'];corr=out['base_correct']+(j.sum(1)[:,None]*dc).sum(0)+(j.sum(0)[:,None]*dc).sum(0);a,b=modal(p);mode=out['base_correct']+dc[a]+dc[b];eligible=out['eligible'].tolist();tc=[c for c in eligible if tail[c]]
 def met(v,cc):return None if not cc else float((v[cc]/den[cc]).mean()*100)
 return dict(counts=count.tolist(),expected_correct=corr.tolist(),modal_correct=mode.tolist(),expected_macro=met(corr,eligible),expected_tail=met(corr,tc),modal_macro=met(mode,eligible),modal_tail=met(mode,tc),modal_actions=[list(ctx['actions'][a]),list(ctx['actions'][b])],legal_actions=len(ctx['actions']),legal_trajectories=int(ctx['mask'].sum()),expected_reward=float((j*out['reward']).sum()))
