"""CPC fixed-candidate objectives; score banks are stopped TRAIN feedback."""
import itertools,math
import torch
from torch import nn
import torch.nn.functional as F
DT=torch.float64
ARMS=['C','E','G','S','M','H']
def unit(z):return F.normalize(z,dim=-1,eps=1e-12)
class Builder(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(1541,128),nn.Tanh(),nn.Linear(128,512));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias);self.double()
 def forward(self,state,p0):
  u=self.net(state);u=u-(u*p0).sum(1,keepdim=True)*p0;r=u*(.1/u.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp_max(1)
  return torch.stack([p0,unit(p0+.5*r),unit(p0+r)],1),r
class Actor(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(2054,64),nn.Tanh(),nn.Linear(64,3));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias);self.double()
 def forward(self,s):return self.net(s).log_softmax(1)
def state(support,classes,stage,rawcounts,birth,p0_override=None):
 p0=torch.stack([unit(z.mean(0)) for z in support]) if p0_override is None else p0_override
 variance=torch.stack([z.var(0,unbiased=True) if len(z)>1 else torch.zeros(512,dtype=DT) for z in support]);C=len(classes)
 relation=(5*p0@p0.T).masked_fill(torch.eye(C,dtype=torch.bool),-torch.inf).softmax(1)@p0-p0
 scalar=torch.tensor([[math.log1p(rawcounts[c])/math.log(1001),math.log1p(len(z))/math.log(6822),float(len(z)>1),(stage-birth[c])/5,float(birth[c]==stage)] for c,z in zip(classes,support)],dtype=DT)
 return torch.cat([p0,variance,relation,scalar],1),p0

def score(z,proto):return torch.stack([20*z.float()@proto[:,k].float().T for k in range(3)],2).to(DT)
def counts_correct(selected,y,C):
 pred=selected.argmax(1);n=torch.bincount(y,minlength=C).to(DT);correct=torch.bincount(y[pred==y],minlength=C).to(DT);return correct,n

def utility(scores,y,actions,tail,protected,reference):
 # actions BxC; no gradient through discrete outcomes.
 if actions.ndim==1:actions=actions[None]
 C=scores.shape[1];n=torch.bincount(y,minlength=C).to(DT);values=[];recalls=[];cost=[]
 for a in actions:
  selected=scores[:,torch.arange(C),a];correct,_=counts_correct(selected,y,C);rec=correct/n
  b=rec.mean()+rec[tail].mean();d=(reference-rec).clamp_min(0)[protected].mean();values.append(torch.stack([b,d,(b-d)/3]));recalls.append(rec)
  row=F.cross_entropy(selected,y,reduction='none');cost.append(torch.stack([row[y==c].mean() for c in range(C)]).mean())
 return torch.stack(values),torch.stack(recalls),torch.stack(cost)

def wins(scores,y):
 N,C,K=scores.shape;true=scores[torch.arange(N),y]
 a=true[:,:,None,None];b=scores[:,None]
 win=(a>b)|((a==b)&(y[:,None,None,None]<torch.arange(C)[None,None,:,None]))
 win=win|(y[:,None,None,None]==torch.arange(C)[None,None,:,None])
 return win.to(DT)
def exact_B(prob,mask,y,tail):
 success=((mask*prob[None,None]).sum(-1).prod(-1)*prob[y]).sum(1);C=len(prob)
 recall=torch.stack([success[y==c].mean() for c in range(C)])
 return recall.mean()+recall[tail].mean(),recall

def sampled(prob,generator,G=8):return torch.multinomial(prob,G,replacement=True,generator=generator).T

def conditional(scores,y,actions,tail,protected,reference,kind='risk'):
 G,C=actions.shape;out=torch.empty(G,C,3,dtype=DT)
 for c in range(C):
  aa=actions.repeat_interleave(3,0);aa[:,c]=torch.arange(3).repeat(G)
  v,_,ce=utility(scores,y,aa,tail,protected,reference)
  out[:,c]=(-v[:,1] if kind=='risk' else ce).reshape(G,3)
 return out

def risk_loss(logp,oldlogp,oldprob,q,local=None,shuffle=False):
 if shuffle:q=q.roll(1,-1)
 center=q-q[:,:,:1];advantage=(center-(center*oldprob[None]).sum(-1,keepdim=True)).detach();ratio=(logp-oldlogp).exp();rr=ratio[None];term=torch.minimum(rr*advantage,rr.clamp(.8,1.2)*advantage)
 if local is None:return -(term*oldprob[None]).sum((1,2)).mean()
 return -term.gather(2,local[:,:,None]).squeeze(2).sum(1).mean()
def kl(logp):return (logp.exp()*(logp+math.log(3))).sum(1).mean()
def group_loss(logp,oldlogp,actions,U):
 adv=(U-U.mean())/(U.std(unbiased=False)+1e-8);ratio=(logp[torch.arange(len(logp))[None],actions]-oldlogp[torch.arange(len(logp))[None],actions]).exp()
 return -torch.minimum(ratio*adv[:,None],ratio.clamp(.8,1.2)*adv[:,None]).sum(1).mean()
