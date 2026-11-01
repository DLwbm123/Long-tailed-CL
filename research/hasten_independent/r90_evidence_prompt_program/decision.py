"""Pure shared policy/utility interface; witness inputs never accept F labels."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as TF
from group_math import coefficients

ARMS=['C','E','X','G','H']
class Policy(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(53,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,4));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias);self.double()
 def tree(self,states):
  lp=self.net(states).log_softmax(1);slots=torch.stack([lp[0].repeat_interleave(4),lp[1:].reshape(-1)],1);joint=slots.sum(1);return joint.exp(),slots,joint

def outcomes(scores,labels,n,old,fullcounts):
 # scores16 x rows x classes; class counts/correct retained for utility/CE.
 assert len(labels)>0 and np.all(np.bincount(labels,minlength=n)>0)
 cnt=np.bincount(labels,minlength=n);pred=scores.argmax(2);rec=np.stack([np.array([(pred[k,labels==c]==c).mean() for c in range(n)]) for k in range(16)])
 logits=torch.as_tensor(scores,dtype=torch.float64)*20;yy=torch.as_tensor(labels);loss=-logits.log_softmax(2).gather(2,yy[None,:,None].expand(16,-1,1)).squeeze(2).numpy();ce=np.stack([np.array([loss[k,labels==c].mean() for c in range(n)]) for k in range(16)])
 anchor=rec[0].copy()
 if old:
  predold=scores[0,labels<old,:old].argmax(1);labold=labels[labels<old]
  for c in range(old):anchor[c]=(predold[labold==c]==c).mean()
 protected=(np.arange(n)<old)|(np.asarray(fullcounts[:n])<250)
 damage=np.maximum(anchor[None]-rec,0)[:,protected].mean(1)
 return {'counts':cnt,'recall':rec,'class_CE':ce,'cost':ce.mean(1),'reward':rec.mean(1)-damage,'anchor':anchor}

def state_tree(wscores,wlabels,wr,n,old,fullcounts,birth,s_counts,w_counts,stage):
 yy=np.asarray(wlabels);idx=np.arange(n);masks=[np.ones(n,dtype=bool),idx<old,np.asarray(fullcounts[:n])<250,idx>=old]
 margins=np.empty((16,n));ce=wr['class_CE'];rec=wr['recall']
 for p in range(16):
  ss=wscores[p].copy();true=ss[np.arange(len(yy)),yy];ss[np.arange(len(yy)),yy]=-np.inf;gap=true-ss.max(1)
  for c in range(n):margins[p,c]=gap[yy==c].mean()
 def pool(x,mask):return x[mask].mean(0) if mask.any() else np.zeros(x.shape[-1])
 rows=[]
 for node,program in enumerate([0,0,4,8,12]):
  cols=np.stack([np.log(np.asarray(fullcounts[:n]))/math.log(1000),np.log(s_counts)/math.log(21),np.log(w_counts)/math.log(21),(stage-np.asarray(birth))/5,(idx>=old).astype(float),rec[program],margins[program]/2,ce[program]/40],1)
  basecols=np.stack([rec[0],margins[0]/2,ce[0]/40],1);diff=cols[:,5:]-basecols
  previous=np.zeros(4)
  if node:previous[node-1]=1
  vector=np.concatenate([*(pool(cols,z) for z in masks),[float(old>0),float(masks[2].any()),float(old<n),math.log(n)/math.log(23),float(node>0)],previous,*(pool(diff,z) for z in masks)])
  assert vector.shape==(53,) and np.isfinite(vector).all();rows.append(vector)
 return torch.tensor(np.stack(rows),dtype=torch.float64)

def standardized(x):
 z=torch.as_tensor(x,dtype=torch.float64);return (z-z.mean())/z.std(unbiased=False).clamp_min(1e-8)
def kl(joint,logjoint):return (joint*(logjoint+math.log(16))).sum()
def group_loss(slots,oldslots,actions,rewards,derange=False):
 rr=torch.as_tensor(rewards,dtype=torch.float64)[actions]
 if derange:rr=rr.roll(-1)
 adv=(rr-rr.mean())/(rr.std(unbiased=False)+1e-8);ratio=(slots[actions]-oldslots[actions]).exp();return -torch.minimum(ratio*adv[:,None],ratio.clamp(.8,1.2)*adv[:,None]).sum(1).mean()
def exact_group_constants(oldjoint,reward,group):
 return coefficients([(float(p),()) for p in oldjoint],list(map(float,reward)),group)
def exact_group_loss(slots,oldslots,oldjoint,co):
 pos,neg,mass,index,_=co;scale=torch.tensor([float(oldjoint[i])/mass[k] for i,k in enumerate(index)],dtype=torch.float64);bp=torch.tensor([pos[k] for k in index],dtype=torch.float64);bn=torch.tensor([neg[k] for k in index],dtype=torch.float64);ratio=(slots-oldslots).exp();return -(scale[:,None]*(bp[:,None]*torch.minimum(ratio,ratio.clamp(.8,1.2))+bn[:,None]*torch.maximum(ratio,ratio.clamp(.8,1.2)))).sum()
def sample(oldslots,uniforms):
 first=oldslots[:,0].reshape(4,4)[:,0].exp().numpy();second=oldslots[:,1].reshape(4,4).exp().numpy();actions=[]
 for u,v in uniforms:
  a=min(int(np.searchsorted(first.cumsum(),u,side='right')),3);b=min(int(np.searchsorted(second[a].cumsum(),v,side='right')),3);actions.append(4*a+b)
 return torch.tensor(actions,dtype=torch.long)
