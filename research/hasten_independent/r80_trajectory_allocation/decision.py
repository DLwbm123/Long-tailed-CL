"""Frozen two-action training trajectories and controller-only GRPO."""
import torch
from torch import nn
import torch.nn.functional as F
ARMS=('balanced_short','macro_controller','protected_controller','shuffled_controller','balanced_long')
LEARNED=ARMS[1:4]
CONTEXTS=8
GROUP=8
HORIZON=3
class Controller(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(8,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh());self.cls=nn.Linear(128,1);self.bal=nn.Linear(64,1)
  nn.init.zeros_(self.cls.weight);nn.init.zeros_(self.cls.bias);nn.init.zeros_(self.bal.weight);nn.init.zeros_(self.bal.bias)
 def forward(self,state,legal):
  z=self.encoder(state);pool=z.mean(0);v=torch.cat([self.bal(pool).reshape(1),self.cls(torch.cat([z,pool.expand_as(z)],1)).flatten()]);return v.masked_fill(~legal,-torch.inf)

def allocation(labels,support,common,priority,generator):
 """Unique batch: 192 common rows, up to 64 priority rows, then balanced unique fill."""
 assert len(common)==192 and len(set(common))==192 and set(common)<=set(support)
 chosen=list(common);seen=set(chosen);n=int(labels.max())+1
 pools=[]
 for c in range(n):
  q=[i for i in support if int(labels[i])==c and i not in seen]
  if q:q=[q[j] for j in torch.randperm(len(q),generator=generator).tolist()]
  pools.append(q)
 if priority>=0:
  take=pools[priority][:64];chosen+=take;seen.update(take);pools[priority]=pools[priority][len(take):]
 while len(chosen)<256:
  progress=False
  for c in torch.randperm(n,generator=generator).tolist():
   if pools[c] and len(chosen)<256:chosen.append(pools[c].pop());progress=True
  assert progress,'INSUFFICIENT_UNIQUE_SUPPORT_FOR_BATCH'
 assert len(chosen)==len(set(chosen))==256 and set(chosen)<=set(support)
 return chosen

def legal_actions(labels,support,common):
 left=torch.bincount(labels[[i for i in support if i not in set(common)]],minlength=int(labels.max())+1)
 return torch.cat([torch.ones(1,dtype=torch.bool),left>0])

def utility(class_brier,anchor,protected,penalty=True):
 damage=(class_brier-anchor).clamp_min(0)
 d=damage[protected].mean() if bool(protected.any()) else class_brier.sum()*0
 return -class_brier.mean()-(d if penalty else 0),d

def grpo_loss(controller,states,masks,actions,old_logp,advantages):
 logps=[];kls=[]
 for ss,mm,aa in zip(states,masks,actions):
  v=controller(ss,mm);lp=F.log_softmax(v,dim=0);p=lp.exp();logps.append(lp[aa]);kls.append((p[mm]*(lp[mm]+mm.sum().float().log())).sum())
 lp=torch.stack(logps).reshape(GROUP,2);ratio=(lp-old_logp).exp();a=advantages[:,None]
 surrogate=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean();kl=torch.stack(kls).mean()
 return surrogate+.04*kl,dict(surrogate=float(surrogate.detach()),controller_KL=float(kl.detach()),clipped_fraction=float(((ratio<.8)|(ratio>1.2)).float().mean()))
