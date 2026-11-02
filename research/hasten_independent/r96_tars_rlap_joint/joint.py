"""Frozen-behavior genuine G8 categorical TARS; reference logits fixed by stage SFT."""
import torch
from decision import rollout,tailness

def loss(logits,labels,old,actions,adv,reference,weights):
 logp=logits.log_softmax(1);oldlog=old.detach().log_softmax(1);ref=reference.detach().log_softmax(1)
 ratio=(logp.gather(1,actions)-oldlog.gather(1,actions)).exp();sur=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean(1)
 kl=(logp.exp()*(logp-ref)).sum(1);objective=(-weights[labels].to(logits.dtype)*sur+.04*kl).mean()
 stats={'groups':len(labels),'zero_advantage_groups':int((adv.abs().sum(1)==0).sum()),'actions':actions.numel(),'reward_sum':int((actions==labels[:,None]).sum()),'clipped_actions':int(((ratio<.8)|(ratio>1.2)).sum()),'KL_sum':float(kl.detach().sum())}
 return objective,stats

class Group:
 def __init__(self,reference,weights,generator,counter):self.reference=reference;self.weights=weights;self.generator=generator;self.counter=counter;self.frozen={};self.update=0
 def __call__(self,m,vs,text,proto,y,j):
  logits,_,balance=m.logits(vs,text,proto,m._births);key=tuple(j.tolist())
  if self.update==0:
   assert key not in self.frozen
   old=logits.detach();a,adv,_=rollout(old,y,self.generator);self.frozen[key]=(old,a,adv);self.counter['sampled_groups']=self.counter.get('sampled_groups',0)+len(y);self.counter['sampled_actions']=self.counter.get('sampled_actions',0)+a.numel()
  old,a,adv=self.frozen[key];value,st=loss(logits,y,old,a,adv,self.reference[j].cuda(),self.weights)
  for k,v in st.items():self.counter[k]=self.counter.get(k,0)+v
  return value+.01*balance,logits
