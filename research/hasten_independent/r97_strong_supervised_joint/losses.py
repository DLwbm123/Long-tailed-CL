import torch
import torch.nn.functional as F
from decision import rollout,tailness
from joint import loss as grpo_loss

def KL(logits,reference):
 lp=logits.log_softmax(1);return (lp.exp()*(lp-reference.detach().log_softmax(1))).sum(1)

class Loss:
 def __init__(self,teacher,rawcounts,phase='SUP',reference=None,weights=None,generator=None,stats=None):
  self.teacher=teacher;self.rawcounts=rawcounts;self.phase=phase;self.reference=reference;self.weights=weights;self.generator=generator;self.stats={} if stats is None else stats;self.frozen={};self.update=0
 def __call__(self,m,vs,text,proto,y,j):
  raw,_,balance=m.logits(vs,text,proto,m._births);teacher=self.teacher[j].cuda();scores=m.combine(raw,teacher);logits=scores+self.rawcounts.log()[None];teacher_logits=teacher+self.rawcounts.log()[None];kd=KL(logits,teacher_logits).mean();reg=.01*m.gains().detach().abs().mean()*balance
  if self.phase=='SUP':value=F.cross_entropy(logits,y)+.1*kd+reg
  else:
   key=tuple(j.tolist())
   if self.update==0:
    assert key not in self.frozen;old=logits.detach();a,adv,_=rollout(old,y,self.generator);self.frozen[key]=(old,a,adv);self.stats['sampled_groups']=self.stats.get('sampled_groups',0)+len(y);self.stats['sampled_actions']=self.stats.get('sampled_actions',0)+a.numel()
   old,a,adv=self.frozen[key];value,st=grpo_loss(logits,y,old,a,adv,self.reference[j].cuda(),self.weights)
   for k,v in st.items():self.stats[k]=self.stats.get(k,0)+v
   value=value+.1*kd+reg
  assert torch.isfinite(value);return value,scores
