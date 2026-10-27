"""CPU-only correctness checks, no images or optimizer."""
import itertools,json
import torch
import torch.nn.functional as F
from decision import *
from decision import _base_objective as objective
ARMS=('sft_continue','grpo','tars_grpo','tars_exact')

def check():
 torch.manual_seed(67)
 old=torch.tensor([[.3,-.1,.2]],dtype=torch.double)
 logits=torch.tensor([[.9,-.5,.1]],dtype=torch.double,requires_grad=True)
 labels=torch.tensor([0]);p=old.softmax(1)[0];op=old.log_softmax(1);lp=logits.log_softmax(1)
 exact=exact_group_surrogate(lp,op,labels)
 brute=torch.zeros((),dtype=torch.double)
 # Exhaustive 3^8 action groups verifies clipping, zero groups and standardization.
 groups=torch.tensor(list(itertools.product(range(3),repeat=GROUP_SIZE)))
 reward=(groups==0).double();adv=(reward-reward.mean(1,keepdim=True))/(reward.std(1,keepdim=True,correction=0)+EPS)
 ratio=(lp[0][groups]-op[0][groups]).exp();prob=p[groups].prod(1)
 brute=(prob*torch.minimum(ratio*adv,ratio.clamp(1-CLIP,1+CLIP)*adv).mean(1)).sum()
 assert torch.allclose(exact,brute[None],atol=1e-12,rtol=1e-12)
 assert torch.allclose(torch.autograd.grad(exact.sum(),logits,retain_graph=True)[0],torch.autograd.grad(brute,logits,retain_graph=True)[0],atol=1e-12)
 assert float(exact_group_surrogate(op,op,labels).abs().max())<1e-12
 counts=torch.tensor([2,8,3]);correct=torch.tensor([0,7,3]);w=tailness(counts,correct);assert bool((w>=1).all()) and w[0]>w[1]
 gen=torch.Generator().manual_seed(6701)
 a,adv,r=rollout(torch.tensor([[1000.,-1000.],[-1000.,1000.]]),torch.tensor([0,0]),gen);assert torch.equal(adv,torch.zeros_like(adv))
 # Tailness applies after group normalization; KL stays unscaled.
 for arm in ARMS:
  z=logits.detach().clone().requires_grad_();loss,stats=objective(z,old,old,labels,w,arm,torch.Generator().manual_seed(17));g=torch.autograd.grad(loss.sum(),z)[0];assert torch.isfinite(loss).all() and torch.isfinite(g).all()
 z=logits.detach().clone().requires_grad_();one,_=objective(z,old,old,labels,torch.ones(3), 'tars_exact');two,_=objective(z,old,old,labels,torch.ones(3)*2,'tars_exact');kl=(z.softmax(1)*(z.log_softmax(1)-old.log_softmax(1))).sum(1);assert torch.allclose(two,2*one-BETA*kl)
 return dict(passed=True,optimizer_steps=0,no_images=True,exact_groups_enumerated=3**GROUP_SIZE,exact_expectation_and_gradient_match=True,finite_all_arms=True,zero_reward_variance_safe=True,KL_not_tail_scaled=True)


base_check=check
def check():
 result=base_check()
 import decision as d
 torch.manual_seed(70)
 for i in range(64):
  delta=[torch.randn(19,dtype=torch.double),torch.randn(7,dtype=torch.double)];guide=[torch.randn_like(z) for z in delta]
  projected,st=d.project_update(delta,guide,True);assert st['result_guide_dot']<=1e-10
  assert sum(z.square().sum() for z in projected)<=sum(z.square().sum() for z in delta)+1e-10
  unchanged,st=d.project_update(delta,guide,False);assert all(torch.equal(a,b) for a,b in zip(delta,unchanged)) and not st['projected']
  zero,st=d.project_update(delta,[torch.zeros_like(z) for z in guide],True);assert all(torch.equal(a,b) for a,b in zip(delta,zero))
 raw=torch.tensor([[.2,.3,-.1]],dtype=torch.double,requires_grad=True);old=raw.detach()-.05;counts=torch.tensor([2.,8.,3.]);labels=torch.tensor([0]);weights=torch.tensor([2.,1.,1.])
 for arm in d.ARMS:
  p=d.policy_logits(raw,counts,arm);o=d.policy_logits(old,counts,arm);assert torch.equal(p,raw+counts.double().log()[None,:]);loss,stats=d.objective(p,o,o,labels,weights,arm,torch.Generator().manual_seed(19));assert torch.isfinite(torch.autograd.grad(loss.sum(),raw,retain_graph=True)[0]).all()
 probe=torch.tensor([[.1,.2,.3]],requires_grad=True);target=torch.tensor([1]);loss=probe.square().sum();guide=torch.autograd.grad(F.cross_entropy(probe,target),probe,retain_graph=True)[0];assert probe.grad is None;loss.backward();assert torch.equal(probe.grad,2*probe.detach())
 counts=torch.tensor([2.,8.,3.],dtype=torch.double);N=counts.sum();C=len(counts)
 for arm in d.ARMS:
  w=d.guide_weights(counts,arm);assert torch.allclose((w*counts).sum(),N)
  if arm.startswith('balanced_'):assert torch.allclose(w*counts/N,torch.ones_like(counts)/C)
  else:assert torch.equal(w,torch.ones_like(counts))
 result.update(guide_class_expectation_uniform=True,natural_guide_preserved=True,no_batch_weight_renormalization=True,guide_does_not_accumulate_into_objective_gradient=True,projection_halfspace_and_minimum_change_checks=64,unprotected_step_bitwise_unchanged=True,zero_guide_safe=True,four_arms_finite=True)
 return result
if __name__=='__main__':print(json.dumps(check(),indent=2))
