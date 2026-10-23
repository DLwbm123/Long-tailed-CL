"""Exercise real parameter-gradient hooks, including native mapping projection."""
from types import SimpleNamespace
import torch
from teacher import selfcheck,apply_anchor,verify_anchor
print(selfcheck())
class Tiny:
    def __init__(self):
        self.w=torch.nn.Parameter(torch.randn(4,4));self.a=torch.nn.Parameter(torch.randn(4,4))
        self.learned_lorentz_exp=SimpleNamespace(project_gradient=lambda g:g@torch.diag(torch.tensor([1.,1.,0.,0.])))
    def named_parameters(self):return [('learned_lorentz_exp.exp_map.0.weight',self.w),('Image_Adapter.0.weight',self.a)]
for sign in [1,-1]:
    net=Tiny();net.w.grad=torch.randn_like(net.w);net.a.grad=torch.randn_like(net.a)
    named=list(net.named_parameters());original=[p.grad.clone() for name,p in named]
    learner=SimpleNamespace(_network=net,_cur_task=1,audit={'teacher':dict(conflict_batches=0,alpha_min=1.,alpha_max=0.,max_applied_parameter_ratio=0.,min_added_base_cosine=1.)})
    learner._teacher_pending=(named,[sign*g for g in original]);apply_anchor(learner)
    net.w.grad=net.learned_lorentz_exp.project_gradient(net.w.grad);verify_anchor(learner)
    assert learner.audit['teacher']['max_applied_parameter_ratio']<=.1001
print('parameter hooks and native projection passed')
from teacher import class_balanced_kl,consistent_mask
logits=torch.randn(3,3,dtype=torch.double,requires_grad=True)
q=torch.tensor([[.8,.1,.1],[.8,.1,.1],[1/3,1/3,1/3]],dtype=torch.double)
y=torch.tensor([0,1,2]);assert consistent_mask(q,y).tolist()==[True,False,False]
loss=class_balanced_kl(logits,q,y);gradient=torch.autograd.grad(loss,logits)[0]
assert gradient[0].norm()>0 and torch.equal(gradient[1:],torch.zeros_like(gradient[1:]))
wrong=torch.tensor([1,1,1]);loss=class_balanced_kl(logits,q,wrong);gradient=torch.autograd.grad(loss,logits)[0]
assert float(loss)==0 and torch.equal(gradient,torch.zeros_like(gradient))
permutation=torch.tensor([2,0,1]);assert torch.allclose(class_balanced_kl(logits,q,y),class_balanced_kl(logits[permutation],q[permutation],y[permutation]))
print('label consistency, rejected-row zero gradients, all-rejected no-op and permutation passed')
