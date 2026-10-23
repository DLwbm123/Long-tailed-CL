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
