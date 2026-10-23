"""Exercise real parameter-gradient hooks, including native mapping projection."""
from types import SimpleNamespace
import torch
from anchor import selfcheck,apply_anchor,verify_anchor
print(selfcheck())
class Tiny:
    def __init__(self):
        self.w=torch.nn.Parameter(torch.randn(4,4));self.a=torch.nn.Parameter(torch.randn(4,4))
        self.learned_lorentz_exp=SimpleNamespace(project_gradient=lambda g:g@torch.diag(torch.tensor([1.,1.,0.,0.])))
    def named_parameters(self):return [('learned_lorentz_exp.exp_map.0.weight',self.w),('Image_Adapter.0.weight',self.a)]
for sign in [1,-1]:
    net=Tiny();net.w.grad=torch.randn_like(net.w);net.a.grad=torch.randn_like(net.a)
    named=list(net.named_parameters());original=[p.grad.clone() for name,p in named]
    learner=SimpleNamespace(_network=net,_cur_task=1,audit={'centroid_anchor':dict(conflict_batches=0,alpha_min=1.,alpha_max=0.,max_applied_parameter_ratio=0.,min_added_base_cosine=1.)})
    learner._anchor_pending=(named,[sign*g for g in original]);apply_anchor(learner)
    net.w.grad=net.learned_lorentz_exp.project_gradient(net.w.grad);verify_anchor(learner)
    assert learner.audit['centroid_anchor']['max_applied_parameter_ratio']<=.1001
print('parameter hooks and native projection passed')

# Synthetic geometry only: verify all-centroid coverage and detached TRAIN means.
import sys,types
from anchor import centroid_loss,prepare_anchor
previous=sys.modules.get('utils')
sys.modules['utils']=types.SimpleNamespace(lorentz=types.SimpleNamespace(pairwise_dist=lambda a,b,c:torch.cdist(a,b)))
try:
    class MeanNet(torch.nn.Module):
        def __init__(self):
            super().__init__();self.fc=torch.nn.Linear(512,3,bias=False)
            self.map=torch.nn.Linear(512,5,bias=False);self.logit_scale=torch.tensor(0.)
        def Image_encode(self,x):return self.map(x)
        def trans_hyper(self,x):return x
        def get_curv(self):return torch.tensor(1.)
    net=MeanNet();text=torch.randn(3,5,requires_grad=True)
    before=net.fc.weight.detach().clone()
    loss=centroid_loss(net,text,3);loss.backward()
    assert net.fc.weight.grad is None and torch.equal(net.fc.weight,before)
    assert net.map.weight.grad.norm()>0 and (text.grad.norm(dim=1)>0).all()
    assert torch.isfinite(loss)
    print('synthetic centroid loss: all-class text gradients, active mapping gradients, detached unchanged means passed')
finally:
    if previous is None:sys.modules.pop('utils',None)
    else:sys.modules['utils']=previous
