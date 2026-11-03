"""New strong-path/gate integration checks; no pixels, encoder or optimizer."""
import time,torch
import torch.nn.functional as F
from strong import StrongAdapter
from losses import KL
from common import write

def run(path):
 start=time.time();cpu=time.process_time();torch.manual_seed(74002);m=StrongAdapter();n=3;p=F.normalize(torch.randn(n,512),dim=1);counts=torch.tensor([10.,3.,6.]);m.context(p,counts,counts,[0]*3,0);teacher=torch.tensor([[4.,-1.,2.],[0.,3.,-2.]]);student=torch.randn(2,n,requires_grad=True);s=m.combine(student,teacher);assert torch.equal(s,teacher);y=torch.tensor([0,1]);F.cross_entropy(s+counts.log(),y).backward();assert m.gate[-1].weight.grad.norm()>0 and student.grad.abs().max()==0
 with torch.no_grad():m.gate[-1].weight.add_(m.gate[-1].weight.grad,alpha=-.0003);m.gate[-1].bias.add_(m.gate[-1].bias.grad,alpha=-.0003)
 student.grad=None;m.zero_grad();F.cross_entropy(m.combine(student,teacher)+counts.log(),y).backward();assert student.grad.norm()>0;assert m.gains().abs().max()<=.25
 with torch.no_grad():
  g=m.gains().clone();perm=torch.tensor([2,0,1]);m.context(p[perm],counts[perm],counts[perm],[0]*3,0);assert torch.allclose(m.gains(),g[perm],atol=1e-7,rtol=0)
 z=torch.randn(4,3,requires_grad=True);ref=z.detach().clone();k=KL(z,ref);assert torch.equal(k,torch.zeros_like(k));assert not ref.requires_grad;assert torch.equal(teacher,s.detach())
 write(path,{'passed':True,'zero_gate_teacher_exact':True,'first_classification_gradient_gate_only':True,'nonzero_correction_gradient_after_gate_opens':True,'gain_bound':.25,'class_permutation_equivariant':True,'teacher_detached':True,'prior_applied_in_TRAIN_loss_not_raw_combine':True,'encoder':0,'optimizer':0,'VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start})
