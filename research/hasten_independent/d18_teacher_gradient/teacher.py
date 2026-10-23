"""TRAIN-only discriminative teacher with an image-geometry gradient budget."""
import torch
import torch.nn.functional as F

def teacher_prob(raw, weight):
    with torch.no_grad():
        x=F.normalize(raw.double(),dim=-1)
        z=torch.cat([x,torch.ones(len(x),1,device=x.device,dtype=x.dtype)],1)@weight.to(x)
        positive=z.clamp_min(0);mass=positive.sum(1,keepdim=True)
        q=positive/mass.clamp_min(1e-12)
        q=torch.where(mass>1e-12,q,torch.full_like(q,1/q.shape[1]))
        assert torch.isfinite(q).all() and torch.allclose(q.sum(1),torch.ones(len(q),device=q.device,dtype=q.dtype))
    return q

def class_balanced_kl(logits, q, labels):
    per=F.kl_div(F.log_softmax(logits,dim=1),q,reduction='none').sum(1)
    return torch.stack([per[labels==c].mean() for c in labels.unique()]).mean()

def bounded_anchor(contrast, kd, image_geometry):
    gc=torch.autograd.grad(contrast,image_geometry,retain_graph=True)[0]
    gt=torch.autograd.grad(kd,image_geometry,retain_graph=True)[0]
    nc=gc.detach().double().norm();nt=gt.detach().double().norm()
    alpha=torch.minimum(torch.ones_like(nc),.1*nc/nt.clamp_min(1e-12)).detach()
    assert torch.isfinite(alpha) and 0<=float(alpha)<=1
    assert float(alpha*nt)<=float(.1*nc)+1e-10
    return alpha*kd,dict(alpha=float(alpha),contrast_geometry_grad=float(nc),teacher_geometry_grad=float(nt),added_geometry_grad=float(alpha*nt))

def selfcheck():
    torch.manual_seed(42)
    x=torch.randn(6,4,dtype=torch.double,requires_grad=True);w=torch.randn(5,3,dtype=torch.double,requires_grad=True)
    q=teacher_prob(x,w);assert not q.requires_grad
    y=torch.tensor([0,0,0,1,1,2]);z=x@torch.randn(4,3,dtype=torch.double)
    kd=class_balanced_kl(z,q,y);con=z.square().mean();anchor,audit=bounded_anchor(con,kd,x)
    assert torch.isfinite(torch.autograd.grad(con+anchor,x)[0]).all()
    permutation=torch.tensor([5,2,1,4,0,3]);assert torch.allclose(kd,class_balanced_kl(z[permutation],q[permutation],y[permutation]))
    assert torch.allclose(teacher_prob(x,torch.zeros_like(w)),torch.full_like(q,1/3))
    return dict(passed=True,teacher_detached=True,class_balanced_permutation=True,gradient_budget_passed=True,**audit)
