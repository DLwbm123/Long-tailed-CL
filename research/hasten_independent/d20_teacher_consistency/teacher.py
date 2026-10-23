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

