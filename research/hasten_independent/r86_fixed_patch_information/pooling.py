import torch
import torch.nn.functional as F

def pool(model,tokens):
    h=model.norm(model.blocks(tokens));assert h.shape[1:]==(197,768)
    cls=F.normalize(model.proj(h[:,0]),dim=-1)
    patches=F.normalize(model.proj(h[:,1:]),dim=-1)
    mean=patches.mean(1);assert torch.all(mean.norm(dim=1)>1e-8)
    patch=F.normalize(mean,dim=-1)
    assert torch.isfinite(cls).all() and torch.isfinite(patch).all()
    return cls,patch

def channels(cls,patch):
    assert cls.shape==patch.shape
    return dict(patch=F.normalize(torch.cat([cls,patch],dim=1),dim=1),
                duplicate=F.normalize(torch.cat([cls,cls],dim=1),dim=1))
