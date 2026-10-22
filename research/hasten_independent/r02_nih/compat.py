"""Auditable runtime corrections to pinned upstream; original checkout stays intact."""
import importlib
import json
import random
import sys
import types
from pathlib import Path
import torch
from torch import nn

PIN = '063f295d08991d4da5b1f46a61bfcd279a94dbc6'

def class_statistics(x):
    assert x.ndim == 2 and len(x) > 0
    mean = x.mean(0)
    centered = x - mean
    scatter = centered.T @ centered / (len(x)-1) if len(x)>1 else torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
    cov = scatter + 1e-5*torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    assert mean.ndim == 1 and torch.isfinite(cov).all()
    return mean, cov

def sample_classes(pool, count, rng=random):
    return rng.choices(tuple(pool), k=count) if count>0 else list(pool)

def corrected_basis(cov, threshold=.95):
    # Preserve upstream squared-singular-value energy and min_keep convention.
    _, s, v = torch.svd(cov.double())
    assert len(s)>10 and s.square().sum()>0
    r = min(int(torch.searchsorted(s.square().cumsum(0)/s.square().sum(), threshold))+1, len(s)-10)
    return v[:, r:].to(cov.dtype).detach()

def apply_delta_projection(weight, before, basis):
    p=basis.double()@basis.double().T
    delta=weight.detach().double()-before.double()
    with torch.no_grad():weight.copy_((before.double()+delta@p).to(weight.dtype))
    actual=weight.detach().double()-before.double()
    residual=(actual@(torch.eye(len(p),device=p.device,dtype=p.dtype)-p)).norm()
    return dict(delta_norm=float(actual.norm()), off_subspace_norm=float(residual), rank=int(basis.shape[1]))

def load_upstream(root):
    root=Path(root);sys.path.insert(0,str(root))
    import utils.inc_net as inc
    inc.LearnableLorentzExp.compute_null_space=lambda self,energy_threshold=.95:setattr(self,'null_space_basis',corrected_basis(self.accumulated_cov,energy_threshold))
    def project(self,g):
        if self.null_space_basis is None:return g
        b=self.null_space_basis.to(g)
        return g@(b@b.T)
    inc.LearnableLorentzExp.project_gradient=project
    import utils.lorentz as geom
    # Same formulas/constants, FP64 geometry so 1+1e-8 does not round to 1.
    for name in ['pairwise_inner','element_inner','pairwise_dist','element_dist','exp_map0','oxy_angle']:
        if hasattr(geom,name):
            original=getattr(geom,name)
            def promoted(*args,_fn=original,**kw):
                return _fn(*[a.double() if torch.is_tensor(a) else a for a in args],**kw)
            setattr(geom,name,promoted)
    source=(root/'models/hasten.py').read_text()
    changes=[
      ('num_workers = 8','num_workers = 0'),
      ('batch_size=64, shuffle=False, num_workers=8','batch_size=64, shuffle=False, num_workers=0'),
      ('                old_class = list(','                old_pool = list('),
      ('                random.shuffle(old_class)','                random.shuffle(old_pool)'),
      ('random.choices(old_class, k=self.args["sample_num"])','random.choices(old_pool, k=self.args["sample_num"])'),
      ('                        else old_class','                        else old_pool'),
      ('                    for j in old_class:','                    self._record_replay(old_class)\n                    for j in old_class:'),
      ('            index = torch.nonzero(y == l).squeeze()','            index = torch.nonzero(y == l, as_tuple=True)[0]'),
      ('            l_mean = l_fea.mean(dim=0)\n            self.prototype.append(l_mean)\n            l_cov = torch.cov((l_fea - l_mean).T) + 1e-05 * torch.eye(\n                self._network.feature_dim, device=self._device\n            )','            l_mean, l_cov = class_statistics(l_fea)\n            self.prototype.append(l_mean)'),
      ('                optimizer.zero_grad()','                self._inspect_loss(loss, contrastive_loss, entailment_loss, hierarchy_loss)\n                optimizer.zero_grad()'),
      ('                optimizer.step()','                self._before_step()\n                optimizer.step()\n                self._after_step()'),
    ]
    for old,new in changes:
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new)
    mod=types.ModuleType('models.hasten');mod.__file__=str(root/'models/hasten.py');mod.class_statistics=class_statistics
    sys.modules['models.hasten']=mod
    exec(compile(source,'<pinned-upstream-compatible>','exec'),mod.__dict__)
    return inc,mod,geom

class FrozenFeatureModel(nn.Module):
    def __init__(self, bank, logit_scale):
        super().__init__();self.out_dim=512
        self.register_buffer('text_bank',bank.detach().clone())
        self.register_buffer('logit_scale',torch.tensor(float(logit_scale)))
        self.visual=nn.Module();self.visual.register_buffer('proj',torch.eye(512))
    def encode_image(self,x,normalize=False):
        assert x.ndim==2 and x.shape[1]==512
        return torch.nn.functional.normalize(x,dim=-1) if normalize else x
    def encode_text(self,t,normalize=False):
        z=self.text_bank[t.flatten().long()]
        return torch.nn.functional.normalize(z,dim=-1) if normalize else z

class TokenLookup:
    def __init__(self,prompts):self.indices={p:i for i,p in enumerate(prompts)}
    def __call__(self,texts):return torch.tensor([self.indices[t] for t in texts],dtype=torch.long)

def install_feature_model(inc,bank,prompts,scale):
    def factory(args,pretrained=False):return FrozenFeatureModel(bank,scale),None,TokenLookup(prompts)
    inc.get_convnet=factory

def selfcheck():
    torch.manual_seed(42)
    stats=[]
    for n in [1,2,4]:
        mean,cov=class_statistics(torch.randn(n,16,dtype=torch.double))
        torch.distributions.MultivariateNormal(mean,covariance_matrix=cov).sample((2,))
        stats.append(dict(n=n,mean_shape=list(mean.shape),covariance_shape=list(cov.shape)))
    pool=tuple(range(8));rng=random.Random(7);seen=set()
    for _ in range(100):seen.update(sample_classes(pool,3,rng))
    assert pool==tuple(range(8)) and seen==set(pool)
    x=torch.randn(64,6,dtype=torch.double)@torch.randn(6,32,dtype=torch.double)
    _,s,v=torch.svd(x.T@x);basis=v[:,6:];p=basis@basis.T;g=torch.randn(32,32,dtype=torch.double)
    ratio=float((x@(g@p).T).norm()/(x@g.T).norm());assert ratio<1e-10
    w=nn.Parameter(torch.randn(32,32,dtype=torch.double));opt=torch.optim.AdamW([w],lr=.01,weight_decay=.1)
    before=w.detach().clone();w.grad=g@p;opt.step();receipt=apply_delta_projection(w,before,basis)
    delta=w.detach()-before;ratio_step=float((x@delta.T).norm()/delta.norm());assert ratio_step<1e-10
    return dict(statistics=stats,replay_pool_immutable=True,projection_ratio=ratio,actual_step_ratio=ratio_step,actual_step=receipt)
