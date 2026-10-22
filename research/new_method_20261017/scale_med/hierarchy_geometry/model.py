"""Matched-capacity geometry/semantic-structure factorial, not full HASTEN."""
import torch
from torch import nn
from torch.nn import functional as F
from core import TYPE,SUB,permutation
ARMS=['Eflat','Etrue','Eshuffled','Hflat','Htrue','Hshuffled']

def distance(p,q,geometry):
    p=p.double();q=q.double()
    if geometry=='E':return ((p-q).square().sum(-1)+1e-10).sqrt()
    alpha=(1+p.square().sum(-1)).sqrt()*(1+q.square().sum(-1)).sqrt()-(p*q).sum(-1)
    return torch.acosh(alpha.clamp_min(1+1e-10))

def entail(p,q,geometry):
    p=p.double();q=q.double();pn=p.norm(dim=-1).clamp_min(1e-8)
    aperture=torch.asin((.2/pn).clamp(max=1-1e-5))
    if geometry=='E':
        d=q-p;cos=(p*d).sum(-1)/(pn*d.norm(dim=-1).clamp_min(1e-8))
    else:
        pt=(1+p.square().sum(-1)).sqrt();qt=(1+q.square().sum(-1)).sqrt()
        alpha=(pt*qt-(p*q).sum(-1)).clamp_min(1+1e-10)
        cos=(qt-pt*alpha)/(pn*(alpha.square()-1).sqrt())
    exterior=torch.acos(cos.clamp(-1+1e-5,1-1e-5))
    return F.softplus(exterior-aperture)

def edges(seed,shuffled):
    perm=permutation(seed) if shuffled else list(range(13));pairs=[]
    for c in range(13):pairs.append((17+SUB[c] if SUB[c]>=0 else 13+TYPE[c],perm[c]))
    pairs += [(14,17),(14,18),(14,19)]
    return pairs

class Model(nn.Module):
    def __init__(self,dim=512):
        super().__init__();self.visual=nn.Parameter(torch.eye(dim));self.text=nn.Parameter(torch.eye(dim));self.mapping=nn.Parameter(torch.eye(dim))
    def representations(self,x,t):return x@self.visual.T@self.mapping.T,t@self.text.T@self.mapping.T
    def logits(self,x,t,geometry):
        v,z=self.representations(x,t)
        return -10*distance(v[:,None,:],z[None,:13,:],geometry)
    def objective(self,x,t,y,arm,seed):
        geometry=arm[0];v,z=self.representations(x,t)
        logits=-10*distance(v[:,None,:],z[None,:13,:],geometry)
        ce=F.cross_entropy(logits,y)
        eye=torch.eye(self.visual.shape[0],device=x.device)
        reg=sum((m-eye).square().mean() for m in [self.visual,self.text,self.mapping])*.001
        tt=ti=ce*0
        if not arm.endswith('flat'):
            ij=torch.tensor(edges(seed,arm.endswith('shuffled')),device=x.device);p=z[ij[:,0]];c=z[ij[:,1]]
            tt=(entail(p,c,geometry)+F.softplus(.2-distance(p,c,geometry))).mean()
            ti=entail(z[y],v,geometry).mean()
        return ce+reg+.5*tt+.1*ti,dict(ce=ce,regularizer=reg,text_tree=tt,image_anchor=ti)

def selfcheck():
    p=torch.tensor([[1.,0.]],dtype=torch.double);q=torch.tensor([[2.,0.]],dtype=torch.double)
    for g in ['E','H']:
        assert distance(p,q,g)>0 and torch.allclose(distance(p,q,g),distance(q,p,g))
        assert entail(p,q,g)<entail(q,p,g)
    expected=torch.acosh(torch.tensor(10.,dtype=torch.double).sqrt()-2)
    assert torch.allclose(distance(p,q,'H')[0],expected)
    torch.manual_seed(47);x=F.normalize(torch.randn(13,8),dim=1);t=F.normalize(torch.randn(20,8),dim=1);y=torch.arange(13)
    for arm in ARMS:
        m=Model(8);val,_=m.objective(x,t,y,arm,74510);val.backward()
        assert torch.isfinite(val) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
        if arm.endswith('flat'):
            t2=t.clone();t2[13:]=torch.randn_like(t2[13:]);v2,_=m.objective(x,t2,y,arm,74510);assert torch.equal(val,v2)
    assert edges(74510,False)!=edges(74510,True)
    assert sorted(c for p,c in edges(74510,True) if c<13)==list(range(13))
    return dict(distance_symmetry=True,lorentz_formula=True,cone_direction=True,all_arm_finite_gradients=True,flat_parent_independence=True,shuffled_leaf_bijection=True)
if __name__=='__main__':print(selfcheck())
