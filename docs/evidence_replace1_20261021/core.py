"""Normalized conditional densities and one fixed spatial latent mixture."""
import math
import torch
from torch import nn

ARMS=['G','E','E_shuffle']

class Flow(nn.Module):
    def __init__(self,d=768,context=34,width=128):
        super().__init__();self.d=d
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(d//2+context,width),nn.SiLU(),nn.Linear(width,d)) for _ in range(2)])
        for layer in self.layers:nn.init.zeros_(layer[-1].weight);nn.init.zeros_(layer[-1].bias)
    def encode(self,u,condition):
        x=u;ld=u.new_zeros(u.shape[:-1])
        for layer in self.layers:
            a,b=x.chunk(2,-1);s,t=layer(torch.cat([a,condition],-1)).chunk(2,-1)
            s=.5*torch.tanh(s);x=torch.cat([(b-t)*torch.exp(-s),a],-1);ld-=s.sum(-1)
        return x,ld
    def inverse(self,x,condition):
        for layer in reversed(self.layers):
            b,a=x.chunk(2,-1);s,t=layer(torch.cat([a,condition],-1)).chunk(2,-1)
            s=.5*torch.tanh(s);x=torch.cat([a,b*torch.exp(s)+t],-1)
        return x
    def ratio(self,u,condition):
        x,ld=self.encode(u,condition)
        return -.5*(x.square().sum(-1)-u.square().sum(-1))+ld


def layout(device='cpu',dtype=torch.float32):
    mats=[];lp=[]
    for size in [1,2,4,7,14]:
        stride=max(1,size//2);positions=[(r,c) for r in range(0,15-size,stride) for c in range(0,15-size,stride)]
        for r,c in positions:
            m=torch.zeros(14,14,device=device,dtype=dtype);m[r:r+size,c:c+size]=1
            mats.append(m.flatten());lp.append(-math.log(5*len(positions)))
    mask=torch.stack(mats);prior=torch.tensor(lp,device=device,dtype=dtype)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,14,device=device,dtype=dtype),torch.linspace(-1,1,14,device=device,dtype=dtype),indexing='ij')
    pos=torch.stack([yy.flatten(),xx.flatten()],-1)
    perm=torch.randperm(196,generator=torch.Generator().manual_seed(74002+384)).to(device)
    return mask,prior,pos,perm


def aggregate(ratio,arm,mask,prior,perm):
    if arm=='G':return ratio.sum(-1)
    if arm=='E_shuffle':ratio=ratio[:,perm]
    return torch.logsumexp(ratio@mask.T+prior,dim=-1)


def context(cls,projection):
    c=cls.to(projection.dtype)@projection
    return torch.cat([c,torch.ones(len(c),1,device=c.device,dtype=c.dtype)],-1)


def whiten(patch,cls,q):
    x=context(cls,q['projection']);mu=torch.einsum('bi,jid->bjd',x,q['beta'])
    u=(patch.to(mu.dtype)-mu)/q['sigma'][None]
    return u,x[:,:32]


def conditions(c,pos):return torch.cat([c[:,None].expand(-1,196,-1),pos[None].expand(len(c),-1,-1)],-1)


def math_check():
    torch.manual_seed(1);f=Flow(4,3,8).double();u=torch.randn(5,4,dtype=torch.float64);c=torch.randn(5,3,dtype=torch.float64)
    assert torch.equal(f.ratio(u,c),torch.zeros(5,dtype=torch.float64))
    with torch.no_grad():
        for l in f.layers:l[-1].weight.normal_(0,.01);l[-1].bias.normal_(0,.01)
    x,ld=f.encode(u,c);err=float((f.inverse(x,c)-u).abs().max());assert err<1e-12
    j=torch.autograd.functional.jacobian(lambda v:f.encode(v[None],c[:1])[0][0],u[0])
    de=float(abs(torch.linalg.slogdet(j)[1]-ld[0]));assert de<1e-12
    m,p,_,perm=layout(dtype=torch.float64);assert abs(float(p.exp().sum())-1)<1e-12
    z=torch.zeros(3,196,dtype=torch.float64)
    for a in ARMS:assert aggregate(z,a,m,p,perm).abs().max()<1e-12
    assert not torch.allclose(aggregate(torch.arange(196,dtype=torch.float64)[None], 'E',m,p,perm),aggregate(torch.arange(196,dtype=torch.float64)[None],'E_shuffle',m,p,perm))
    return dict(inverse_max_error=err,logdet_error=de,regions=len(m),identity_ratio_zero=True,region_prior_sum=float(p.exp().sum()))
