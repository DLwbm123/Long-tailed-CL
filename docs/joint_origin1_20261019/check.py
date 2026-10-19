import torch
from core import surrogate_transform,same_block

torch.set_default_dtype(torch.float64);torch.set_num_threads(4);torch.manual_seed(74002)
c=torch.randn(29,4);m=torch.randn(29,4);u=torch.cat([c,torch.ones(29,1)],1);y=torch.randn(29,3);w=torch.rand(29);w/=w.sum()
s=1.2;alpha=1.5;g=u.T@(u*w[:,None]);b=u.T@(y*w[:,None]);cross=u.T@(m*w[:,None])
z=torch.cat([c/2**.5,s*m/2**.5,torch.ones(29,1)],1)
head=torch.linalg.solve(z.T@(z*w[:,None])+alpha*torch.eye(9),z.T@(y*w[:,None]))
base,p=same_block(g,b,cross,alpha)
raw_m=head[4:8]*s/2**.5
assert torch.allclose(z@head,u@base+(m-u@p)@raw_m,atol=1e-12)
raw=torch.cat([c,m,torch.ones(29,1)],1);pen=torch.tensor([2*alpha]*4+[2*alpha/s**2]*4+[alpha])
h=torch.linalg.solve(raw.T@(raw*w[:,None])+torch.diag(pen),raw.T@(y*w[:,None]))
assert torch.allclose(raw@h,z@head,atol=1e-12)
a=torch.linalg.solve(g+torch.eye(5),cross);t=surrogate_transform(a,s)
assert torch.allclose(u@t,torch.cat([c/2**.5,s*(u@a)/2**.5,torch.ones(29,1)],1))
assert torch.allclose(t.T@g@t,(u@t).T@((u@t)*w[:,None]),atol=1e-12)
print('PASS: raw penalties; Schur reconstruction; surrogate features and sufficient statistics')
