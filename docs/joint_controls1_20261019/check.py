import torch
from core import spectrum,degrees,match_alpha,scaled

torch.manual_seed(74002);torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
c=torch.randn(31,5);p=torch.randn(31,5)*.2;w=torch.rand(31);w/=w.sum()
a=torch.cat([c,torch.ones(31,1)],1);x=torch.cat([c,p,torch.ones(31,1)],1);y=torch.randn(31,3)
g=x.T@(x*w[:,None]);t=x.T@(y*w[:,None])
s=float((c.square()*w[:,None]).sum()/(p.square()*w[:,None]).sum())**.5
ng,nt,f=scaled(g,t,s);z=x*f
assert torch.allclose(ng,z.T@(z*w[:,None]),atol=1e-12)
assert torch.allclose(nt,z.T@(y*w[:,None]),atol=1e-12)
assert abs(float((z[:,:-1].square()*w[:,None]).sum()-(c.square()*w[:,None]).sum()))<1e-12
cg=a.T@(a*w[:,None]);target=degrees(spectrum(cg),1.)
alpha=match_alpha(spectrum(ng),target)
# Independent hat-matrix trace oracle, including penalized intercept.
fit=torch.linalg.solve(ng+alpha*torch.eye(11),z.T*w)
hat=z@fit
assert abs(float(hat.trace())-target)<1e-10
q=torch.cat([c/2**.5,c/2**.5,torch.ones(31,1)],1)
base=a@torch.linalg.solve(cg+torch.eye(6),a.T@(y*w[:,None]))
dup=q@torch.linalg.solve(q.T@(q*w[:,None])+torch.eye(11),q.T@(y*w[:,None]))
assert torch.allclose(base,dup,atol=1e-12)
assert degrees(spectrum(ng),alpha/2)>target>degrees(spectrum(ng),alpha*2)
print('PASS: scaled moments and energy; df root; independent hat trace; duplicate invariance')
