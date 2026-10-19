import torch
from core import joint_gram,duplicate_head

torch.set_default_dtype(torch.float64);torch.manual_seed(74002)
c=torch.randn(23,5);p=torch.randn(23,5);w=torch.rand(23);w/=w.sum()
a=torch.cat([c,torch.ones(23,1)],1);b=torch.cat([p,torch.ones(23,1)],1)
x=torch.cat([c,p,torch.ones(23,1)],1);y=torch.randn(23,3)
g=a.T@(a*w[:,None]);h=b.T@(b*w[:,None]);cross=c.T@(p*w[:,None])
assert torch.allclose(joint_gram(g,h,cross),x.T@(x*w[:,None]),atol=1e-12)
t=a.T@(y*w[:,None]);base=torch.linalg.solve(g+torch.eye(6),t)
for scale,expected in [(1.,duplicate_head(g,t)),(2**-.5,base)]:
 z=torch.cat([c*scale,c*scale,torch.ones(23,1)],1)
 fit=torch.linalg.solve(z.T@(z*w[:,None])+torch.eye(11),z.T@(y*w[:,None]))
 assert torch.allclose(z@fit,a@expected,atol=1e-12)
print('PASS: joint Gram; raw duplicate half feature penalty; calibrated duplicate exact baseline')
