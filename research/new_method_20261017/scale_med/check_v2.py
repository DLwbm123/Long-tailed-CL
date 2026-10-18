import torch
from run_v2 import local_scores,centered_energy,fuse,ridge_model
from core import gaussian,logpdf

torch.manual_seed(3)
x=torch.randn(3,5,4,dtype=torch.float64);cov=torch.eye(4,dtype=torch.float64)
stats=[{'groups':n,'local':(torch.randn(4,dtype=torch.float64),cov*(i+1))} for i,n in enumerate([1,20,100])]
ref={'local':(torch.zeros(4),cov,.001)}
a,b=local_scores(x,stats,ref,20)
assert torch.allclose(b[:,0],logpdf(x,gaussian(stats[0]['local'][0],cov,.001)).mean(1)/4)
small=local_scores(x,stats[:2],ref,20)
assert torch.equal(a[:,:2],small[0]) and torch.equal(b[:,:2],small[1])
r=torch.randn(3,3,dtype=torch.float64)
assert torch.allclose(centered_energy(r),centered_energy(r+torch.randn(3,1)))
assert torch.allclose(fuse(r,b,[2,3],.5),r/2+b/6)
assert torch.equal(fuse(r,b,[2,3],0).argmax(-1),r.argmax(-1))
try:fuse(r,b,[0,1],.5)
except AssertionError:pass
else:raise AssertionError('Zero scale accepted')
print('PASS: local density, old-score invariance, scale offset invariance, fusion and invalid-scale checks')
