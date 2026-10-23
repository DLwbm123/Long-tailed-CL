import torch
from compat import corrected_image_ce,LOSS_AUDIT
torch.manual_seed(3)
y=torch.tensor([0,0,0,1,2,2]);u=torch.randn(6,3,dtype=torch.double,requires_grad=True)
a=corrected_image_ce(u[:,y],y);b=torch.nn.functional.cross_entropy(u,y)
ga=torch.autograd.grad(a,u,retain_graph=True)[0];gb=torch.autograd.grad(b,u)[0]
assert torch.allclose(a,b,atol=1e-12) and torch.allclose(ga,gb,atol=1e-12)
v=torch.randn(5,5,dtype=torch.double,requires_grad=True);yy=torch.arange(5)
assert torch.equal(corrected_image_ce(v,yy),torch.nn.functional.cross_entropy(v,yy))
p=torch.tensor([2,5,1,3,0,4]);assert torch.allclose(corrected_image_ce(u.detach()[p][:,y[p]],y[p]),a.detach(),atol=1e-12)
print({'duplicate_equivalence':True,'gradient_max_error':float((ga-gb).abs().max()),'unique_noop':True,'permutation_invariance':True})
LOSS_AUDIT.update(calls=0,examples=0,duplicate_columns=0,max_multiplicity=0)
