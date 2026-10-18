"""Check moment reuse, nested ridge, patient weights, and missing information."""
import torch
from core import augmented, weights, project_moments, ridge, moments, gda_parameters, gda_score


torch.manual_seed(74002);torch.set_default_dtype(torch.float64)
projection=torch.linalg.qr(torch.randn(7,3),mode='reduced').Q
xs=[torch.randn(9,7),torch.randn(7,7)]
groups=[['a']*5+['b']*4,['c']*3+['d']*4]
grams=[];targets=[];projected=[];ptargets=[]
for x,gs in zip(xs,groups):
    w=weights(gs);a=augmented(x)
    gram=a.T@(a*w[:,None]);target=(a*w[:,None]).sum(0)
    p,q=project_moments(gram,target,projection)
    direct=augmented(x@projection)
    assert torch.allclose(p,direct.T@(direct*w[:,None]),atol=1e-12)
    assert torch.allclose(q,(direct*w[:,None]).sum(0),atol=1e-12)
    mu,cov=moments(p)
    assert torch.allclose(mu,((x@projection)*w[:,None]).sum(0))
    residual=x@projection-mu
    assert torch.allclose(cov,residual.T@(residual*w[:,None]),atol=1e-12)
    grams.append(gram);targets.append(target);projected.append(p);ptargets.append(q)
w=ridge(projected,ptargets,1.)
full=torch.zeros(8,2);full[:-1]=projection@w[:-1];full[-1]=w[-1]
query=torch.randn(5,7)
assert torch.allclose(augmented(query)@full,augmented(query@projection)@w,atol=1e-12)
fit,eps=gda_parameters(projected,[2,2],2,20,.001)
value=gda_score(query@projection,fit)
for c,(mu,chol,_) in enumerate(fit):
    oracle=torch.distributions.MultivariateNormal(mu,scale_tril=chol).log_prob(query@projection)
    assert torch.allclose(value[:,c],oracle,atol=1e-10)
# Equal patch marginals do not determine the covariance of image means.
a=torch.tensor([[[1.],[1.]],[[-1.],[-1.]]])
b=torch.tensor([[[1.],[-1.]],[[1.],[-1.]]])
assert torch.equal(a.flatten().sort().values,b.flatten().sort().values)
assert not torch.equal(a.mean(1).square().mean(),b.mean(1).square().mean())
print('PASS: patient weighting; projected sufficient statistics; nested ridge scores; image covariance; GDA oracle; patch marginal insufficiency')
