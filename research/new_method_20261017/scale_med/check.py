import torch
from core import group_weights, moments, covariance, gaussian, logpdf, evidence, entropy_pool, scores

def main():
    torch.manual_seed(7)
    x = torch.randn(3, 5, 4, dtype=torch.float64)
    w = group_weights(['a', 'b', 'b'])
    mu, sec = moments(x, w)
    xx = torch.cat([x[:1], x[:1], x[1:]])
    mm, ss = moments(xx, group_weights(['a','a','b','b']))
    assert torch.allclose(mu, mm) and torch.allclose(sec, ss)
    cv = covariance(mu, sec)
    actual = logpdf(x, gaussian(mu, cv, 0.01))
    expected = torch.distributions.MultivariateNormal(mu, covariance_matrix=cv + .01 * torch.eye(4)).log_prob(x)
    assert torch.allclose(actual, expected)
    q = torch.tensor([[[-9.,-5.,-102.],[-8.,-9.,-104.],[-4.,-6.,-102.],[-3.,-5.,-105.]]],dtype=torch.float64)
    assert entropy_pool(q[:,:,:2]).argmax(-1).item() == 1
    assert entropy_pool(q).argmax(-1).item() == 0
    assert torch.equal(evidence(q, torch.zeros(1,4), .5)[:,:,:2], evidence(q[:,:,:2], torch.zeros(1,4), .5))
    stats = [{'groups': n, 'local': (mu+n/100,cv), 'global':(mu+n/100,cv)} for n in [1,10,100]]
    ref = {'local':(mu,cv,.01), 'global':(mu,cv,.01)}
    cfg = {'tau':20.,'eta':.5,'local_lambda':.5}
    a,b = scores(x,x[:,0],stats[:2],ref,cfg), scores(x,x[:,0],stats,ref,cfg)
    for k in a:
        if k != 'entropy': assert torch.equal(a[k], b[k][:,:2]), k
    assert torch.isfinite(b['global_local']).all()
    print('PASS: group duplication, normalized Gaussian, entropy rank flip, fixed old scores, singleton shrinkage')

if __name__ == '__main__': main()
