"""Independent dense-density and grouping checks; no medical data or queries."""
import math
import torch
from core import finish, gaussian, joint, logpdf, models, scores


def check():
    torch.manual_seed(74002)
    torch.set_default_dtype(torch.float64)
    p, d = 4, 3
    a, b = torch.randn(d, d), torch.randn(d, d)
    w, between = a @ a.T + torch.eye(d), b @ b.T
    mu, x = torch.randn(d), torch.randn(7, p, d)
    dense = torch.kron(torch.ones(p, p), between) + torch.kron(torch.eye(p), w)
    exact = torch.distributions.MultivariateNormal(mu.repeat(p), covariance_matrix=dense).log_prob(x.flatten(1))
    value = joint(x, gaussian(mu, w), gaussian(mu, between + w / p))
    assert torch.allclose(value, exact, atol=1e-10, rtol=1e-10)
    independent = logpdf(x, gaussian(mu, w)).sum(1)
    no_shared = joint(x, gaussian(mu, w), gaussian(mu, w / p))
    assert torch.allclose(independent, no_shared, atol=1e-10, rtol=1e-10)
    xs = [torch.randn(4, d), torch.randn(12, d), torch.randn(8, d)]
    sufficient = [(len(v), v.sum(0), v.T @ v) for v in xs]
    s = finish(sufficient)
    expected_mean = torch.stack([v.mean(0) for v in xs]).mean(0)
    expected_w = torch.stack([torch.cov(v.T) for v in xs]).mean(0)
    assert torch.allclose(s['mu'], expected_mean)
    assert torch.allclose(s['W'], expected_w)
    assert torch.linalg.eigvalsh(s['B']).min() >= -1e-12
    s.update(image_mu=s['mu'], image_cov=s['pooled'])
    fit = models([s, s], s, p, 20, .001)
    values = scores(x, fit)
    assert all(torch.equal(v[:, 0], v[:, 1]) for v in values.values())
    assert values['hierarchical'].argmax(-1).eq(0).all()
    # Full-dimensional PCA whitening + log-Jacobian equals Cholesky density.
    eigen, vec = torch.linalg.eigh(w)
    z = ((x - mu) @ vec) / eigen.sqrt()
    pca = -.5 * (z.square().sum(-1) + eigen.log().sum() + d * math.log(2 * math.pi))
    assert torch.allclose(pca, logpdf(x, gaussian(mu, w)), atol=1e-10)
    one = finish(sufficient[:1])
    assert not one['between_identified'] and one['B'].eq(0).all()
    print('PASS: dense joint identity; B=0 reduction; patient moments; PSD; ties; full PCA equivalence; singleton flag')


if __name__ == '__main__':
    check()
