"""Gaussian score integration and fixed, training-side head selection."""
import math
import torch


@torch.no_grad()
def probabilities(bank, heads, seed, budget, batches=8, samples=4096):
    h, d, k = heads.shape
    flat = heads.permute(1, 0, 2).reshape(d, h*k).double()
    estimates = heads.new_empty((batches, h, k), dtype=torch.float64); audit = []
    for c in range(k):
        budget(); mu = bank['mu'][c]; cov = bank['Q'][c]-torch.outer(mu, mu)
        eigenvalues, vectors = torch.linalg.eigh((cov+cov.T)/2)
        assert float(eigenvalues.min()) >= -1e-8, 'Covariance is not numerically PSD'
        projection = (vectors*eigenvalues.clamp_min(0).sqrt()).T @ flat
        center = mu @ flat
        generator = torch.Generator(device=heads.device).manual_seed(seed+1009*c)
        for batch in range(batches):
            budget()
            noise = torch.randn(samples, d, generator=generator, device=heads.device, dtype=torch.float64)
            scores = (noise @ projection+center).reshape(samples, h, k)
            estimates[batch, :, c] = (scores.argmax(2) == c).double().mean(0)
        audit.append(dict(class_index=c, training_images=bank['n'][c], minimum_covariance_eigenvalue=float(eigenvalues.min()),
                          clipped_negative_eigenvalues=int((eigenvalues < 0).sum()), covariance_trace=float(eigenvalues.sum())))
    return estimates, audit


def choose(probability_batches, meta_accuracy, old, tail, candidates):
    hybrid = probability_batches.clone(); hybrid[:, :, old:] = meta_accuracy[None, :, old:]
    groups = dict(all=list(range(hybrid.shape[2])), old=list(range(old)), current=list(range(old, hybrid.shape[2])))
    if tail:
        groups['tail'] = tail
    stats = {}
    for name, classes in groups.items():
        values = hybrid[:, :, classes].mean(2)
        delta = values-values[:, :1]
        stats[name] = dict(mean_gain=delta.mean(0), integration_se=delta.std(0, unbiased=True)/math.sqrt(len(delta)))
    records = {}
    for index in candidates:
        gains = {name: {key: float(value[index]) for key, value in terms.items()} for name, terms in stats.items()}
        protected = all(gains[name]['mean_gain'] >= -1e-12 for name in groups if name != 'all')
        resolution = max(1e-8, 2*gains['all']['integration_se'])
        records[index] = dict(groups=gains, protection_pass=protected, numerical_resolution=resolution,
                              eligible=protected and gains['all']['mean_gain'] > resolution)
    eligible = [i for i in candidates if records[i]['eligible']]
    selected = min(eligible, key=lambda i: (-records[i]['groups']['all']['mean_gain'], i)) if eligible else 0
    return selected, records


def self_check():
    torch.set_num_threads(2)
    mu = torch.tensor([[1.], [-1.]], dtype=torch.float64)
    bank = dict(mu=mu, Q=torch.tensor([[[1.25]], [[1.25]]], dtype=torch.float64), n=[20, 20])
    heads = torch.tensor([[[1., -1.]], [[-1., 1.]]], dtype=torch.float64)
    p, _ = probabilities(bank, heads, 1803, lambda: None, batches=4, samples=4096)
    expected = .5*(1+math.erf(2/math.sqrt(2)))
    assert abs(float(p[:, 0].mean())-expected) < .01
    assert torch.allclose(p[:, 0]+p[:, 1], torch.ones_like(p[:, 0]))
    selected, rows = choose(p, torch.tensor([[0., .9], [0., .1]]), 1, [0], [0, 1])
    assert selected == 0 and rows[0]['groups']['all']['mean_gain'] == 0
    values = torch.tensor([[[.6,.5],[.7,.5]]]*4, dtype=torch.float64)
    selected, _ = choose(values, torch.tensor([[0.,.5],[0.,.5]]), 1, [0], [0, 1])
    assert selected == 1
    selected, _ = choose(values, torch.tensor([[0.,.5],[0.,.4]]), 1, [0], [0, 1])
    assert selected == 0
    # Numerical Monte Carlo uncertainty is not training-data uncertainty.
    values[:,1,0] = torch.tensor([.5,.75,.5,.75])
    assert choose(values, torch.tensor([[0.,.5],[0.,.5]]), 1, [0], [0,1])[0] == 0
