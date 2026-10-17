"""Fixed-coordinate Gaussian evidence; no cross-class normalization in stable arms."""
import math
import torch


def group_weights(groups, device='cpu'):
    from collections import Counter
    counts = Counter(groups)
    return torch.tensor([1.0 / (len(counts) * counts[g]) for g in groups],
                        dtype=torch.float64, device=device)


def moments(x, weights):
    # Each image carries its group weight, shared equally across its patches.
    x = x.double()
    w = weights.double()[:, None].expand(x.shape[:2]) / x.shape[1]
    flat, w = x.flatten(0, 1), w.flatten()
    return (flat * w[:, None]).sum(0), flat.T @ (flat * w[:, None])


def covariance(mean, second):
    cov = second - mean[:, None] * mean[None, :]
    return (cov + cov.T) / 2


def gaussian(mean, cov, eps):
    cov = cov + eps * torch.eye(len(mean), dtype=cov.dtype, device=cov.device)
    chol = torch.linalg.cholesky(cov)
    return mean, chol, 2 * chol.diag().log().sum()


def logpdf(x, model):
    mean, chol, logdet = model
    z = torch.linalg.solve_triangular(chol, (x.double() - mean).flatten(0, -2).T,
                                      upper=False).T
    return (-0.5 * (z.square().sum(-1) + len(mean) * math.log(2 * math.pi) + logdet)).reshape(x.shape[:-1])


def evidence(logq, logp0, eta):
    return torch.logaddexp(torch.full_like(logq, math.log1p(-eta)),
                          math.log(eta) + logq - logp0[..., None])


def entropy_pool(logq):
    # PatchBELL Appendix A4: per-image/class patch z-score, class softmax, entropy.
    z = (logq - logq.mean(1, keepdim=True)) / (logq.std(1, keepdim=True) + 1e-6)
    q = (5 * z).softmax(-1)
    h = -(q * (q + 1e-12).log()).sum(-1) / (math.log(logq.shape[-1]) + 1e-6)
    return ((1 - h)[..., None] * logq).sum(1)


def scores(patches, cls, stats, reference, cfg):
    d = patches.shape[-1]
    raw, shrunk, global_scores = [], [], []
    for s in stats:
        beta = s['groups'] / (s['groups'] + cfg['tau'])
        for key, target in [('local', raw), ('global', global_scores)]:
            mu, cov = s[key]
            if key == 'global':
                cov = beta * cov + (1 - beta) * reference[key][1]
            target.append(logpdf(patches if key == 'local' else cls,
                                 gaussian(mu, cov, reference[key][2])))
        mu, cov = s['local']
        cov = beta * cov + (1 - beta) * reference['local'][1]
        shrunk.append(logpdf(patches, gaussian(mu, cov, reference['local'][2])))
    raw, shrunk = torch.stack(raw, -1), torch.stack(shrunk, -1)
    p0 = logpdf(patches, gaussian(*reference['local']))
    local = evidence(shrunk, p0, cfg['eta']).mean(1) / d
    glob = torch.stack(global_scores, -1) / d
    return {'uniform': raw.mean(1) / d, 'entropy': entropy_pool(raw) / d,
            'reference': evidence(raw, p0, cfg['eta']).mean(1) / d,
            'reference_shrink': local, 'global': glob,
            'global_local': glob + cfg['local_lambda'] * local}
