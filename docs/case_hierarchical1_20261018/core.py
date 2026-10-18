"""Patient random-intercept Gaussian with mergeable class statistics (R=I)."""
import math
import torch


def sym(x):
    return (x + x.T) / 2


def psd(x):
    values, vectors = torch.linalg.eigh(sym(x))
    return sym((vectors * values.clamp_min(0)) @ vectors.T), int((values < 0).sum())


def accumulator(d, device):
    return dict(n=0, mean=torch.zeros(d, dtype=torch.float64, device=device),
                mean_second=torch.zeros(d, d, dtype=torch.float64, device=device),
                within=torch.zeros(d, d, dtype=torch.float64, device=device),
                second=torch.zeros(d, d, dtype=torch.float64, device=device), inv_count=0.)


def add_patient(acc, count, total, second):
    assert count > 1
    mean = total / count
    acc['n'] += 1
    acc['mean'] += mean
    acc['mean_second'] += torch.outer(mean, mean)
    acc['within'] += sym((second - count * torch.outer(mean, mean)) / (count - 1))
    acc['second'] += second / count
    acc['inv_count'] += 1 / count


def finish_accumulator(acc):
    n = acc['n']
    assert n > 0
    mu, w = acc['mean'] / n, acc['within'] / n
    between_mean = sym((acc['mean_second'] - n * torch.outer(mu, mu)) / max(n - 1, 1))
    b, clipped = psd(between_mean - w * (acc['inv_count'] / n)) if n > 1 else (w * 0, 0)
    pooled = sym(acc['second'] / n - torch.outer(mu, mu))
    return dict(mu=mu, W=sym(w), B=b, pooled=pooled, patients=n,
                clipped_eigenvalues=clipped, between_identified=n > 1)


def finish(patients):
    """Input: (patch count, patch sum, patch second moment) per patient.

    Patients carry equal mass regardless of their number of images. W is the
    mean unbiased within-patient covariance. Mean covariance subtracts W E[1/T].
    This moment estimator assumes a common W within each class; it is not REML.
    """
    acc = accumulator(len(patients[0][1]), patients[0][1].device)
    for count, total, second in patients:
        add_patient(acc, count, total, second)
    return finish_accumulator(acc)


def gaussian(mu, covariance):
    chol = torch.linalg.cholesky(sym(covariance))
    return mu, chol, 2 * chol.diag().log().sum()


def logpdf(x, model):
    mu, chol, ld = model
    delta = (x.double() - mu).reshape(-1, len(mu)).T
    z = torch.linalg.solve_triangular(chol, delta, upper=False).T
    return (-.5 * (z.square().sum(-1) + ld + len(mu) * math.log(2 * math.pi))).reshape(x.shape[:-1])


def joint(x, within_model, mean_model):
    """Exact bag log density in orthonormal mean/residual coordinates.

    Cov(vec X) = 11' tensor B + I tensor W. The mean coordinate is sqrt(P)*m
    with covariance W+P*B. Its Jacobian contributes -d/2 log(P), common to
    classes here but retained for the dense-Gaussian identity check.
    """
    p, d = x.shape[-2:]
    mean = x.double().mean(-2)
    residual = x.double() - mean[..., None, :]
    _, chol, logdet = within_model
    z = torch.linalg.solve_triangular(chol, residual.reshape(-1, d).T, upper=False)
    quadratic = z.T.square().sum(-1).reshape(x.shape[:-1]).sum(-1)
    return (logpdf(mean, mean_model) - d / 2 * math.log(p)
            - .5 * (quadratic + (p - 1) * (logdet + d * math.log(2 * math.pi))))


def entropy_pool(raw):
    centered = raw - raw.mean(1, keepdim=True)
    sd = raw.std(1, keepdim=True)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    q = (5 * centered / (sd + 1e-6)).softmax(-1)
    h = -(q * q.clamp_min(1e-6).log()).sum(-1) / (math.log(raw.shape[-1]) + 1e-6)
    return ((1 - h)[..., None] * raw).sum(1)


def models(stats, reference, p, tau, eps):
    result = []
    eye = torch.eye(len(stats[0]['mu']), dtype=torch.float64, device=stats[0]['mu'].device)
    for s in stats:
        beta = s['patients'] / (s['patients'] + tau)
        w = beta * s['W'] + (1 - beta) * reference['W'] + eps * eye
        # A single patient cannot identify between-patient variance.
        b = beta * s['B'] + (1 - beta) * reference['B'] if s['between_identified'] else reference['B']
        cov = beta * s['pooled'] + (1 - beta) * reference['pooled'] + eps * eye
        mu = s['mu']
        result.append(dict(image=gaussian(s['image_mu'], s['image_cov'] + eps * eye),
                           patient=gaussian(mu, s['pooled'] + eps * eye),
                           shrunk=gaussian(mu, cov), within=gaussian(mu, w),
                           marginal=gaussian(mu, w + b), mean=gaussian(mu, b + w / p)))
    return result


def scores(x, fitted):
    image = torch.stack([logpdf(x, m['image']) for m in fitted], -1)
    out = dict(image_uniform=image.sum(1), patchbell_adapted_entropy=entropy_pool(image))
    for arm, key in [('patient_uniform', 'patient'), ('patient_shrunk', 'shrunk'),
                     ('hierarchical_no_shared', 'within'), ('hierarchical_independent', 'marginal')]:
        out[arm] = torch.stack([logpdf(x, m[key]).sum(1) for m in fitted], -1)
    out['hierarchical'] = torch.stack([joint(x, m['within'], m['mean']) for m in fitted], -1)
    assert all(torch.isfinite(v).all() for v in out.values())
    return out
