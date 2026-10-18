"""DENO Sec. 2.2--2.3 pooling; fixed-parameter transfer to Gaussian evidence.

Source: https://openreview.net/forum?id=cVzyEuOJ9y (anonymous submission).
No author implementation was used. Input evidence is already scaled/clipped.
"""
import torch


def kernel(x, bandwidth):
    x = torch.nn.functional.normalize(x.double(), dim=-1)
    distances = torch.cdist(x, x).square()
    similarity = (-distances / (2 * bandwidth**2)).exp()
    degrees = similarity.sum(-1)
    normalized = similarity / (degrees[..., :, None] * degrees[..., None, :]).sqrt()
    return (normalized + normalized.transpose(-1, -2)) / 2


def determinant_evidence(k, e):
    """k: B,P,P; e: B,P,C. Two SPD Cholesky logdets per class."""
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
    out = []
    for c in range(e.shape[-1]):
        v = e[..., c]
        logs = []
        for sign in [1, -1]:
            d = (sign * v / 4).exp()
            matrix = eye + d[..., :, None] * k * d[..., None, :]
            # I+DKD is SPD with minimum eigenvalue >=1 in exact arithmetic.
            chol, info = torch.linalg.cholesky_ex(matrix)
            if bool(info.any()):
                raise ArithmeticError('DENO Cholesky failure; no adaptive scientific retry')
            logs.append(2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        out.append(2 * (logs[0] - logs[1]))
    result = torch.stack(out, -1)
    assert torch.isfinite(result).all()
    return result


def pool(x, raw, reference, scale, bandwidth, clip, beta, tau):
    raw_e = (raw - reference[..., None]) / scale
    e = raw_e.clamp(-clip, clip)
    k = kernel(x, bandwidth)
    determinant = determinant_evidence(k, e)
    mean = e.mean(1)
    return dict(reference_mean_raw=raw_e.mean(1), reference_mean_clipped=mean,
                deno_determinant=determinant / tau,
                deno_mixture=beta * determinant / tau + (1-beta) * mean,
                identity_mixture=beta * e.sum(1) / tau + (1-beta) * mean), raw_e
