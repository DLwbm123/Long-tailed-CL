"""Task-local residual drift moments; no stored examples or generated features."""
import math

import torch


def strength(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Residual drift weight must be finite and nonnegative')
    return float(value)


def config_weight(config):
    weight = strength(config.get('residual_drift_weight', 0.))
    if weight and (config.get('residual_drift_authorized') is not True or
                   config.get('residual_drift_version') != 'residual_drift_v1' or
                   config.get('module') not in ('base', 'local_transport')):
        raise ValueError('Residual drift requires a separate explicit protocol')
    return weight


@torch.no_grad()
def estimate(before, after, weights, shift):
    """E[ee^T], e=after-before-native_shift; do not recenter or normalize its trace."""
    if (before.ndim != 2 or after.shape != before.shape or not len(before) or
            weights.shape != (len(before),) or shift.shape != before.shape[1:] or
            any(v.device != before.device or v.dtype != before.dtype for v in (after, weights, shift)) or
            not all(torch.isfinite(v).all() for v in (before, after, weights, shift)) or
            (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('Invalid paired current-training drift inputs')
    weights = weights / weights.sum()
    residual = after - before - shift
    matrix = residual.T @ (weights[:, None] * residual)
    matrix = (matrix + matrix.T) / 2
    if not torch.isfinite(matrix).all():
        raise ValueError('Nonfinite residual drift moment')
    return matrix, dict(residual_trace=float(matrix.trace()), residual_frobenius_norm=float(matrix.norm()),
                        residual_mean_norm=float((weights @ residual).norm()),
                        current_samples=len(before), historical_samples=0)


def validate(matrix, dim):
    if (matrix.shape != (dim, dim) or not matrix.is_floating_point() or
            not torch.isfinite(matrix).all() or
            not torch.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10)):
        raise ValueError('Invalid residual drift matrix')
    # Check at head preparation/load boundaries, never inside the minibatch solver.
    tolerance = 1e-10 * max(1., float(matrix.abs().max()))
    _, info = torch.linalg.cholesky_ex(matrix + tolerance * torch.eye(dim, dtype=matrix.dtype, device=matrix.device))
    if int(info) != 0:
        raise ValueError('Residual drift matrix must be positive semidefinite')


def add_to_prior(prior, matrix, weight, regularization=.001, *, bias=False):
    """Fold lambda*U into the existing ridge prior so every solve sees it."""
    weight = strength(weight)
    if not weight:
        return prior  # Exact historical path, including its arithmetic and RNG.
    if not math.isfinite(regularization) or regularization <= 0 or matrix is None:
        raise ValueError('Positive ridge scale and residual drift matrix required')
    dim = len(prior) - int(bias)
    validate(matrix, dim)
    if matrix.device != prior.device or matrix.dtype != prior.dtype:
        raise ValueError('Residual drift prior dtype/device mismatch')
    result = prior.clone()
    result[:dim, :dim] += (weight / regularization) * matrix
    return result
