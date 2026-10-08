"""Fit-only affine drift; exact class moments conditional on the fitted map."""
import math
import torch


@torch.no_grad()
def fit(before, after, labels, regularization=.001):
    if before.ndim != 2 or before.shape != after.shape or labels.ndim != 1 or len(labels) != len(before) or not len(before):
        raise ValueError('Invalid paired drift observations')
    if not math.isfinite(regularization) or regularization <= 0 or not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('Invalid affine drift inputs')
    classes = labels.unique()
    weights = before.new_empty(len(before))
    for c in classes:
        selected = labels == c
        weights[selected] = 1./(len(classes)*int(selected.sum()))
    mean = (weights[:, None]*before).sum(0)
    shift = (weights[:, None]*(after-before)).sum(0)
    x = (before-mean)*weights.sqrt()[:, None]
    delta = (after-before-shift)*weights.sqrt()[:, None]
    dimension = before.shape[1]
    if len(x) <= dimension:
        system = x @ x.T + regularization*torch.eye(len(x), device=x.device, dtype=x.dtype)
        solution = torch.linalg.solve(system, delta)
        residual = (system @ solution-delta).norm()/delta.norm().clamp_min(1e-15)
        change = x.T @ solution
    else:
        system = x.T @ x + regularization*torch.eye(dimension, device=x.device, dtype=x.dtype)
        rhs = x.T @ delta
        change = torch.linalg.solve(system, rhs)
        residual = (system @ change-rhs).norm()/rhs.norm().clamp_min(1e-15)
    matrix = torch.eye(dimension, device=x.device, dtype=x.dtype)+change.T
    offset = shift-change.T @ mean
    error = before @ matrix.T+offset-after
    if not torch.isfinite(matrix).all() or not torch.isfinite(offset).all() or residual > 1e-8:
        raise ValueError('Invalid affine drift solve')
    return dict(matrix=matrix, offset=offset, diagnostics=dict(
        kind='affine', regularization=regularization, observations=len(before), classes=len(classes),
        relative_residual=float(residual), identity_change_norm=float(change.norm()),
        translation_mse=float(delta.square().sum()), fitted_mse=float((weights[:, None]*error.square()).sum())))


@torch.no_grad()
def transport(bank, mapping):
    a, b = mapping['matrix'], mapping['offset']
    dimension = bank['mu'].shape[1]
    if a.shape != (dimension, dimension) or b.shape != (dimension,) or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid affine moment transform')
    center = bank['mu'] @ a.T
    second = a @ bank['Q'] @ a.T
    second = second + center[:, :, None]*b[None, None, :] + b[None, :, None]*center[:, None, :]
    second = second + torch.outer(b, b)[None]
    # ponytail: old component radii stay unused; store their covariance before enabling radius-based historical readout.
    components = [dict(c, center=a @ c['center']+b, radius_status='unchanged_unused') for c in bank['components']]
    return dict(mu=center+b, Q=(second+second.transpose(1, 2))/2, n=list(bank['n']), components=components)
