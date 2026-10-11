"""Four-region observed binary-label aggregates and inverse ridge prior."""
import numpy as np


def empty_local(classes, dim):
    return dict(n=np.zeros((classes, 2), dtype=np.int64),
                means=np.zeros((classes, 2, 4, dim), dtype=np.float64))


def add_local(state, parts, y, mask):
    parts, y, mask = np.asarray(parts, dtype=np.float64), np.asarray(y), np.asarray(mask)
    c, _, regions, d = state['means'].shape
    if regions != 4 or parts.shape != (len(y), 4, d) or y.shape != (len(parts), c) or mask.shape != y.shape:
        raise ValueError('Local aggregate shape differs')
    if mask.dtype != np.bool_ or not np.isfinite(parts).all() or not np.isin(y[mask], (0, 1)).all():
        raise ValueError('Local aggregates require finite regions and observed binary targets')
    result = dict(n=state['n'].copy(), means=state['means'].copy())
    for label in range(c):
        for value in (0, 1):
            x = parts[mask[:, label] & (y[:, label] == value)]
            if len(x):
                n = state['n'][label, value]
                result['means'][label, value] = (n * state['means'][label, value] + x.sum(0)) / (n + len(x))
                result['n'][label, value] += len(x)
    return result


def transport_local(state, before, after, y, mask, weights):
    from ml_statistics import translation
    if before.shape != after.shape or before.shape[1:] != state['means'].shape[2:]:
        raise ValueError('Regional before/after alignment differs')
    shifts = np.stack([translation(before[:, r], after[:, r], y, mask, weights)[0] for r in range(4)])
    means = state['means'] + shifts[None, None] * (state['n'] > 0)[..., None, None]
    return dict(n=state['n'].copy(), means=means), dict(regional_shift_norm=float(np.linalg.norm(shifts)))


def penalty(state, label):
    centers = state['means'][label, state['n'][label] > 0].reshape(-1, state['means'].shape[-1])
    d = state['means'].shape[-1]
    if not np.isfinite(centers).all():
        raise ValueError('Nonfinite local binary centers')
    P = (centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12) / 2).T
    eye = np.eye(d)
    metric = eye - P @ np.linalg.solve(np.eye(P.shape[1]) + P.T @ P, P.T)
    metric = (metric + metric.T) / 2
    result = np.eye(d + 1)
    result[:-1, :-1] = metric
    return result
