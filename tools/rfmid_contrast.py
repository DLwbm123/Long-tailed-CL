"""Four-column binary contrasts; production use requires a versioned RFMiD protocol."""
import numpy as np


def columns(state, label, *, remove_shared=True):
    n, means = np.asarray(state['n']), np.asarray(state['means'], dtype=np.float64)
    if (means.ndim != 4 or means.shape[1:3] != (2, 4) or means.shape[-1] < 1 or
            n.shape != means.shape[:2] or not np.issubdtype(n.dtype, np.integer) or
            (n < 0).any() or not isinstance(label, (int, np.integer)) or
            not 0 <= label < len(n) or not isinstance(remove_shared, bool) or
            not np.isfinite(means[label]).all()):
        raise ValueError('Require finite four-region binary means and nonnegative integer counts')
    d = means.shape[-1]
    audit = dict(both_signs_observed=bool((n[label] > 0).all()), shared_rank=0,
                 raw_energy=0., retained_energy=0., retained_fraction=0., shared_leakage=0.)
    if not audit['both_signs_observed']:
        return np.zeros((d, 4)), audit
    negative, positive = means[label]
    delta = (positive - negative).T
    shared = ((positive + negative) / 2).T
    scale = np.maximum(np.linalg.norm(positive, axis=1) + np.linalg.norm(negative, axis=1), 1e-12)
    if not all(np.isfinite(x).all() for x in (delta, shared, scale)):
        raise ValueError('Nonfinite contrast arithmetic')
    basis, singular, _ = np.linalg.svd(shared, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(shared.shape) * singular[0]
    basis = basis[:, singular > tolerance]
    raw = delta / scale[None, :] / 2
    P = raw.copy()
    if remove_shared:
        # A second pass removes floating-point leakage, not another scientific constraint.
        for _ in range(2):
            P -= basis @ (basis.T @ P)
    raw_energy, retained = float(np.sum(raw * raw)), float(np.sum(P * P))
    if not np.isfinite(P).all() or retained > 1 + 1e-12:
        raise ArithmeticError('Contrast prior exceeds its fixed energy bound')
    audit.update(shared_rank=basis.shape[1], raw_energy=raw_energy, retained_energy=retained,
                 retained_fraction=retained / raw_energy if raw_energy else 0.,
                 shared_leakage=float(np.linalg.norm(basis.T @ P)))
    return P, audit


def penalty(state, label, *, remove_shared=True):
    P, _ = columns(state, label, remove_shared=remove_shared)
    result = np.eye(P.shape[0] + 1)
    metric = result[:-1, :-1] - P @ np.linalg.solve(np.eye(4) + P.T @ P, P.T)
    result[:-1, :-1] = (metric + metric.T) / 2
    return result
