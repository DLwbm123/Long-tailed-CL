"""Class-balanced ridge and prototype-conditioned transport; no historical examples."""
import numpy as np


def empty(dim):
    return dict(S=np.zeros((dim, dim), dtype=np.float64),
                mu=np.empty((0, dim), dtype=np.float64), n=np.empty(0, dtype=np.int64))


def append(bank, x, y, classes):
    x = np.asarray(x, dtype=np.float64)
    if len(classes) == 0 or not np.isfinite(x).all():
        raise ValueError('Empty classes or nonfinite features')
    means, counts = [], []
    S = bank['S'].copy()
    for c in classes:
        z = x[y == c]
        if not len(z):
            raise ValueError('A class has no training observations')
        means.append(z.mean(0)); counts.append(len(z))
        S += z.T @ z / len(z)
    return dict(S=S, mu=np.vstack([bank['mu'], means]), n=np.r_[bank['n'], counts])


def ridge(bank, regularization=.001):
    C = len(bank['n'])
    A = bank['S'] / C + regularization * np.eye(bank['S'].shape[0])
    B = bank['mu'].T / C
    W = np.linalg.solve(A, B)
    residual = float(np.linalg.norm(A @ W - B) / max(np.linalg.norm(B), 1e-15))
    if not np.isfinite(W).all() or residual > 1e-7:
        raise ValueError('Ridge solve failed')
    return W, residual


def move(bank, shifts):
    """Exact raw moment translation, one distinct displacement per old class."""
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.shape != bank['mu'].shape or not np.isfinite(shifts).all():
        raise ValueError('Invalid prototype shifts')
    mu = bank['mu']
    return dict(S=bank['S'] + mu.T @ shifts + shifts.T @ mu + shifts.T @ shifts,
                mu=mu + shifts, n=bank['n'].copy())


def transport(bank, before, after, y, classes, local=False):
    means = np.stack([before[y == c].mean(0) for c in classes]).astype(np.float64)
    deltas = np.stack([(after[y == c] - before[y == c]).mean(0) for c in classes])
    global_shift = deltas.mean(0)
    shifts = np.tile(global_shift, (len(bank['n']), 1))
    if local and len(shifts):
        def unit(x):
            return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        similarity = unit(bank['mu']) @ unit(means).T
        logits = similarity / .1
        weights = np.exp(logits - logits.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        # ponytail: one prototype per class cannot capture multimodal class drift.
        shifts = .5 * (weights @ deltas) + .5 * shifts
    return move(bank, shifts), dict(global_shift_norm=float(np.linalg.norm(global_shift)),
        mean_old_shift_norm=float(np.linalg.norm(shifts, axis=1).mean()) if len(shifts) else 0.,
        observed_class_shift_norms=np.linalg.norm(deltas, axis=1).tolist())
