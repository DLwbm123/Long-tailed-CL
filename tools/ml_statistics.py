"""Per-label binary sufficient statistics, translation, and fixed ridge heads.

Only counts, feature sums and second moments persist across tasks. Multi-label
co-occurrence is retained in each label's feature moments; labels are not treated
as mutually exclusive classes. Targets are signed (-1,+1), making zero the score
decision boundary. Sigmoid scores are not claimed to be calibrated probabilities.
"""
import numpy as np


def empty(dim, classes):
    return dict(n=np.zeros((classes, 2), dtype=np.int64),
                s=np.zeros((classes, 2, dim), dtype=np.float64),
                G=np.zeros((classes, 2, dim, dim), dtype=np.float64))


def add(state, z, y, mask):
    z, y, mask = np.asarray(z, dtype=np.float64), np.asarray(y), np.asarray(mask)
    c, _, d = state['s'].shape
    if z.shape != (len(y), d) or y.shape != mask.shape or y.shape != (len(z), c):
        raise ValueError('Feature, label, or mask shape mismatch')
    if mask.dtype != np.bool_ or not np.isfinite(z).all():
        raise ValueError('Require a boolean mask and finite features')
    if not np.isin(y[mask], [0, 1]).all():
        raise ValueError('Observed targets must be binary')
    for label in range(c):
        for value in (0, 1):
            x = z[mask[:, label] & (y[:, label] == value)]
            state['n'][label, value] += len(x)
            state['s'][label, value] += x.sum(0)
            state['G'][label, value] += x.T @ x
    return state


def translated(state, shift):
    b = np.asarray(shift, dtype=np.float64)
    if b.shape != state['s'].shape[-1:] or not np.isfinite(b).all():
        raise ValueError('Invalid feature translation')
    s, n = state['s'], state['n']
    return dict(n=n.copy(), s=s + n[..., None] * b,
                G=state['G'] + s[..., :, None] * b + b[:, None] * s[..., None, :]
                + n[..., None, None] * np.outer(b, b))


def translation(before, after, y, mask, weights):
    """Current observed-pair weights; no history or future label frequencies."""
    before, after = np.asarray(before), np.asarray(after)
    y, mask, weights = np.asarray(y), np.asarray(mask), np.asarray(weights)
    if before.shape != after.shape or len(before) != len(y):
        raise ValueError('Before/after current-feature alignment mismatch')
    safe = np.where(mask, y, 0).astype(np.int64)
    w = weights[np.arange(y.shape[1])[None, :], safe] * mask
    row_w = w.sum(1) / np.maximum(mask.sum(1), 1)
    if not np.isfinite(after).all() or not row_w.sum() > 0:
        raise ValueError('No finite observed data for translation')
    row_w /= row_w.sum()
    delta = after - before
    shift = row_w @ delta
    return shift, dict(shift_norm=float(np.linalg.norm(shift)),
                       residual_mean=float(row_w @ ((delta - shift) ** 2).mean(1)))


def ridge(state, regularization=.001):
    """Equal mass for available positive/negative groups within each label."""
    if regularization <= 0:
        raise ValueError('Positive ridge regularization required')
    c, _, d = state['s'].shape
    head = np.zeros((d + 1, c))
    residuals = []
    for label in range(c):
        groups = np.flatnonzero(state['n'][label] > 0)
        if not len(groups):
            residuals.append(None)
            continue
        G, rhs = np.zeros((d + 1, d + 1)), np.zeros(d + 1)
        for value in groups:
            n, s, moment = state['n'][label, value], state['s'][label, value], state['G'][label, value]
            gram = np.block([[moment, s[:, None]], [s[None, :], np.array([[n]])]])
            G += gram / (len(groups) * n)
            rhs += np.r_[s, n] * (2 * value - 1) / (len(groups) * n)
        system = G + regularization * np.eye(d + 1)
        head[:, label] = np.linalg.solve(system, rhs)
        residual = float(np.linalg.norm(system @ head[:, label] - rhs) / max(np.linalg.norm(rhs), 1e-30))
        if not np.isfinite(head[:, label]).all() or residual > 1e-8:
            raise ArithmeticError('Binary analytic head solve failed')
        residuals.append(residual)
    return head, dict(residuals=residuals, counts=state['n'].tolist(),
                      active_labels=(state['n'].sum(1) > 0).tolist(),
                      both_signs_observed=(state['n'] > 0).all(1).tolist(),
                      target='signed_binary', bias_regularized=True, regularization=regularization)
