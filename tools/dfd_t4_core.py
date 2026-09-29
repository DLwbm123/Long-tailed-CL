"""DFD-T4-P-v1 fixed feature penalty; no training or data access on import."""
import hashlib
import numpy as np
import torch
from ct3p_core import features


def old_directions(mu):
    mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != (1536, 6) or not np.isfinite(mu).all():
        raise ValueError('BLOCKED_T_BANK')
    matrix = (mu - mu.mean(axis=1, keepdims=True)) / np.sqrt(6)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = max(1e-12, 1e-8 * singular[0])
    rank = int(np.sum(singular > threshold))
    if not 1 <= rank <= 5:
        raise ValueError('BLOCKED_T_RANK')
    return u[:, :rank].copy(), singular, threshold


def random_directions(dataset, seed, rank):
    if dataset not in ('HK', 'ISIC') or seed not in (1993, 1994, 1995) or not 1 <= rank <= 5:
        raise ValueError('BLOCKED_RANDOM_SPEC')
    key = f'DFD-T4-P-v1|random|{dataset}|{seed}|4'.encode('ascii')
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
    rng = np.random.Generator(np.random.PCG64(local_seed))
    q, r = np.linalg.qr(rng.standard_normal((1536, rank)), mode='reduced')
    q *= np.where(np.diag(r) < 0, -1., 1.)
    return q, local_seed


def penalty(delta, q, parallel=10., perpendicular=1.):
    """Coefficients are configurable only for the required algebraic self-checks."""
    return (perpendicular * delta.square().sum(1)
            + (parallel - perpendicular) * (delta @ q).square().sum(1)).mean()


def directional_fd(student, teacher, x, learner, q):
    with torch.no_grad():
        target = features(teacher, x, learner)
    delta = features(student, x, learner) - target
    return penalty(delta, q), delta.detach()


def math_check():
    rng = np.random.Generator(np.random.PCG64(64201))
    mu = rng.normal(size=(1536, 6))
    qd, _, _ = old_directions(mu)
    qr, random_seed = random_directions('ISIC', 1993, qd.shape[1])
    tol = dict(atol=1e-10, rtol=1e-10)
    for q in (qd, qr):
        np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), **tol)
        q32 = q.astype(np.float32)
        assert np.max(np.abs(q32.T @ q32 - np.eye(q.shape[1]))) <= 1e-5
    # Compare projectors, since SVD basis signs need not be identical.
    shifted, _, _ = old_directions(mu + rng.normal(size=(1536, 1)))
    np.testing.assert_allclose(shifted - qd @ (qd.T @ shifted), 0, **tol)
    delta = torch.tensor(rng.normal(size=(7, 1536)), requires_grad=True)
    q = torch.tensor(qd)
    loss = penalty(delta, q)
    projected = (delta @ q) @ q.T
    separate = (10 * projected.square().sum(1) + (delta - projected).square().sum(1)).mean()
    torch.testing.assert_close(loss, separate, **tol)
    grad = torch.autograd.grad(loss, delta)[0]
    torch.testing.assert_close(grad, 2 / len(delta) * (delta + 9 * projected), **tol)
    ordinary = delta.square().sum(1).mean()
    torch.testing.assert_close(penalty(delta, q[:, :0]), ordinary, **tol)
    torch.testing.assert_close(penalty(delta, torch.eye(1536, dtype=delta.dtype)), 10 * ordinary, **tol)
    torch.testing.assert_close(penalty(delta, q, 3., 3.), 3 * ordinary, **tol)
    # The low-rank update has r eigenvalues 9 and d-r eigenvalues zero.
    np.testing.assert_allclose(np.linalg.eigvalsh(qd.T @ qd), np.linalg.eigvalsh(qr.T @ qr), **tol)
    np.testing.assert_array_equal(qr, random_directions('ISIC', 1993, qd.shape[1])[0])
    assert penalty(torch.zeros_like(delta), q).item() == 0
    for invalid in (np.zeros((1536, 6)), np.full((1536, 6), np.nan), np.ones((1536, 5))):
        try:
            old_directions(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid bank accepted')
    return dict(status='PASS', scope='CPU_SYNTHETIC_ONLY', rank=qd.shape[1],
                random_seed=random_seed, atol=1e-10, rtol=1e-10,
                checks=['orthogonality_float64_and_float32', 'equivalent_loss_forms',
                        'empty_Q_beta1', 'full_Q_beta10', 'equal_coefficients',
                        'analytic_gradient', 'same_feature_metric_spectrum',
                        'translation_invariant_span', 'deterministic_local_random',
                        'zero_initial_delta', 'invalid_bank_rejection'],
                real_model_regression='NOT_RUN', optimizer_steps=0)


if __name__ == '__main__':
    import json
    print(json.dumps(math_check(), indent=2))
