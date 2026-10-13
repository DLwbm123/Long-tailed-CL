"""Boundary-only readout repair; frozen evaluation uses no training feedback."""
import math

import torch
import core1_competition as competition

VERSION = 'old_pair_residual_guard_v1'


def validate_design(p):
    if (p['version'] != VERSION or p['status'] != 'CPU_DESIGN_ONLY'
            or p['GPU_launch_authorized'] or p['production_integrated']
            or p['optimizer_updates'] != 0 or p['test_accessed']
            or p['medical_data_access_authorized'] or p['checkpoint_access_authorized']
            or p['future_gpus'] != [0, 1] or p['seed'] != 74002
            or p['relative_pair_curvature'] != 1. or p['native_energy_radius'] != .1
            or p['pair_target'] != 1. or p['training_feedback']
            or p['primary_control'] != 'local_transport'):
        raise ValueError('Independent CPU design scope required')


def _finite_double(*values):
    if any(x.dtype != torch.float64 or x.device != values[0].device
           or not torch.isfinite(x).all() for x in values):
        raise ValueError('Finite, same-device float64 inputs required')


def contrast_basis(count, like):
    """Deterministic Helmert basis, with no preferred old class."""
    v = like.new_zeros((count, max(0, count-1)))
    for j in range(1, count):
        v[:j, j-1] = 1 / math.sqrt(j*(j+1))
        v[j, j-1] = -j / math.sqrt(j*(j+1))
    return v


def native_operator(bank, pairs, R, W):
    k = len(bank['n'])
    q = bank['Q'] / k
    lap, _ = competition.laplacians(pairs)
    return (q.sum(0)+.001*R) @ W + .5*torch.einsum('cdk,ckl->dl', q @ W, lap)


def geometry(bank, pairs, R, old_count):
    mu, Q = bank['mu'], bank['Q']; k, d = mu.shape
    _finite_double(mu, Q, pairs, R)
    if (isinstance(old_count, bool) or not isinstance(old_count, int) or not 0 <= old_count <= k
            or k < 2 or d < 1 or Q.shape != (k, d, d) or R.shape != (d, d) or pairs.shape != (k, k)
            or len(bank['n']) != k or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in bank['n'])
            or (pairs < 0).any() or pairs.diag().any()
            or not torch.allclose(pairs.sum(1), pairs.new_ones(k), atol=1e-8, rtol=1e-8)
            or not torch.allclose(R, R.T, atol=1e-10, rtol=1e-10)
            or not torch.allclose(Q, Q.transpose(1, 2), atol=1e-10, rtol=1e-10)):
        raise ValueError('Invalid frozen bank, competition, or old-class prefix')
    V = mu.new_zeros((k, max(0, old_count-1)))
    V[:old_count] = contrast_basis(old_count, mu)
    if old_count < 2:
        return mu.new_empty((d, 0)), V, mu.new_empty((0, 0)), mu.new_empty((0, 0)), mu.new_empty(0), []
    # No projection against local_transport: residual targets already condition on its head.
    centers = (mu[:old_count]-mu[:old_count].mean(0)).T
    B, singular, _ = torch.linalg.svd(centers, full_matrices=False)
    tol = torch.finfo(mu.dtype).eps * max(centers.shape) * float(singular.max())
    B = B[:, singular > tol]; r, s = B.shape[1], V.shape[1]
    if not r:
        return B, V, mu.new_empty((0, 0)), mu.new_empty((0, 0)), mu.new_empty(0), []
    qr = (B.T @ Q @ B) / k
    A = qr.sum(0)+.001*(B.T @ R @ B)
    lap, _ = competition.laplacians(pairs)
    H = torch.kron(A.contiguous(), torch.eye(s, dtype=mu.dtype, device=mu.device))
    for c in range(k):
        H += .5*torch.kron(qr[c].contiguous(), (V.T @ lap[c] @ V).contiguous())
    H = (H+H.T)/2
    edges = [(c, j) for c in range(old_count) for j in range(old_count) if c != j]
    rows = [torch.outer(mu[c] @ B, V[c]-V[j]).flatten() for c, j in edges]
    # One pseudo-count is a fixed reliability shrinkage, not an old-drift confidence estimate.
    reliability = mu.new_tensor([bank['n'][c]/(bank['n'][c]+1) for c, _ in edges])
    omega = reliability / (old_count*(old_count-1))
    return B, V, H, torch.stack(rows), omega, edges


def quadratic_step(H, L, omega, residual, energy_budget, normalized=True):
    """Closed quadratic solution, with one scalar trust-region multiplier."""
    _finite_double(H, L, omega, residual)
    if (not isinstance(normalized, bool) or not math.isfinite(energy_budget) or energy_budget < 0
            or H.shape != (L.shape[1], L.shape[1]) or omega.shape != residual.shape
            or residual.shape != (len(L),) or (omega <= 0).any() or (residual < 0).any()):
        raise ValueError('Invalid pair system or native-energy budget')
    C = torch.linalg.cholesky(H)
    D = torch.linalg.solve_triangular(C, L.T, upper=False).T
    G = D.T @ (omega[:, None]*D); G = (G+G.T)/2
    eigen, U = torch.linalg.eigh(G); eigen = eigen.clamp_min(0)
    radius = float(eigen.max())
    if radius <= 0 or not residual.any() or not energy_budget:
        return H.new_zeros(len(H)), dict(relative_residual=0., multiplier=0., effective_curvature=0., trust_active=False, step_energy=0.)
    scale = 1/radius if normalized else 1.
    b = scale*(D.T @ (omega*residual)); rotated = U.T @ b
    def at(dual):
        return U @ (rotated/(1+scale*eigen+dual))
    v = at(0.); dual = 0.; active = float(v.square().sum()) > energy_budget
    if active:
        lo, hi = 0., 1.
        while float(at(hi).square().sum()) > energy_budget:
            hi *= 2
            if not math.isfinite(hi):
                raise ArithmeticError('Trust-region bracketing failed')
        for _ in range(64):
            mid = (lo+hi)/2
            if float(at(mid).square().sum()) > energy_budget:lo = mid
            else:hi = mid
        dual = hi; v = at(dual)
    z = torch.linalg.solve_triangular(C.T, v[:, None], upper=True).flatten()
    rhs = scale*(L.T @ (omega*residual))
    gradient = (1+dual)*(H @ z)+scale*(L.T @ (omega*(L @ z)))-rhs
    error = float(gradient.norm()/rhs.norm().clamp_min(1e-30))
    step_energy = float(z @ H @ z)
    if not torch.isfinite(z).all() or error > 1e-8 or step_energy > energy_budget*(1+1e-8)+1e-18:
        raise ArithmeticError('Repair KKT or trust-region check failed')
    return z, dict(relative_residual=error, multiplier=dual, effective_curvature=scale*radius,
        curvature_scale=scale, unscaled_curvature=radius, trust_active=active, step_energy=step_energy)


@torch.no_grad()
def fit(bank, pairs, R, reference, old_count, *, normalized=True):
    """Reference is the unmodified native double head, with exactly its saved pairs/R."""
    _finite_double(reference, bank['mu'])
    if reference.shape != bank['mu'].T.shape or not isinstance(normalized, bool):
        raise ValueError('Reference or ablation scope mismatch')
    B, V, H, L, omega, edges = geometry(bank, pairs, R, old_count)
    _, targets = competition.laplacians(pairs)
    rhs = bank['mu'].T/len(pairs)+.5*(bank['mu'].T/len(pairs)) @ targets
    applied = native_operator(bank, pairs, R, reference)
    native_error = float((applied-rhs).norm()/rhs.norm().clamp_min(1e-30))
    if native_error > 1e-8:
        raise ValueError('Reference is not the exact native boundary head')
    delta = torch.zeros_like(reference)
    audit = dict(version=VERSION, old_count=old_count, feature_rank=B.shape[1],
        coefficients=H.shape[0], native_residual=native_error, optimizer_updates=0,
        normalized=normalized, new_columns_exact_zero=True, old_drift_estimated=False)
    if not len(H):
        return delta, dict(audit, exact_identity=True, reason='fewer than two old classes or zero old contrast rank')
    residual = reference.new_tensor([max(0., 1-float(bank['mu'][c] @ (reference[:, c]-reference[:, j]))) for c, j in edges])
    energy = float((reference*applied).sum())
    if not math.isfinite(energy) or energy < 0:
        raise ValueError('Invalid native reference energy')
    z, checked = quadratic_step(H, L, omega, residual, .1**2*energy, normalized)
    delta = B @ z.reshape(B.shape[1], V.shape[1]) @ V.T
    before = float((omega*residual.square()).sum())
    after = float((omega*(L @ z-residual).square()).sum())
    if after > before+1e-10*max(1., before) or delta[:, old_count:].any():
        raise ArithmeticError('Pair residual or new-column invariant failed')
    return delta, dict(audit, **checked, exact_identity=not bool(delta.any()),
        pair_deficit_before=before, pair_deficit_after=after,
        relative_native_energy_step=math.sqrt(checked['step_energy']/max(energy, 1e-30)),
        delta_head_norm=float(delta.norm()), historical_samples_read=0)


def guard(base_scores, delta_scores, old_count):
    """Preserve all new logits and the old maximum for every input, without labels."""
    if (base_scores.ndim != 2 or delta_scores.shape != base_scores.shape
            or delta_scores.dtype != base_scores.dtype or delta_scores.device != base_scores.device
            or base_scores.dtype not in (torch.float32, torch.float64)
            or isinstance(old_count, bool) or not isinstance(old_count, int)
            or not 0 <= old_count <= base_scores.shape[1]
            or not torch.isfinite(base_scores).all() or not torch.isfinite(delta_scores).all()
            or delta_scores[:, old_count:].any()):
        raise ValueError('Invalid old-only readout correction')
    result = base_scores.clone()
    if old_count:
        raw = base_scores[:, :old_count]+delta_scores[:, :old_count]
        old_max = base_scores[:, :old_count].max(1, keepdim=True).values
        result[:, :old_count] = old_max-(raw.max(1, keepdim=True).values-raw)
        unchanged = ~delta_scores[:, :old_count].any(1)
        result[unchanged] = base_scores[unchanged]
        if not torch.equal(result[:, :old_count].max(1).values, old_max.flatten()):
            raise ArithmeticError('Old maximum changed')
    if not torch.isfinite(result).all():raise ValueError('Nonfinite guarded readout')
    return result


def checkpoint(delta, old_count, metadata):
    required = {'dataset', 'control', 'control_source', 'order', 'seen', 'seed', 'task', 'source'}
    if (set(metadata) != required or metadata['dataset'] not in ('ISIC', 'HK')
            or metadata['control'] not in ('base', 'local_transport') or metadata['seed'] != 74002
            or metadata['seen'] != metadata['order'][:len(metadata['seen'])]
            or len(metadata['seen']) != delta.shape[1] or not isinstance(old_count, int)
            or isinstance(old_count, bool) or not 0 <= old_count <= delta.shape[1]
            or not torch.isfinite(delta).all() or delta[:, old_count:].any()):
        raise ValueError('Invalid sidecar provenance or correction')
    return dict(version=VERSION, old_count=old_count, metadata=dict(metadata), delta=delta.detach().cpu().float())


def restore(state, metadata, old_count, shape):
    if (state.get('version') != VERSION or state.get('metadata') != metadata
            or state.get('old_count') != old_count or state['delta'].shape != shape):
        raise ValueError('Wrong correction version, provenance, task, or dimensions')
    checkpoint(state['delta'], old_count, metadata)
    return state['delta'].clone()
