"""Design-only LCA-inspired old-margin prior; no production worker imports this file."""
import math

import torch
import core1_competition as competition

VERSION = 'supported_old_margin_v1'


def config_enabled(config):
    enabled = config.get('lca_supported', False)
    if not isinstance(enabled, bool):
        raise ValueError('LCA authorization must be boolean')
    if enabled and (config.get('lca_version') != VERSION or
        config.get('lca_authorized') is not True or config.get('module') not in ('base','local_transport')
        or config.get('name') not in ('ISIC','HK') or config.get('method') != 'static_pc'
        or config.get('evaluation_split') != 'development_validation'
        or config.get('lca_map_ridge') != .001 or config.get('lca_curvature_cap') != .1
        or config.get('residual_drift_weight',0) != 0 or config.get('checkpoint_root')):
        raise ValueError('Independent frozen main-four-grid LCA protocol required')
    return enabled


@torch.no_grad()
def prepare(before, after, weights, shift, task_start_bank, candidate, arm, dataset, seen, local, R, proximal=0.):
    import module_additions as modules
    U,a=estimate(before,after,weights,shift,task_start_bank['mu'],task_start_bank['Q'])
    raw=U
    if arm=='local_transport':
        R0,_=modules.metric(candidate,'base',dataset,seen)
        U,d=remove_local_prior_span(U,R0,R)
    else:
        d=dict(local_relaxation_rank=0,retained_risk_trace=float(U.trace()),raw_risk_trace=float(U.trace()),overlap_leakage=0.)
    H=candidate['Q'].sum(0)/len(candidate['n'])+.001*R+proximal*torch.eye(len(R),dtype=R.dtype,device=R.device)
    S,b=bounded_penalty(U,H,raw_risk=raw)
    M=old_margin_metric(len(seen),len(task_start_bank['n']),before)
    support=[v for v in a.pop('old_supported_energy_fractions') if v is not None]
    a.update(old_support_min=min(support) if support else 0.,old_support_mean=sum(support)/len(support) if support else 0.,
        old_support_max=max(support) if support else 0.,old_support_defined_classes=len(support),**d,**b,
        output_margin_rank=max(0,len(task_start_bank['n'])-1),new_columns_direct_force_zero=True,
        detached=True,version=VERSION,estimate_before_epoch=True)
    return S,M,a


def bank_head(bank,pairs,regularizer,S,M):
    k=len(bank['n'])
    return solve(bank['Q']/k,bank['mu']/k,bank['mu'].new_full((k,),1/k),pairs,regularizer,S,M)


def psd(value):
    if (value.ndim != 2 or value.shape[0] != value.shape[1] or value.dtype != torch.float64
            or not torch.isfinite(value).all() or not torch.allclose(value, value.T, atol=1e-10, rtol=1e-10)):
        raise ValueError('Expected a finite symmetric float64 moment')
    eigen, vectors = torch.linalg.eigh((value + value.T) / 2)
    tolerance = 1e-10 * max(1., float(eigen.abs().max()))
    if float(eigen.min()) < -tolerance:
        raise ValueError('Moment is not positive semidefinite')
    return (vectors * eigen.clamp_min(0)) @ vectors.T


@torch.no_grad()
def estimate(before, after, weights, shift, old_mu, old_Q, ridge=.001):
    """Extrapolate an affine residual map to old moments, never to generated samples."""
    if (before.ndim != 2 or not len(before) or before.dtype != torch.float64
            or after.shape != before.shape or weights.shape != (len(before),)
            or shift.shape != before.shape[1:] or old_mu.ndim != 2
            or old_mu.shape[1] != before.shape[1]
            or old_Q.shape != (len(old_mu), before.shape[1], before.shape[1])
            or not math.isfinite(ridge) or ridge <= 0
            or any(v.dtype != before.dtype or v.device != before.device or not torch.isfinite(v).all()
                   for v in (after, weights, shift, old_mu, old_Q))
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('Invalid paired fit features or task-start old moments')
    weights = weights / weights.sum()
    mean = weights @ before
    x = before - mean
    residual = after - before - shift
    intercept = weights @ residual
    centered = residual - intercept
    covariance = psd(x.T @ (weights[:, None] * x))
    eigen, vectors = torch.linalg.eigh(covariance)
    tolerance = torch.finfo(before.dtype).eps * max(before.shape) * max(float(eigen.max()), 1e-30)
    basis = vectors[:, eigen > tolerance]
    projector = basis @ basis.T
    inverse = (vectors * (1 / (eigen.clamp_min(0) + ridge))) @ vectors.T
    mapping = projector @ inverse @ (x.T @ (weights[:, None] * centered))
    predicted = x @ mapping + intercept
    moments, support = [], []
    for mu, Q in zip(old_mu, old_Q):
        cov = psd(Q - torch.outer(mu, mu))
        predicted_mean = (mu - mean) @ mapping + intercept
        moments.append(psd(mapping.T @ cov @ mapping + torch.outer(predicted_mean, predicted_mean)))
        centered_old = cov + torch.outer(mu - mean, mu - mean)
        total = float(centered_old.trace())
        support.append(float((projector @ centered_old).trace()) / total if total > 1e-30 else None)
    U = torch.stack(moments).mean(0) if moments else covariance.new_zeros(covariance.shape)
    audit = dict(old_classes=len(old_mu), current_fit_samples=len(before), old_samples_read=0,
                 estimate_is_proxy=True, current_support_rank=len(basis.T),
                 old_supported_energy_fractions=support,
                 fit_residual_mse=float((weights * (predicted - residual).square().sum(1)).sum()),
                 raw_old_risk_trace=float(U.trace()), map_norm=float(mapping.norm()),
                 intercept_norm=float(intercept.norm()), generated_features=0)
    return U, audit


@torch.no_grad()
def remove_local_prior_span(U, base_prior, local_prior):
    """Remove overlap with actual added local-prior relaxation, not all regional centers."""
    U = psd(U)
    if base_prior.shape != U.shape or local_prior.shape != U.shape:
        raise ValueError('Prior dimensions differ')
    delta = psd(base_prior - local_prior)
    eigen, vectors = torch.linalg.eigh(delta)
    tolerance = torch.finfo(U.dtype).eps * len(U) * max(float(eigen.max()), 1e-30)
    basis = vectors[:, eigen > tolerance]
    complement = torch.eye(len(U), dtype=U.dtype, device=U.device) - basis @ basis.T
    filtered = psd(complement @ U @ complement)
    # Reproject after numerical PSD repair; no renormalization of removed energy.
    filtered = (complement @ filtered @ complement)
    filtered = (filtered + filtered.T) / 2
    return filtered, dict(local_relaxation_rank=basis.shape[1], retained_risk_trace=float(filtered.trace()),
                          raw_risk_trace=float(U.trace()), overlap_leakage=float((basis.T @ filtered).norm()),
                          is_exact_transport_correction=False)


@torch.no_grad()
def bounded_penalty(U, reference_system, cap=.1, raw_risk=None):
    """Freeze S at epoch/boundary: 0 <= S <= cap * reference_system."""
    if isinstance(cap, bool) or not math.isfinite(cap) or not 0 <= cap <= 1:
        raise ValueError('Invalid prospective curvature cap')
    U = psd(U)
    H = psd(reference_system)
    if H.shape != U.shape or H.dtype != U.dtype or H.device != U.device:
        raise ValueError('Reference system mismatch')
    eigen, vectors = torch.linalg.eigh(H)
    if float(eigen.min()) <= 0:
        raise ValueError('Reference curvature must be positive definite')
    inverse_half = (vectors * eigen.rsqrt()) @ vectors.T
    whitened = psd(inverse_half @ U @ inverse_half)
    radius = float(torch.linalg.eigvalsh(whitened).max())
    retained_radius = radius
    if raw_risk is not None:
        raw_risk = psd(raw_risk)
        if raw_risk.shape != U.shape:
            raise ValueError('Raw risk dimension mismatch')
        radius = max(radius, float(torch.linalg.eigvalsh(psd(inverse_half @ raw_risk @ inverse_half)).max()))
    S = cap / (1 + radius) * U
    return S, dict(curvature_cap=cap, raw_whitened_radius=radius,
                   actual_relative_bound=cap * retained_radius / (1 + radius), penalty_trace=float(S.trace()),
                   removed_energy_not_renormalized=raw_risk is not None)


def old_margin_metric(classes, old_count, like):
    if isinstance(old_count, bool) or not isinstance(old_count, int) or not 0 <= old_count <= classes:
        raise ValueError('Invalid old-class prefix')
    M = like.new_zeros((classes, classes))
    if old_count > 1:
        M[:old_count, :old_count] = (torch.eye(old_count, dtype=like.dtype, device=like.device)
                                   - like.new_ones((old_count, old_count)) / old_count)
    return M


def solve(q, mu, mass, pairs, regularizer, S, M, *, beta=.5, previous=None,
          proximal=0., inverse=None, x=None, alpha=None, native=None):
    """Solve native Hessian plus S @ W @ M; preserve native RHS and competition."""
    control, control_audit = competition.solve(q, mu, mass, pairs, beta=beta, previous=previous,
        proximal=proximal, inverse=inverse, x=x, alpha=alpha, native=native, regularizer=regularizer)
    S, M = psd(S), psd(M)
    if S.shape != regularizer.shape or M.shape != (mu.shape[0], mu.shape[0]):
        raise ValueError('LCA prior dimensions differ')
    if not S.any() or not M.any():
        return control, dict(control_audit, exact_native_reduction=True)
    d = q.shape[1]
    A = q.sum(0) + .001 * regularizer + proximal * torch.eye(d, dtype=q.dtype, device=q.device)
    rhs = mu.T.clone()
    if proximal:
        rhs += proximal * previous
    lap, targets = competition.laplacians(pairs)
    rhs += beta * mu.T @ targets
    if inverse is None:
        factor = torch.linalg.cholesky(A)
        def precondition(value):
            return torch.cholesky_solve(value, factor)
    else:
        u = x.T * alpha.sqrt()[None, :]
        v = inverse @ u
        factor = torch.linalg.cholesky(torch.eye(len(x), dtype=x.dtype, device=x.device) + u.T @ v)
        def precondition(value):
            return inverse @ value - v @ torch.cholesky_solve(u.T @ (inverse @ value), factor)
    def operator(w):
        return A @ w + beta * torch.einsum('cdk,ckl->dl', q @ w, lap) + S @ w @ M
    result, audit = competition.pcg(operator, rhs, precondition, control)
    return result, dict(audit, exact_native_reduction=False,
                       explicit_new_column_force_norm=float((S @ result @ M)[:, (M.abs().sum(0)==0)].norm()))


def loss(W, S, M):
    return .5 * (W * (S @ W @ M)).sum()


def checkpoint(S, M):
    return dict(version=VERSION, S=psd(S).detach().cpu(), M=psd(M).detach().cpu())


def restore(state, dim, classes, like):
    if state.get('version') != VERSION:
        raise ValueError('LCA checkpoint version mismatch')
    S, M = psd(state['S']), psd(state['M'])
    if S.shape != (dim, dim) or M.shape != (classes, classes):
        raise ValueError('LCA checkpoint scope mismatch')
    return S.to(like), M.to(like)


def binary_prior(prior, S, old_label, ridge=.001):
    """Compatibility helper only: independent signed binary heads have no class centering."""
    if not isinstance(old_label, bool) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Invalid binary scope')
    if not old_label:
        return prior
    S = psd(S)
    if prior.shape != (len(S)+1, len(S)+1):
        raise ValueError('Expected existing augmented binary bias prior')
    result = prior.clone()
    result[:-1, :-1] += S / ridge
    return result
