"""CPU design only: temporal prototype rotations and a consistent quadratic head anchor."""
import math

import torch

import core1_competition as native

VERSION = 'temporal_native_geometry_anchor_v1_CPU_DESIGN'
SUPPORT_SMOOTHING = 16.
RELATIVE_CURVATURE = .1


def validate(*tensors):
    if any(t.device.type != 'cpu' or t.dtype != torch.float64 or not torch.isfinite(t).all() for t in tensors):
        raise ValueError('Finite CPU float64 inputs required')


def temporal_target(reference, before, current, start_curvature, counts, ages):
    """Keep the native head/prototype offset; rotate only the observed centered direction."""
    validate(reference, before, current, start_curvature)
    d, k = reference.shape
    if before.shape != (k, d) or current.shape != before.shape or start_curvature.shape != (d, d):
        raise ValueError('Invalid frozen task-start geometry')
    if len(counts) != k or len(ages) != k or any(type(n) is not int or n <= 0 for n in counts) or any(type(a) is not int or a < 0 for a in ages):
        raise ValueError('Actual support and transport ages required')
    if not torch.allclose(start_curvature, start_curvature.T, atol=1e-12, rtol=1e-10):
        raise ValueError('Task-start curvature must be symmetric')
    C = torch.linalg.cholesky(start_curvature)
    b0 = torch.cholesky_solve((before-before.mean(0)).T, C)
    bt = torch.cholesky_solve((current-current.mean(0)).T, C)
    common = reference.mean(1, keepdim=True)
    w0 = reference-common
    rotated = w0.clone()
    target = reference.clone()
    support = reference.new_tensor([n/(n+SUPPORT_SMOOTHING)/(1+a) for n,a in zip(counts, ages)])
    audit = []
    for c in range(k):
        norm0, normt = float(b0[:, c].norm()), float(bt[:, c].norm())
        reason = 'identified_rotation'
        angle = applied = 0.
        if min(norm0, normt) <= 1e-12:
            support[c] = 0.; reason = 'zero_centered_direction_keep_reference'
        else:
            u, v = b0[:, c]/norm0, bt[:, c]/normt
            cosine = float(u.dot(v).clamp(-1., 1.))
            orthogonal = v-cosine*u
            sine = float(orthogonal.norm())
            angle = math.atan2(sine, cosine)
            if cosine < 0 and sine <= 1e-8:
                support[c] = 0.; reason = 'antipodal_plane_unidentified_keep_reference'
            elif sine > 1e-8:
                tangent = orthogonal/sine
                applied = float(support[c])*angle
                a, b = math.cos(applied), math.sin(applied)
                x = w0[:, c]; xu, xt = u.dot(x), tangent.dot(x)
                rotated[:, c] = x+(a-1)*(xu*u+xt*tangent)+b*(xu*tangent-xt*u)
                target[:, c] = rotated[:, c]+common[:, 0]
            else:
                reason = 'no_direction_change_keep_reference'
        audit.append(dict(column=c, support=counts[c], transport_age=ages[c], reliability_score=float(support[c]),
                          prototype_rotation_radians=angle, applied_rotation_radians=applied, reason=reason,
                          raw_rotation_norm_error=float(abs(rotated[:, c].norm()-w0[:, c].norm()))))
    return target, support, audit


def anchor_matrix(weights):
    validate(weights)
    if weights.ndim != 1 or (weights < 0).any() or (weights > 1).any():
        raise ValueError('Bounded reliability scores required')
    k = len(weights)
    if k < 2:
        raise ValueError('At least two seen classes required')
    P = torch.eye(k, dtype=weights.dtype)-torch.ones((k,k), dtype=weights.dtype)/k
    return P@torch.diag(weights)@P


def anchor_loss(head, target, weights, strength):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Nonnegative finite anchor curvature required')
    H = anchor_matrix(weights)
    error = head-target
    return .5*strength*(error*(error@H)).sum()


def solve(q, mu, mass, pairs, R, target, weights, strength, previous=None, proximal=0.):
    """Same native objective plus a PSD class-contrast term; no inference entry point."""
    validate(q, mu, mass, pairs, R, target, weights)
    k, d = mu.shape
    if q.shape != (k,d,d) or R.shape != (d,d) or pairs.shape != (k,k) or target.shape != (d,k) or weights.shape != (k,):
        raise ValueError('Native/anchor shape mismatch')
    if mass.shape != (k,) or (mass < 0).any() or (pairs < 0).any() or pairs.diag().any() or not torch.allclose(pairs.sum(1), pairs.new_ones(k), atol=1e-10, rtol=1e-10):
        raise ValueError('Invalid native class masses or competition')
    if not torch.allclose(R, R.T, atol=1e-12, rtol=1e-10) or not torch.allclose(q, q.transpose(1, 2), atol=1e-12, rtol=1e-10):
        raise ValueError('Symmetric statistical geometry required')
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Invalid strength')
    if not math.isfinite(proximal) or proximal < 0 or (proximal and previous is None):
        raise ValueError('Invalid native proximal reference')
    if previous is not None:
        validate(previous)
        if previous.shape != target.shape:
            raise ValueError('Proximal reference shape mismatch')
    if strength == 0 or not weights.any():
        return native.solve(q, mu, mass, pairs, regularizer=R, previous=previous, proximal=proximal)
    H = anchor_matrix(weights)
    G = q.sum(0)+.001*R+proximal*torch.eye(d, dtype=q.dtype)
    factor = torch.linalg.cholesky(G)
    lap, targets = native.laplacians(pairs)
    rhs = mu.T+.5*(mu.T@targets)+strength*(target@H)
    if proximal:
        rhs = rhs+proximal*previous

    def operator(w):
        return G@w+.5*torch.einsum('cdk,ckl->dl', q@w, lap)+strength*(w@H)

    def precondition(w):
        return torch.cholesky_solve(w, factor)

    initial = precondition(rhs)
    return native.pcg(operator, rhs, precondition, initial)
