"""Frozen temporal rotations and a consistent analytic-head anchor; no val selection."""
import math

import torch

import core1_competition as native

VERSION = 'temporal_native_geometry_anchor_v1'
SUPPORT_SMOOTHING = 16.
RELATIVE_CURVATURE = .1


def validate(*tensors):
    if any(t.dtype != torch.float64 or t.device != tensors[0].device or not torch.isfinite(t).all() for t in tensors):
        raise ValueError('Finite same-device float64 inputs required')


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
    P = torch.eye(k, dtype=weights.dtype, device=weights.device)-torch.ones((k,k), dtype=weights.dtype, device=weights.device)/k
    return P@torch.diag(weights)@P


def anchor_loss(head, target, weights, strength):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Nonnegative finite anchor curvature required')
    H = anchor_matrix(weights)
    error = head-target
    return .5*strength*(error*(error@H)).sum()


def solve(q, mu, mass, pairs, R, target, weights, strength, previous=None, proximal=0., inverse=None, x=None, alpha=None, native_head=None):
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
        return native.solve(q, mu, mass, pairs, regularizer=R, previous=previous, proximal=proximal, inverse=inverse, x=x, alpha=alpha, native=native_head)
    H = anchor_matrix(weights)
    G = q.sum(0)+.001*R+proximal*torch.eye(d, dtype=q.dtype, device=q.device)
    lap, targets = native.laplacians(pairs)
    rhs = mu.T+.5*(mu.T@targets)+strength*(target@H)
    if proximal:
        rhs = rhs+proximal*previous

    def operator(w):
        return G@w+.5*torch.einsum('cdk,ckl->dl', q@w, lap)+strength*(w@H)

    if inverse is None:
        factor = torch.linalg.cholesky(G)
        def precondition(w):return torch.cholesky_solve(w, factor)
    else:
        validate(inverse, x, alpha)
        u=x.T*alpha.sqrt()[None,:];v=inverse@u
        factor=torch.linalg.cholesky(torch.eye(len(x),dtype=x.dtype,device=x.device)+u.T@v)
        def precondition(w):
            value=inverse@w
            return value-v@torch.cholesky_solve(u.T@value,factor)
    initial = precondition(rhs) if native_head is None else native_head
    head,audit=native.pcg(operator, rhs, precondition, initial)
    force=strength*((head-target)@H)
    return head,dict(audit,anchor_force_norm=float(force.norm()))


@torch.no_grad()
def prepare(reference_state, candidate, pairs, R, mode, ages):
    k=len(candidate['n'])
    if reference_state is None:
        W0,_=native.bank_head(candidate,pairs,regularizer=R)
        G0=candidate['Q'].sum(0)/k+.001*R
        reference_state=dict(W0=W0.detach().clone(),mu0=candidate['mu'].detach().clone(),G0=G0.detach().clone(),
                             counts=list(candidate['n']),ages=list(ages),strength=RELATIVE_CURVATURE*float(G0.diagonal().mean()))
    A,rho,angles=temporal_target(reference_state['W0'],reference_state['mu0'],candidate['mu'],reference_state['G0'],reference_state['counts'],reference_state['ages'])
    if mode not in ('dynamic','static'):raise ValueError('Frozen dynamic/static target required')
    dynamic_difference=float((A-reference_state['W0']).norm()/reference_state['W0'].norm().clamp_min(1e-30))
    if mode=='static':A=reference_state['W0'].clone()
    state=dict(version=VERSION,target=A.detach(),weights=rho.detach(),strength=reference_state['strength'])
    return reference_state,state,dict(strength=state['strength'],support_min=float(rho.min()),support_mean=float(rho.mean()),
        support_max=float(rho.max()),prototype_rotation_mean=sum(a['prototype_rotation_radians'] for a in angles)/k,
        applied_rotation_mean=sum(a['applied_rotation_radians'] for a in angles)/k,
        dynamic_target_relative_difference=dynamic_difference,target_relative_difference=float((A-reference_state['W0']).norm()/reference_state['W0'].norm().clamp_min(1e-30)),
        raw_rotation_norm_error=max(a['raw_rotation_norm_error'] for a in angles),detached=True,fit_only=True,
        zero_or_unidentified_columns=sum(a['reliability_score']==0 for a in angles))


def bank_head(bank,pairs,regularizer,state):
    k=len(bank['n'])
    return solve(bank['Q']/k,bank['mu']/k,bank['mu'].new_full((k,),1/k),pairs,regularizer,
                 state['target'],state['weights'],state['strength'])


def checkpoint(state):
    return dict(version=VERSION,target=state['target'].detach().cpu(),weights=state['weights'].detach().cpu(),strength=state['strength'])


def restore(state,dim,classes,like):
    if state.get('version')!=VERSION or state['target'].shape!=(dim,classes) or state['weights'].shape!=(classes,):
        raise ValueError('Anchor checkpoint version/dimension mismatch')
    target,weights=state['target'].to(like),state['weights'].to(like);validate(target,weights);anchor_matrix(weights)
    if not math.isfinite(state['strength']) or state['strength']<0:raise ValueError('Invalid saved curvature')
    return dict(version=VERSION,target=target,weights=weights,strength=state['strength'])
