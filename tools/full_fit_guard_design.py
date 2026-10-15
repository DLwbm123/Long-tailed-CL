"""CPU-only instrumentation of the frozen pointwise design, never a medical worker."""
import time
import torch
import native_pointwise_guard_design as native


def constrained(op, rhs, solve, support, labels, ref, wu, old_count, trace,
                max_atoms=64, time_limit=240., tolerance=2e-8):
    k = ref.shape[1]
    if labels.shape != (len(support),) or labels.dtype != torch.long or not len(labels):
        raise ValueError('Invalid current-fit labels')
    if not 0 <= old_count < k or bool(((labels < old_count) | (labels >= k)).any()):
        raise ValueError('Historical/future support rejected')
    if not all(torch.isfinite(z).all() for z in (support, ref, wu)):
        raise ValueError('Nonfinite supports/head')
    begun = time.monotonic(); target = native.contrasts(support, labels, ref)
    rows = []; atoms = []; directions = []; lam = support.new_empty(0)
    gram = support.new_empty((0, 0)); audits = []; w = wu.clone()
    result = 'UNKNOWN'
    while True:
        slack = native.contrasts(support, labels, w)-target
        slack.scatter_(1, labels[:, None], torch.inf)
        minimum = float(slack.min())
        raw_value = float(native.objective(op, rhs, w))
        trace.append(dict(atoms=len(rows), positive_duals=int((lam > 1e-9).sum()),
                          minimum_slack=minimum, relaxed_objective=raw_value,
                          solver_wall_seconds=time.monotonic()-begun))
        if minimum >= -tolerance: result = 'FEASIBLE'; break
        if len(rows) == max_atoms: result = 'ATOM_CAP_NOT_SOLVED'; break
        if time.monotonic()-begun > time_limit: result = 'TIME_CAP_NOT_SOLVED'; break
        row, rival = divmod(int(slack.argmin()), k)
        if (row, rival) in rows: raise ValueError('Repeated violated active constraint')
        at = native.atom(support, labels, row, rival, k)
        direction, audit = solve(at); audits.append(audit)
        if atoms:
            av = torch.stack(atoms).flatten(1) @ direction.flatten()
            va = torch.stack(directions).flatten(1) @ at.flatten()
            edge = (av+va)/2
            g = support.new_empty((len(rows)+1, len(rows)+1))
            g[:-1, :-1] = gram; g[:-1, -1] = edge; g[-1, :-1] = edge
            gram = g
        else: gram = support.new_empty((1, 1))
        gram[-1, -1] = (at*direction).sum()
        rows.append((row, rival)); atoms.append(at); directions.append(direction)
        cross = torch.stack([target[i, j]-(v*wu).sum() for (i, j), v in zip(rows, atoms)])
        lam = native.nonnegative_qp(gram, cross)
        w = wu+torch.einsum('a,adk->dk', lam, torch.stack(directions))
    force = torch.einsum('a,adk->dk', lam, torch.stack(atoms)) if atoms else torch.zeros_like(w)
    stationary = float((op(w)-rhs-force).norm()/(1+rhs.norm()+force.norm()))
    slack = native.contrasts(support, labels, w)-target
    complement = max([abs(float(l*slack[i, j])) for l, (i, j) in zip(lam, rows)]+[0.])
    if stationary > 1e-8 or complement > 1e-7: raise ValueError('Relaxation KKT failed')
    ref_value = float(native.objective(op, rhs, ref)); proposal_value = float(native.objective(op, rhs, wu))
    value = float(native.objective(op, rhs, w)); gain = ref_value-value; available = ref_value-proposal_value
    delta = (w-ref)-(w-ref).mean(1, keepdim=True)
    proposal_delta = (wu-ref)-(wu-ref).mean(1, keepdim=True)
    relative = float(delta.norm()/proposal_delta.norm().clamp_min(1e-30))
    feasible = result == 'FEASIBLE'
    audit = dict(status=result, constraints=len(support)*(k-1), atoms=len(rows),
        positive_duals=int((lam > 1e-9).sum()), minimum_slack=float(slack.min()),
        stationarity=stationary, complementarity=complement, objective_reference=ref_value,
        objective_unconstrained=proposal_value, objective_relaxation=value,
        available_quadratic_gain=available, feasible_quadratic_gain=gain if feasible else None,
        feasible_retained_gain=gain/available if feasible and available > 1e-14 else None,
        remaining_gain_upper_bound=max(0., gain),
        discriminative_delta=float(delta.norm()) if feasible else None,
        relative_discriminative_delta=relative if feasible else None,
        discriminative_identity=(float(delta.norm()) <= 1e-10 or relative <= 1e-6) if feasible else None,
        useful_synthetic_update=(float(delta.norm()) > 1e-10 and gain > 1e-12 and gain > .01*available) if feasible else None,
        support_score_change_norm=float((support @ delta).norm()) if feasible else None,
        old_class_risk_or_val_gain_certified=False, cached_directions_bytes=len(rows)*w.numel()*w.element_size(),
        cached_atoms_bytes=len(rows)*w.numel()*w.element_size(), gram_bytes=gram.numel()*gram.element_size(),
        inverse_total_iterations=sum(a['iterations'] for a in audits),
        inverse_max_relative_residual=max([a['relative_residual'] for a in audits]+[0.]),
        solver_wall_seconds=time.monotonic()-begun)
    # Incomplete relaxation is a bound, never an accepted guard or an identity result.
    return w, audit, [(i, j, float(l)) for (i, j), l in zip(rows, lam)]


def refresh_term(ref_op, ref_rhs, ref_solve, reference, force):
    """VJP of Wref=H0^-1 B0, with a/R/previous frozen inside this block."""
    with torch.no_grad(): adjoint, audit = ref_solve(force.detach())
    value = (adjoint.detach()*(ref_rhs-ref_op(reference.detach()))).sum()
    return value, audit
