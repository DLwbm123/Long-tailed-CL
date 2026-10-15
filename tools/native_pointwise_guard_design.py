"""CPU-only native quadratic pointwise-margin design; no medical worker."""
import torch
import core1_competition as native


def system(q, mu, mass, a, regularizer, previous, proximal=.01, inverse=None,
           x=None, alpha=None):
    """Exactly the native full-second-moment competition/proximal operator."""
    d, k = mu.shape[1], len(mu)
    if q.shape != (k, d, d) or regularizer.shape != (d, d):
        raise ValueError('Invalid native dimensions')
    if not all(torch.isfinite(v).all() for v in (q, mu, mass, a, regularizer, previous)):
        raise ValueError('Nonfinite native inputs')
    if not torch.allclose(regularizer, regularizer.T, atol=1e-10, rtol=1e-10):
        raise ValueError('Asymmetric metric')
    base = q.sum(0)+.001*regularizer+proximal*torch.eye(d, dtype=q.dtype, device=q.device)
    lap, targets = native.laplacians(a)
    rhs = mu.T+proximal*previous+.5*(mu.T @ targets)
    if inverse is None:
        factor = torch.linalg.cholesky(base)
        def precondition(v): return torch.cholesky_solve(v, factor)
    else:
        u = x.T*alpha.sqrt()[None, :]; v = inverse @ u
        small = torch.linalg.cholesky(torch.eye(len(x), dtype=x.dtype, device=x.device)+u.T @ v)
        def precondition(z):
            value = inverse @ z
            return value-v @ torch.cholesky_solve(u.T @ value, small)
    def operator(w):
        return base @ w+.5*torch.einsum('cdk,ckl->dl', q @ w, lap)
    def solve(b):
        return native.pcg(operator, b, precondition, precondition(b), tolerance=1e-11, iterations=256)
    return operator, rhs, solve


def contrasts(x, labels, w):
    scores = x @ w
    return scores.gather(1, labels[:, None])-scores


def atom(x, labels, row, rival, classes):
    value = x.new_zeros((x.shape[1], classes))
    value[:, int(labels[row])] = x[row]
    value[:, rival] -= x[row]
    return value


def nonnegative_qp(gram, cross, tolerance=1e-11):
    """Small active-face dual solve, with blocking steps for negative multipliers."""
    n = len(cross); lam = torch.zeros_like(cross); passive = []
    for _ in range(20*max(1, n*n)):
        gain = cross-gram @ lam
        remaining = [i for i in range(n) if i not in passive]
        if not remaining or float(gain[remaining].max()) <= tolerance:
            return lam
        passive.append(remaining[int(gain[remaining].argmax())])
        while passive:
            ids = torch.tensor(passive, device=cross.device)
            block = gram[ids][:, ids]
            z = torch.zeros_like(lam)
            # shortcut: capped active set; fail explicitly rather than approximate a large medical QP.
            z[ids] = torch.linalg.lstsq(block, cross[ids], rcond=1e-12).solution
            if float((block @ z[ids]-cross[ids]).norm()) > 1e-8*(1+float(cross[ids].norm())):
                raise ValueError('Dependent active constraints are inconsistent')
            if bool((z[ids] > tolerance).all()):
                lam = z; break
            bad = [i for i in passive if float(z[i]) <= tolerance]
            ratios = [(float(lam[i]/(lam[i]-z[i])), i) for i in bad if float(lam[i]-z[i]) > 0]
            if ratios:
                step = min(v[0] for v in ratios)
                lam = lam+step*(z-lam)
            for i in bad:
                if float(lam[i]) <= tolerance:
                    lam[i] = 0.; passive.remove(i)
    raise ValueError('Dual active-face iteration cap reached')


def guarded(operator, rhs, solve, x, labels, reference, unconstrained, old_count,
            enabled=True, max_atoms=64, tolerance=2e-8):
    if x.ndim != 2 or reference.shape != unconstrained.shape or x.shape[1] != len(reference):
        raise ValueError('Invalid guard dimensions')
    k = reference.shape[1]
    if labels.dtype != torch.long or labels.shape != (len(x),) or not len(x):
        raise ValueError('Invalid guard labels')
    if not 0 <= old_count < k or bool(((labels < old_count) | (labels >= k)).any()):
        raise ValueError('Guard accepts current-fit class labels only')
    if not all(torch.isfinite(z).all() for z in (x, reference, unconstrained)):
        raise ValueError('Nonfinite guard inputs')
    if not enabled:
        return unconstrained.clone(), dict(disabled_exact=True), []
    target = contrasts(x, labels, reference)
    w = unconstrained.clone(); rows = []; atoms = []; directions = []; audits = []; lam = x.new_empty(0)
    for _ in range(max_atoms+1):
        slack = contrasts(x, labels, w)-target
        slack.scatter_(1, labels[:, None], torch.inf)
        worst = int(slack.argmin()); row, rival = divmod(worst, k)
        if float(slack[row, rival]) >= -tolerance:
            break
        if (row, rival) in rows or len(rows) == max_atoms:
            raise ValueError('Guard separation/active-set cap reached')
        rows.append((row, rival)); atoms.append(atom(x, labels, row, rival, k))
        direction, audit = solve(atoms[-1]); directions.append(direction); audits.append(audit)
        gram = torch.stack([torch.stack([(v*z).sum() for z in directions]) for v in atoms])
        gram = (gram+gram.T)/2
        cross = torch.stack([target[i, j]-(atoms[t]*unconstrained).sum() for t, (i, j) in enumerate(rows)])
        lam = nonnegative_qp(gram, cross)
        w = unconstrained+sum((l*z for l, z in zip(lam, directions)), torch.zeros_like(w))
    correction = sum((l*v for l, v in zip(lam, atoms)), torch.zeros_like(w))
    stationarity = float((operator(w)-rhs-correction).norm()/(1+rhs.norm()+correction.norm()))
    slacks = contrasts(x, labels, w)-target
    complementarity = max([abs(float(lam[t]*slacks[i, j])) for t, (i, j) in enumerate(rows)]+[0.])
    audit = dict(constraints=len(x)*(k-1), separated_atoms=len(rows), active=int((lam > 1e-9).sum()),
        minimum_slack=float(slacks.min()), stationarity=stationarity, complementarity=complementarity,
        min_multiplier=float(lam.min()) if len(lam) else 0.,
        inverse_max_relative_residual=max([v['relative_residual'] for v in audits]+[0.]),
        inverse_total_iterations=sum(v['iterations'] for v in audits),
        discriminative_delta=float(((w-reference)-(w-reference).mean(1, keepdim=True)).norm()),
        cached_inverse_bytes=len(rows)*w.numel()*w.element_size(), full_constraint_tensor_built=False,
        full_head_hessian_built=False)
    if stationarity > 1e-8 or complementarity > 1e-7:
        raise ValueError('Guard KKT verification failed')
    return w, audit, [(i, j, float(lam[t])) for t, (i, j) in enumerate(rows)]


def objective(operator, rhs, w):
    return .5*(w*operator(w)).sum()-(rhs*w).sum()
