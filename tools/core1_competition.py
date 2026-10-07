"""Prototype-directed quadratic class competition, without historical samples."""
import torch


def competition(bank, reference, mode='prototype', historical_count=None):
    k = len(bank['n'])
    if k < 2 or mode not in ('prototype', 'uniform'):
        raise ValueError('Competition requires at least two classes and a valid mode')
    uniform = (1-torch.eye(k, dtype=reference.dtype, device=reference.device))/(k-1)
    hardness = torch.zeros_like(uniform)
    for c in bank['components']:
        label = c['label']; scores = c['center'] @ reference
        hardness[label] += c['mass']*(1-scores[label]+scores).clamp_min(0).square()
    hardness.fill_diagonal_(0)
    total = hardness.sum(1, keepdim=True)
    directed = torch.where(total > 1e-12, hardness/total.clamp_min(1e-12), uniform)
    a = uniform if mode == 'uniform' else .5*uniform+.5*directed
    if not torch.isfinite(a).all() or not torch.allclose(a.sum(1), torch.ones(k, device=a.device, dtype=a.dtype)):
        raise ValueError('Invalid competition mass')
    if historical_count is not None:
        if not isinstance(historical_count, int) or not 0 <= historical_count <= k:
            raise ValueError('Invalid historical class count')
        # Keep historical row mass unchanged; current labels retain native supervision.
        a[historical_count:] = 0
    return a.detach()


def laplacians(a):
    k = len(a); eye = torch.eye(k, dtype=a.dtype, device=a.device)
    differences = eye[:, None, :]-eye[None, :, :]
    return torch.einsum('cj,cjl,cjm->clm', a, differences, differences), (a.sum(1)[:, None]*eye-a)


def moments(old, x, y, alpha, classes):
    """Mass-scaled moments of exactly the native minibatch objective."""
    qs, mus, masses = [], [], []
    for c in range(classes):
        if c < len(old['n']):
            qs.append(old['Q'][c]/classes); mus.append(old['mu'][c]/classes)
            masses.append(x.new_tensor(1/classes))
        else:
            rows = y == c; z, w = x[rows], alpha[rows]
            qs.append(z.T @ (w[:, None]*z)); mus.append((w[:, None]*z).sum(0)); masses.append(w.sum())
    return torch.stack(qs), torch.stack(mus), torch.stack(masses)


def pair_moments(q, mu, mass, mean_only):
    if not mean_only:
        return q
    return mu[:, :, None]*mu[:, None, :]/mass.clamp_min(1e-30)[:, None, None]


def pcg(operator, rhs, precondition, initial, tolerance=1e-9, iterations=128):
    w = initial.clone(); r = rhs-operator(w)
    scale = rhs.norm().clamp_min(1e-30)
    z = precondition(r); p = z.clone(); rz = (r*z).sum()
    count = 0
    while float(r.norm()/scale) > tolerance and count < iterations:
        ap = operator(p); curvature = (p*ap).sum()
        if not torch.isfinite(curvature) or curvature <= 0:
            raise ValueError('Competition Hessian lost positive definiteness')
        step = rz/curvature; w = w+step*p; r = r-step*ap
        z = precondition(r); next_rz = (r*z).sum()
        p = z+(next_rz/rz)*p; rz = next_rz; count += 1
    residual = float((operator(w)-rhs).norm()/scale)
    if not torch.isfinite(w).all() or residual > 1e-8:
        raise ValueError(f'Competition solve failed: residual={residual}')
    return w, dict(iterations=count, relative_residual=residual)


def solve(q, mu, mass, a, beta=.5, mean_only=False, previous=None,
          proximal=0., inverse=None, x=None, alpha=None, native=None, pair_linear=None):
    if beta < 0 or proximal < 0:
        raise ValueError('Nonnegative penalties required')
    d = q.shape[1]; eye = torch.eye(d, dtype=q.dtype, device=q.device)
    base = q.sum(0)+(.001+proximal)*eye
    rhs = mu.T.clone()
    if proximal:
        if previous is None: raise ValueError('Missing proximal reference')
        rhs += proximal*previous
    if inverse is None:
        factor = torch.linalg.cholesky(base)
        def precondition(v): return torch.cholesky_solve(v, factor)
    else:
        # Reuse the native Woodbury inverse; no d-by-d factorization per batch.
        u = x.T*alpha.sqrt()[None, :]; v = inverse @ u
        small = torch.linalg.cholesky(torch.eye(len(x), dtype=x.dtype, device=x.device)+u.T @ v)
        def precondition(value):
            result = inverse @ value
            return result-v @ torch.cholesky_solve(u.T @ result, small)
    if native is None: native = precondition(rhs)
    if not beta:
        residual = float((base @ native-rhs).norm()/rhs.norm().clamp_min(1e-30))
        if residual > 1e-8: raise ValueError('Native solve mismatch')
        return native, dict(iterations=0, relative_residual=residual)
    lap, targets = laplacians(a)
    qp = pair_moments(q, mu, mass, mean_only)
    if pair_linear is not None and (pair_linear.shape != rhs.shape or not torch.isfinite(pair_linear).all()):
        raise ValueError('Invalid competition target cross moment')
    rhs = rhs+beta*(mu.T @ targets if pair_linear is None else pair_linear)
    def operator(w):
        return base @ w+beta*torch.einsum('cdk,ckl->dl', qp @ w, lap)
    return pcg(operator, rhs, precondition, native)


def bank_head(bank, a, beta=.5, mean_only=False):
    k = len(bank['n']); mass = bank['mu'].new_full((k,), 1/k)
    return solve(bank['Q']/k, bank['mu']/k, mass, a, beta, mean_only)


def pair_loss(old, x, y, alpha, w, a, mean_only=False):
    scores = x @ w; k = w.shape[1]
    if mean_only:
        value = scores.new_zeros(())
        for c in y.unique().tolist():
            selected = y == c; mass = alpha[selected].sum()
            center = (alpha[selected, None]*scores[selected]).sum(0)/mass
            value += mass*(a[c]*(center[c]-center-1).square()).sum()
    else:
        margins = scores.gather(1, y[:, None])-scores-1
        value = (alpha*(a[y]*margins.square()).sum(1)).sum()
    if len(old['n']):
        lap, targets = laplacians(a)
        q = old['Q'] if not mean_only else old['mu'][:, :, None]*old['mu'][:, None, :]
        value += (torch.einsum('dl,cdk,ckl->c', w, q @ w, lap[:len(q)])
                  -2*(old['mu'] @ w*targets[:len(q)]).sum(1)+a[:len(q)].sum(1)).sum()/k
    return .5*value


def self_check():
    import prototype_coherent as native
    torch.manual_seed(43); torch.set_num_threads(2)
    x = torch.randn(18, 5, dtype=torch.float64); y = torch.arange(3).repeat_interleave(6)
    group = torch.arange(6).repeat_interleave(3); weights = torch.full((18,), 1/6, dtype=x.dtype)
    bank = native.append(native.empty(5), x, y, group, weights)
    ref, _ = native.head(bank, 0); a = competition(bank, ref)
    assert torch.allclose(a.sum(1), torch.ones(3, dtype=x.dtype)) and torch.all(a.diag() == 0)
    assert not torch.allclose(a, competition(bank, ref, 'uniform'))
    for mean_only in (False, True):
        w, audit = bank_head(bank, a, mean_only=mean_only)
        target = torch.nn.functional.one_hot(y, 3)
        def explicit(v):
            base = .5*((x @ v-target).square().sum(1)).mean()+.0005*v.square().sum()
            pair = pair_loss(native.empty(5), x, y, x.new_full((18,), 1/18), v, a, mean_only)
            return base+.5*pair
        w0 = torch.zeros((5,3), dtype=x.dtype, requires_grad=True)
        h = torch.autograd.functional.hessian(explicit, w0).reshape(15,15)
        b = -torch.autograd.grad(explicit(w0), w0)[0].flatten()
        oracle = torch.linalg.solve(h,b).reshape(5,3)
        assert torch.allclose(w, oracle, atol=1e-8, rtol=1e-8), (mean_only,audit)
        old = dict(mu=bank['mu'][:1],Q=bank['Q'][:1],n=[6],components=bank['components'][:2])
        explicit_old = pair_loss(native.empty(5),x,y,x.new_full((18,),1/18),w,a,mean_only)
        saved_old = pair_loss(old,x[6:],y[6:],x.new_full((12,),1/18),w,a,mean_only)
        assert torch.allclose(explicit_old,saved_old,atol=1e-10)
        q,mu,mass = moments(old,x[6:],y[6:],x.new_full((12,),1/18),3)
        inv,cross = native.proximal_base(old,torch.eye(5,dtype=x.dtype),3)
        target2 = target[6:].double(); previous = ref
        nw = native.proximal_head(x[6:],target2,x.new_full((12,),1/18),inv,cross,previous)
        pw,_ = solve(q,mu,mass,a,mean_only=mean_only,previous=previous,proximal=.01,
                    inverse=inv,x=x[6:],alpha=x.new_full((12,),1/18),native=nw)
        direct,_ = solve(q,mu,mass,a,mean_only=mean_only,previous=previous,proximal=.01)
        assert torch.allclose(pw,direct,atol=1e-8,rtol=1e-8)
    zero,_ = bank_head(bank,a,beta=0)
    assert torch.allclose(zero,ref,atol=1e-10)
    # Envelope gradient: independent finite difference with a fresh exact solve.
    def optimized(v):
        b = native.append(native.empty(5),v,y,group,weights)
        z,_ = bank_head(b,a)
        z = z.detach()
        t = torch.nn.functional.one_hot(y,3)
        return .5*(v @ z-t).square().sum(1).mean()+.0005*z.square().sum()+.5*pair_loss(native.empty(5),v,y,v.new_full((18,),1/18),z,a)
    variable=x.clone().requires_grad_(); value=optimized(variable)
    gradient=torch.autograd.grad(value,variable)[0][2,1]
    eps=1e-5;xp=x.clone();xm=x.clone();xp[2,1]+=eps;xm[2,1]-=eps
    finite=(optimized(xp)-optimized(xm))/(2*eps)
    assert torch.allclose(gradient,finite,atol=2e-6,rtol=2e-5)
    print('PASS: explicit-sample Hessian oracle, old moment identity, native reduction, Woodbury consistency, envelope gradient')


if __name__ == '__main__': self_check()
