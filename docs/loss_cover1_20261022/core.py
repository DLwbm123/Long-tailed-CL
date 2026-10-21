"""Finite empirical loss covers, not a population or privacy guarantee."""
import numpy as np
import torch
from collections import Counter

ARMS = ['POINT', 'CLASS', 'CELL']


def make_cover(z, groups, cap):
    """Greedy image centers, at most one center/patient; cover every current row."""
    z = z.double().contiguous()
    assert torch.isfinite(z).all() and len(z) == len(groups)
    counts = Counter(groups)
    mass = torch.tensor([1 / counts[g] for g in groups], dtype=torch.float64)
    mass /= mass.sum()
    mean = mass @ z
    first = int(((z - mean).square().sum(1)).argmin())
    chosen, used = [], set()
    d2 = torch.full((len(z),), float('inf'), dtype=torch.float64)
    for _ in range(min(cap, len(set(groups)))):
        if not chosen:
            k = first
        else:
            available = torch.tensor([g not in used for g in groups])
            k = int(d2.masked_fill(~available, -1).argmax())
        chosen.append(k)
        used.add(groups[k])
        d2 = torch.minimum(d2, (z - z[k]).square().sum(1))
    centers = z[chosen].clone()
    assignment, distances = [], []
    for block in z.split(256):
        d = (block[:, None] - centers[None]).square().sum(2)
        value, idx = d.min(1)
        assignment.append(idx)
        distances.append(value.sqrt())
    assignment, distances = torch.cat(assignment), torch.cat(distances)
    weights = torch.zeros(len(chosen), dtype=torch.float64)
    weights.scatter_add_(0, assignment, mass)
    radii = torch.zeros(len(chosen), dtype=torch.float64)
    radii.scatter_reduce_(0, assignment, distances, reduce='amax', include_self=True)
    # Roundoff cushion is fixed, never calibrated against validation.
    radii += 1e-12
    assert abs(float(weights.sum()) - 1) < 1e-10
    return dict(centers=centers, weights=weights, radii=radii), chosen, assignment, mass


def tensors(covers, seen, arm):
    xs, ys, rs, ms = [], [], [], []
    for label, c in enumerate(seen):
        v = covers[c]
        xs.append(v['centers'])
        ys.extend([label] * len(v['weights']))
        radius = v['radii'].clone()
        if arm == 'POINT': radius.zero_()
        if arm == 'CLASS': radius.fill_(float(radius.max()))
        rs.append(radius)
        ms.append(v['weights'] / len(seen))
    return torch.cat(xs), torch.tensor(ys), torch.cat(rs), torch.cat(ms)


def objective(theta, x, y, radius, mass, cfg):
    w, b = theta[:, :-1], theta[:, -1]
    diff = w[None] - w[y, None]
    norm = (diff.square().sum(2) + cfg['norm_smoothing'] ** 2).sqrt()
    norm = norm.masked_fill(torch.arange(len(w))[None] == y[:, None], 0)
    logits = x @ w.T + b
    relative = logits - logits.gather(1, y[:, None]) + radius[:, None] * norm
    return (mass * torch.logsumexp(relative, 1)).sum() + cfg['ridge'] / 2 * theta.square().sum()


def fit(covers, seen, arm, cfg, callback):
    x, y, radius, mass = tensors(covers, seen, arm)
    theta = torch.zeros(len(seen), x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=cfg['LBFGS_max_iter'],
        max_eval=cfg['LBFGS_max_eval'], tolerance_grad=cfg['LBFGS_gradient_tolerance'],
        tolerance_change=cfg['LBFGS_change_tolerance'], line_search_fn='strong_wolfe')
    calls = 0
    def closure():
        nonlocal calls
        assert calls < cfg['LBFGS_max_eval'], 'Frozen objective evaluation cap reached'
        opt.zero_grad()
        loss = objective(theta, x, y, radius, mass, cfg)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(theta.grad).all()
        calls += 1
        callback(calls, float(loss.detach()), float(theta.grad.abs().max()), int(opt.state[theta].get('n_iter', 0)))
        return loss
    opt.step(closure)
    opt.zero_grad()
    loss = objective(theta, x, y, radius, mass, cfg)
    loss.backward()
    calls += 1  # Separately counted final acceptance evaluation after capped optimization.
    callback(calls, float(loss.detach()), float(theta.grad.abs().max()), int(opt.state[theta].get('n_iter', 0)))
    return theta.detach(), dict(evaluations=calls, updates=int(opt.state[theta].get('n_iter', 0)),
        loss=float(loss.detach()), gradient_inf=float(theta.grad.abs().max()),
        converged=bool(theta.grad.abs().max() <= cfg['accept_gradient_inf']))


def math_check(cfg):
    from verify import numpy_objective
    gen = torch.Generator().manual_seed(74711)
    theta = torch.randn(3, 5, generator=gen, dtype=torch.float64, requires_grad=True)
    x = torch.randn(7, 4, generator=gen, dtype=torch.float64)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    r = torch.linspace(.01, .3, 7, dtype=torch.float64)
    m = torch.ones(7, dtype=torch.float64) / 7
    loss = objective(theta, x, y, r, m, cfg)
    independent = numpy_objective(theta.detach().numpy(), x.numpy(), y.numpy(), r.numpy(), m.numpy(), cfg)
    assert abs(float(loss.detach()) - independent) < 1e-10
    loss.backward()
    derivative_errors = []
    for a, b in [(0, 0), (1, 4), (2, 2)]:
        t = theta.detach().numpy().copy(); h = 1e-5
        t[a, b] += h; plus = numpy_objective(t, x.numpy(), y.numpy(), r.numpy(), m.numpy(), cfg)
        t[a, b] -= 2*h; minus = numpy_objective(t, x.numpy(), y.numpy(), r.numpy(), m.numpy(), cfg)
        derivative_errors.append(abs((plus-minus)/(2*h) - float(theta.grad[a,b])))
    assert max(derivative_errors) < 1e-8
    point = objective(theta, x, y, r*0, m, cfg)
    ce = torch.nn.functional.cross_entropy(x @ theta[:, :-1].T + theta[:, -1], y)
    assert abs(float((point-ce-cfg['ridge']/2*theta.square().sum()).detach())) < 1e-10
    # Direct point perturbations must lie below the cell upper bound.
    w, b = theta[:, :-1].detach(), theta[:, -1].detach()
    upper = torch.logsumexp(x @ w.T+b-(x @ w.T+b).gather(1,y[:,None]) + r[:,None]*torch.linalg.vector_norm(w[None]-w[y,None],dim=2),1)
    violation = 0.
    for _ in range(16):
        direction = torch.randn(x.shape, generator=gen, dtype=torch.float64)
        shift = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True) * r[:,None]
        perturbed = torch.nn.functional.cross_entropy((x+shift) @ w.T+b, y, reduction='none')
        violation = max(violation, float((perturbed-upper).max()))
    assert violation <= 1e-10
    return dict(objective_error=abs(float(loss.detach())-independent),
        gradient_error=max(derivative_errors), zero_radius_is_CE=True,
        sampled_bound_violation=violation, uniform_common_intersection_warning=True)
