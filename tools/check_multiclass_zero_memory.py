"""Shared-zero multiclass CE memory, with the frozen low-rank sequence and budget."""
import math
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import check_shared_zero_memory as shared
from check_lowrank_noise_memory import fixture_binding
from check_progressive_decision_memory import DT, project_ball

BETA = 8.
RADIUS = .125
ORIGINAL_TRAIN = shared.train_head


def exact(x, w, label, weights=None):
    p = torch.full((len(x),), 1 / len(x), dtype=DT) if weights is None else weights
    logits = BETA * x @ w.T
    value = (p * (logits.logsumexp(1) - logits[:, label])).sum()
    grad = BETA * logits.softmax(1).T @ (p[:, None] * x)
    grad[label] -= BETA * (p[:, None] * x).sum(0)
    return value, grad


def quadratic(mu, u, ev, rho, w, label):
    assert bool((w.norm(dim=1) <= RADIUS + 1e-12).all()), 'Shared head outside local domain'
    k = len(w)
    centered = w - w.mean(0)
    proj = centered @ u
    value = math.log(k) - BETA * centered[label] @ mu + BETA**2 / (2*k) * (proj.square() * ev).sum()
    grad = (BETA/k) * mu.expand_as(w) + (BETA**2/k) * (proj * ev) @ u.T
    grad = grad.clone()
    grad[label] -= BETA * mu
    norms = centered.norm(dim=1)
    radius = norms.max()
    # Cumulant range and softmax second-derivative bounds, not a tight error estimate.
    loss_bound = BETA**2/(2*k) * rho * norms.square().sum() + (4/3)*BETA**3*radius**3
    grad_bounds = BETA**2/k * rho * norms + 4*BETA**3*radius**2
    return value, grad, loss_bound, grad_bounds


def train_with_diagnostics(w, current, old, mode, steps, oracle=None):
    w, row = ORIGINAL_TRAIN(w, current, old, mode, steps, oracle=oracle)
    value = torch.zeros((), dtype=DT)
    gradient = torch.zeros_like(w)
    if oracle is not None:
        for j, x in oracle.items():
            a, g = exact(x, w, j)
            value += a / len(w)
            gradient += g / len(w)
    else:
        for j, x in current.items():
            a, g = exact(x, w, j)
            value += a / len(w)
            gradient += g / len(w)
        for j, memory in old.items():
            if mode == 'moment_local':
                x, p = memory['moment']
                a, g = exact(x, w, j, p)
            else:
                a, g, _, _ = quadratic(*memory['quadratic'], w, j)
            value += a / len(w)
            gradient += g / len(w)
    radius = 2. if mode == 'oracle_wide' else RADIUS
    # Read-only projected-gradient mapping at the last iterate; no extra update.
    residual = 16 * (w - project_ball(w - gradient / 16, radius)).norm()
    row.update(final_training_objective=float(value), projected_gradient_mapping_norm=float(residual),
               diagnostic_risk_gradient_evaluations=1)
    assert torch.isfinite(value) and torch.isfinite(residual)
    return w, row


@contextmanager
def loss_binding():
    old = shared.exact, shared.quadratic, shared.train_head
    assert shared.train_head is ORIGINAL_TRAIN
    shared.exact, shared.quadratic, shared.train_head = exact, quadratic, train_with_diagnostics
    try:
        yield
    finally:
        shared.exact, shared.quadratic, shared.train_head = old


def unit_checks():
    gen = torch.Generator().manual_seed(339)
    x = F.normalize(torch.randn(19, 6, generator=gen, dtype=DT), dim=1)
    w = project_ball(torch.randn(3, 6, generator=gen, dtype=DT), .04)
    mu = x.mean(0)
    ev, u = torch.linalg.eigh(x.T @ x / len(x))
    rho = ev[-3]
    u, ev = u[:, -2:], ev[-2:]
    value, grad = exact(x, w, 1)
    param = w.clone().requires_grad_()
    expected = F.cross_entropy(BETA*x@param.T, torch.ones(len(x), dtype=torch.long))
    autograd = torch.autograd.grad(expected, param)[0]
    exact_error = float((autograd-grad).abs().max())
    assert exact_error < 1e-12 and abs(float(value-expected)) < 1e-12
    q, g, bound, gb = quadratic(mu, u, ev, rho, w, 1)
    assert abs(float(q-value)) <= float(bound)+1e-10
    assert bool(((g-grad).norm(dim=1) <= gb+1e-10).all())
    direction = F.normalize(torch.randn(w.numel(), generator=gen, dtype=DT), dim=0).reshape_as(w)
    eps = 1e-6
    fd = abs(float((quadratic(mu,u,ev,rho,w+eps*direction,1)[0]-quadratic(mu,u,ev,rho,w-eps*direction,1)[0])/(2*eps)-(g*direction).sum()))
    assert fd < 1e-7
    shift = F.normalize(torch.randn(6, generator=gen, dtype=DT), dim=0)*.02
    shifted_value, shifted_grad = exact(x,w+shift,1)
    shifted_q, shifted_g, _, _ = quadratic(mu,u,ev,rho,w+shift,1)
    shift_error = max(abs(float(shifted_value-value)), float((shifted_grad-grad).abs().max()),
                      abs(float(shifted_q-q)), float((shifted_g-g).abs().max()))
    assert shift_error < 1e-12
    perm = torch.tensor([2,0,1]); label = int((perm==1).nonzero().item())
    pv, pg = exact(x,w[perm],label); pq, qg, _, _ = quadratic(mu,u,ev,rho,w[perm],label)
    assert abs(float(pv-value)) < 1e-12 and abs(float(pq-q)) < 1e-12
    assert float((pg-grad[perm]).abs().max()) < 1e-12 and float((qg-g[perm]).abs().max()) < 1e-12
    zero = torch.zeros_like(w)
    zv,zg = exact(x,zero,1);zq,zqg,_,_ = quadratic(mu,u,ev,rho,zero,1)
    assert abs(float(zv-zq)) < 1e-12 and float((zg-zqg).abs().max()) < 1e-12
    original = shared.exact
    try:
        with loss_binding():
            raise ValueError('intentional restoration check')
    except ValueError:
        pass
    assert shared.exact is original and shared.train_head is ORIGINAL_TRAIN
    rejected = False
    try:
        quadratic(mu,u,ev,rho,torch.ones_like(w),1)
    except AssertionError:
        rejected = True
    assert rejected
    current = {0:x[:9],1:x[9:]}; initial = torch.zeros(2,6,dtype=DT)
    with loss_binding():
        a, _ = shared.train_head(initial,current,{},'oracle_local',1,oracle=current)
        b, _ = shared.train_head(initial,current,{},'quadratic_local',1)
    assert torch.equal(a,b)
    return dict(status='PASS', exact_gradient_error=exact_error, quadratic_gradient_fd_error=fd,
                common_shift_error=shift_error, class_permutation_pass=True, zero_anchor_exact=True,
                positive_omitted_spectrum_bound_pass=True, domain_rejection=True,
                binding_restored=True, first_step_without_memory_exact=True, synthetic_head_updates=2)


def run(cfg, lock, journal):
    torch.set_num_threads(lock['threads']); torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    if cfg['kind'] == 'unit':
        result = unit_checks()
    else:
        assert cfg['active_dimension'] == 2*cfg['classes'] <= cfg['dimension']
        with fixture_binding(), loss_binding():
            result = shared.run_scale(cfg,lock,journal)
        result.update(fixture_active_dimension=cfg['active_dimension'], head_loss='multiclass_softmax_CE',
                      scale_diagnostic_risk_gradient_evaluations=4*len(cfg['task_sizes']))
    result.update(id=cfg['id'], GPU_initialized=torch.cuda.is_initialized(), medical_images=0,
                  medical_checkpoints=0, val_accessed=False, test_accessed=False, formal_updates=0)
    assert not result['GPU_initialized'] and shared.train_head is ORIGINAL_TRAIN
    return result
