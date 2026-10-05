"""Masked binary objectives; independent Bernoulli actions and FD projection."""
import math
import torch
from torch.nn import functional as F

METHODS = ('A', 'R05', 'R01', 'P01', 'EP01')


def clean_targets(y, mask):
    if y.ndim != 2 or mask.shape != y.shape or mask.dtype != torch.bool:
        raise ValueError('Targets and boolean observation mask must be B x C')
    if not torch.isfinite(y[mask]).all() or not ((y[mask] == 0) | (y[mask] == 1)).all():
        raise ValueError('Observed labels must be binary')
    return torch.where(mask, y, 0.)


def binary_weights(y, mask):
    """Balance each observed binary label from CURRENT training targets only."""
    y = clean_targets(y, mask)
    counts = torch.stack(((mask & (y == 0)).sum(0), (mask & (y == 1)).sum(0)), 1)
    groups = (counts > 0).sum(1).clamp_min(1)
    weights = counts.sum(1, keepdim=True) / (groups[:, None] * counts.clamp_min(1))
    return torch.where(counts > 0, weights, 0.).to(y.dtype).detach()


def scores(z, head):
    if head.requires_grad or head.shape[0] != z.shape[1] + 1:
        raise ValueError('Expected a detached (D+1) x C analytic head including bias')
    return z @ head[:-1] + head[-1]


def bernoulli_reward(logits, teacher_logits, y, retention_labels):
    logp = torch.stack((F.logsigmoid(-logits), F.logsigmoid(logits)), -1)
    with torch.no_grad():
        logq = torch.stack((F.logsigmoid(-teacher_logits), F.logsigmoid(teacher_logits)), -1)
        correct = F.one_hot(y.long(), 2).to(logits.dtype)
        retention = torch.exp(-20 * (logp.detach() - logq).clamp_min(0))
        reward = correct + .5 * retention * retention_labels[None, :, None]
    return logp, reward, retention


def auxiliary(logp, reward, mask, entry_weights, sigma, mode, generator=None, actions=None):
    """Exact is the expectation of RAW LOO gradients, before projection/clip."""
    zero = logp.sum() * 0
    if not mask.any():
        return zero, None
    lp, r = logp[mask], reward[mask].detach()
    w = entry_weights[mask]
    scale = max(float(sigma), .05)
    if mode == 'exact':
        p = lp.exp()
        baseline = (p.detach() * r).sum(-1, keepdim=True)
        return -(w * (p * ((r - baseline) / scale).detach()).sum(-1)).mean(), None
    if mode != 'sample':
        raise ValueError('Unknown auxiliary mode')
    if actions is None:
        if generator is None:
            raise ValueError('An isolated action generator is required')
        actions = torch.multinomial(lp.detach().exp(), 8, replacement=True, generator=generator)
    if actions.shape != (len(lp), 8):
        raise ValueError('Expected eight actions per observed image-label pair')
    sampled = r.gather(1, actions)
    advantage = ((sampled - (sampled.sum(1, keepdim=True) - sampled) / 7) / scale).detach()
    return -(w[:, None] * advantage * lp.gather(1, actions)).mean(), actions


def objective(z, target, head, y, mask, weights, method, task, sigma, generator, old_labels):
    if method not in METHODS or task < 1:
        raise ValueError('Invalid method or one-based task')
    y = clean_targets(y, mask)
    logits = scores(z, head)
    if logits.shape != y.shape or weights.shape != (y.shape[1], 2):
        raise ValueError('Head/label/weight shape mismatch')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Weights must be finite and nonnegative')
    old_labels = torch.as_tensor(old_labels, dtype=torch.bool, device=z.device)
    if old_labels.shape != (y.shape[1],):
        raise ValueError('Teacher support must have one flag per label')
    weight = weights.gather(1, y.long().T).T.detach()
    observed = mask.sum().clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * weight * mask).sum() / observed
    zero = z.sum() * 0
    row_weight = (weight * mask).sum(1) / mask.sum(1).clamp_min(1)
    active_rows = mask.any(1)
    fd = ((z - target.detach()).square().sum(1) * row_weight).sum() / active_rows.sum().clamp_min(1) if task > 1 else zero
    aux, actions, next_sigma = zero, None, float(sigma)
    coefficient = 0. if method == 'A' or task == 1 else .5 if method == 'R05' else .1
    diagnostics = dict(observed_pairs=int(mask.sum()), sigma_before=float(sigma), reward_coefficient=coefficient)
    if coefficient and mask.any():
        logp, reward, retention = bernoulli_reward(logits, scores(target.detach(), head), y, old_labels)
        aux, actions = auxiliary(logp, reward, mask, weight, sigma,
                                 'exact' if method == 'EP01' else 'sample', generator)
        with torch.no_grad():
            p, r = logp[mask].exp(), reward[mask]
            mean = (p * r).sum(1).mean()
            std = ((p * r.square()).sum(1).mean() - mean.square()).clamp_min(0).sqrt()
            next_sigma = .999 * float(sigma) + .001 * float(std)
            diagnostics.update(reward_mean=float(mean), reward_std=float(std),
                               entropy=float(-(p * logp[mask]).sum(1).mean()),
                               teacher_supported_pairs=int((mask & old_labels).sum()))
    return bce + 10 * fd + coefficient * aux, (bce, fd, aux), coefficient, next_sigma, diagnostics, actions


def dot(a, b):
    values = [(x.double() * y.double()).sum() for x, y in zip(a, b) if x is not None and y is not None]
    return float(torch.stack(values).sum()) if values else 0.


def project(aux, fd):
    dd, hd, hh = dot(fd, fd), dot(aux, fd), dot(aux, aux)
    trigger = dd > 1e-20 and hd < 0
    projected = tuple(None if h is None and d is None else
                      (0 if h is None else h) - hd / dd * (0 if d is None else d)
                      for h, d in zip(aux, fd)) if trigger else aux
    after, pp = dot(projected, fd), dot(projected, projected)
    tolerance = 1e-12 + 1e-6 * math.sqrt(dd * hh)
    if trigger and after < -tolerance:
        raise ArithmeticError('Projection numerical tolerance exceeded')
    return projected, dict(triggered=trigger, dot_before=hd, dot_after=after,
                          numerical_tolerance=tolerance,
                          remaining_ratio=math.sqrt(pp / hh) if hh else 0.,
                          near_zero=math.sqrt(pp) <= max(1e-12, .001 * math.sqrt(hh)))


def update(encoder, teacher, optimizer, x, y, mask, head, weights, method, task,
           sigma, generator, old_labels):
    """One clip and one optimizer step; entirely unobserved batches are skipped."""
    params = [p for p in encoder.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable encoder parameters')
    optimizer.zero_grad(set_to_none=True)
    if not mask.any():
        return float(sigma), dict(skipped=True, observed_pairs=0), None
    with torch.no_grad():
        target = teacher(x)
    z = encoder(x)
    total, terms, coefficient, next_sigma, diagnostics, actions = objective(
        z, target, head, y, mask, weights, method, task, sigma, generator, old_labels)
    if not torch.isfinite(total):
        raise ArithmeticError('Nonfinite loss')
    if method in ('P01', 'EP01') and task > 1:
        c = torch.autograd.grad(terms[0], params, retain_graph=True, allow_unused=True)
        d = torch.autograd.grad(terms[1], params, retain_graph=True, allow_unused=True)
        h = torch.autograd.grad(terms[2], params, allow_unused=True)
        hp, info = project(tuple(None if v is None else coefficient * v for v in h), d)
        for param, ce, fd, aux in zip(params, c, d, hp):
            param.grad = None if ce is None and fd is None and aux is None else (
                (0 if ce is None else ce) + 10 * (0 if fd is None else fd) + (0 if aux is None else aux)).detach()
        diagnostics['projection'] = info
    else:
        total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    diagnostics.update(loss=float(total.detach()), bce=float(terms[0].detach()),
                       fd=float(terms[1].detach()), auxiliary=float(terms[2].detach()),
                       pre_clip_norm=float(grad_norm), clipped=bool(grad_norm > 1),
                       sigma_after=next_sigma, clip_calls=1, optimizer_step_calls=1, skipped=False)
    return next_sigma, diagnostics, actions
