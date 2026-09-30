"""Frozen categorical reward objectives for NB-RL-A1; no data/model access."""
import itertools
import math
import torch
from torch.nn import functional as F


def reward_terms(logp, logq, y, retention):
    pb = logp.detach().exp()
    acc = F.one_hot(y, logp.shape[1]).to(logp.dtype)
    ret = torch.exp(-20 * (logp.detach() - logq.detach()).clamp_min(0))
    reward = acc + (0.5 * ret if retention else 0)
    mean = (pb * reward).sum(1).mean()
    var = ((pb * reward.square()).sum(1).mean() - mean.square()).clamp_min(0)
    return pb, reward.detach(), ret.detach(), var.sqrt().detach()


def objective(z, target, W, y, weights, method, task, sigma, generator):
    """One collection forward, detached rewards, one independent action draw."""
    assert method in ('S', 'H', 'K', 'G', 'R', 'E') and not W.requires_grad
    logp = (z @ W).log_softmax(1)
    with torch.no_grad():
        logq = (target @ W).log_softmax(1)
    w = weights[y]
    ce = -(w * logp.gather(1, y[:, None]).squeeze(1)).mean()
    zero = z.sum() * 0
    fd = (w * (z - target).square().sum(1)).mean() if task > 1 else zero
    kl = (w * (logp.exp() * (logp - logq)).sum(1)).mean() if task > 1 else zero
    pb, reward, ret, batch_std = reward_terms(logp, logq, y, method in ('R', 'E') and task > 1)
    scale = max(float(sigma), .05)
    pg = exact = zero
    actions = None
    diag = dict(reward_mean=float((pb * reward).sum(1).mean()), reward_std=float(batch_std),
                sigma_before=float(sigma), scale=scale, entropy=float(-(pb * logp.detach()).sum(1).mean()),
                q_min=float(logq.exp().min()), pb_min=float(pb.min()),
                ret_mean=float((pb * ret).sum(1).mean()), ret_saturated=float((pb * (ret >= 1 - 1e-7)).sum(1).mean()),
                group_all_correct=None, group_all_wrong=None, group_mixed=None, effective_groups=None)
    if method in ('G', 'R'):
        # torch.multinomial is the categorical sampler with an explicit local RNG.
        actions = torch.multinomial(pb, 8, replacement=True, generator=generator)
        r = reward.gather(1, actions)
        baseline = (r.sum(1, keepdim=True) - r) / 7
        advantage = ((r - baseline) / scale).detach()
        pg = -(w[:, None] * advantage * logp.gather(1, actions)).mean()
        hits = actions.eq(y[:, None]).sum(1)
        diag.update(group_all_correct=float((hits == 8).float().mean()),
                    group_all_wrong=float((hits == 0).float().mean()),
                    group_mixed=float(((hits > 0) & (hits < 8)).float().mean()),
                    effective_groups=float((advantage.abs().sum(1) > 1e-12).float().mean()))
    elif method == 'E':
        baseline = (pb * reward).sum(1, keepdim=True)
        exact = -(w * (logp.exp() * ((reward - baseline) / scale).detach()).sum(1)).mean()
    coefficients = dict(CE=1., FD=10. if method == 'H' else 1.,
                        KL=.1 if method == 'K' else 0., PG=.1 if method in ('G', 'R') else 0.,
                        Exact=.1 if method == 'E' else 0.)
    terms = dict(CE=ce, FD=fd, KL=kl, PG=pg, Exact=exact)
    total = sum(coefficients[k] * v for k, v in terms.items())
    next_sigma = .999 * float(sigma) + .001 * float(batch_std) if method in ('G', 'R', 'E') else float(sigma)
    return total, terms, coefficients, next_sigma, diag, actions


def selfcheck():
    errors = []
    for classes, group in [(3, 4), (2, 8)]:
        logits = torch.tensor([.2, -.4, .8][:classes], dtype=torch.float64, requires_grad=True)
        p = logits.softmax(0); pb = p.detach(); reward = torch.tensor([.1, 1.4, -.2][:classes], dtype=p.dtype)
        scale = .5; baseline = (pb * reward).sum()
        exact = -(p * (reward - baseline) / scale).sum()
        expected = torch.autograd.grad(exact, logits, retain_graph=True)[0]
        loo = torch.zeros_like(expected); centered = torch.zeros_like(expected)
        for group_actions in itertools.product(range(classes), repeat=group):
            a = torch.tensor(group_actions); r = reward[a]
            probability = pb[a].prod()
            lp = logits.log_softmax(0)[a]
            loss = -(((r - (r.sum() - r) / (group - 1)) / scale) * lp).mean()
            mean_loss = -(((r - r.mean()) / scale) * lp).mean()
            loo += probability * torch.autograd.grad(loss, logits, retain_graph=True)[0]
            centered += probability * torch.autograd.grad(mean_loss, logits, retain_graph=True)[0]
        torch.testing.assert_close(loo, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(centered, expected * (group - 1) / group, atol=1e-12, rtol=1e-12)
        errors.append(float((loo - expected).abs().max()))
    logits = torch.tensor([[.2, -.3, .7]], requires_grad=True)
    logp = logits.log_softmax(1); y = torch.tensor([1])
    _, r, ret, _ = reward_terms(logp, logp, y, True)
    assert not r.requires_grad and not ret.requires_grad
    torch.testing.assert_close(ret, torch.ones_like(ret))
    a = torch.tensor([[0, 1, 2, 1, 2, 0, 1, 0]])
    const = torch.ones_like(a, dtype=logp.dtype)
    advantage = const - (const.sum(1, keepdim=True) - const) / 7
    grad = torch.autograd.grad(-(advantage * logp.gather(1, a)).mean(), logits)[0]
    assert not grad.count_nonzero()
    return dict(status='PASS',enumerated_groups=[81,256],LOO_expectation_errors=errors,
                uncorrected_group_mean_scaling=[.75,.875],constant_reward_zero=True,
                equal_policy_retention_constant=True,reward_detached=True)


if __name__ == '__main__':
    import json
    print(json.dumps(selfcheck()))
