"""Shared directed-edge actions with a fixed competition-budget floor."""
import math
import torch
from torch.nn import functional as F

import core1_competition as competition
import prototype_coherent as native


@torch.no_grad()
def descriptors(bank, reference, old):
    k = len(bank['n'])
    if not 0 <= old < k or k < 2:
        raise ValueError('Invalid edge context')
    labels = torch.arange(k, device=reference.device)
    historical = labels < old
    off = labels[:, None] != labels[None, :]
    groups = torch.stack([historical[:, None] & ~historical[None, :],
                          ~historical[:, None] & historical[None, :],
                          (historical[:, None] == historical[None, :]) & off], -1).to(reference)
    hardness = reference.new_zeros((k, k))
    for component in bank['components']:
        c = component['label']
        scores = component['center'] @ reference
        hardness[c] += component['mass'] * (1-scores[c]+scores).clamp_min(0).square()
    hardness.fill_diagonal_(0)
    relative = (hardness / (hardness.sum(1, keepdim=True)/(k-1)).clamp_min(1e-12)-1).tanh()
    margin, variance = [], []
    for c in range(k):
        direction = reference[:, c:c+1]-reference
        mean = bank['mu'][c] @ direction
        var = ((bank['Q'][c] @ direction)*direction).sum(0)-mean.square()
        margin.append(mean)
        variance.append(var.clamp_min(0))
    margin, variance = torch.stack(margin), torch.stack(variance)
    risk = (-margin/(variance+1e-4).sqrt()).tanh()
    centers = F.normalize(bank['mu'], dim=1)
    similarity = (centers @ centers.T).clamp(-1, 1)
    count = reference.new_tensor(bank['n'])
    scarcity = (count.log()[None, :]-count.log()[:, None]).tanh()
    dispersion = ((variance+1e-12).sqrt()/(margin.abs()+.01)).tanh()
    features = torch.cat([groups, torch.stack([relative, risk, similarity, scarcity, dispersion], -1)], -1)
    features *= off[:, :, None]
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite edge descriptors')
    return features


def allocation(base, features, coefficients):
    if (features.shape[:2] != base.shape or features.shape[-1] != len(coefficients) or
            not torch.isfinite(base).all() or not torch.isfinite(features).all() or
            not torch.isfinite(coefficients).all() or (base < 0).any() or base.sum() <= 0):
        raise ValueError('Invalid edge action')
    if bool((coefficients == 0).all()):
        return base.clone()
    support = base > 0
    tilt = math.log(2.) * (features @ coefficients).tanh()
    logits = (base.clamp_min(1e-30).log()+tilt).masked_fill(~support, -torch.inf)
    result = .5*base+.5*base.sum()*logits.flatten().softmax(0).reshape_as(base)
    if not torch.isfinite(result).all() or not torch.allclose(result.sum(), base.sum()):
        raise ValueError('Competition mass was not conserved')
    return result


@torch.no_grad()
def risks(old, head, x, y):
    target = F.one_hot(y, head.shape[1]).to(x)
    loss = .5*(x @ head-target).square().sum(1)
    current = torch.stack([loss[y == c].mean() for c in sorted(y.unique().tolist())])
    values = torch.cat([.5*native.old_square_losses(old, head), current])
    if not torch.isfinite(values).all():
        raise ValueError('Invalid edge feedback')
    return values


def reward(values, baseline, old, tail):
    gain = (baseline-values)/baseline.clamp_min(.1)
    negative = (-gain).clamp_min(0)
    penalty = negative[old:].mean()
    if old:
        penalty += negative[:old].mean()
    if tail:
        penalty += negative[tail].mean()
    return gain.mean()-penalty


def action_kl(pairs, baseline):
    mask = baseline > 0
    p, q = pairs[mask]/pairs.sum(), baseline[mask]/baseline.sum()
    return (p*(p/q).log()).sum()


class Controller:
    def __init__(self, arm, seed):
        if arm not in ('static_pc', 'coarse_rl', 'edge_rl', 'edge_search'):
            raise ValueError('Unknown edge arm')
        self.arm = arm
        self.parameter = torch.zeros(3 if arm == 'coarse_rl' else 8, dtype=torch.float64, requires_grad=True)
        self.generator = torch.Generator().manual_seed(seed+93517)
        self.optimizer = torch.optim.Adam([self.parameter], lr=.05)
        self.reference = self.parameter.detach().clone()
        self.updates = 0

    def mean(self):
        return 2.*torch.tanh(self.parameter/2.)

    def begin_task(self):
        self.reference = self.mean().detach().clone()
        self.optimizer = torch.optim.Adam([self.parameter], lr=.05)

    def choose(self, bank, old, base, metric, meta_x, meta_y, tail, budget):
        old_count = len(old['n'])
        budget()
        with torch.no_grad():
            baseline_head, solve = competition.bank_head(bank, base, regularizer=metric)
            baseline = risks(old, baseline_head, meta_x, meta_y)
            state = descriptors(bank, baseline_head, old_count)
            if self.arm == 'coarse_rl':
                state = state[:, :, :3]
        evaluations = 1
        max_residual = solve['relative_residual']

        @torch.no_grad()
        def evaluate(coefficients):
            nonlocal evaluations, max_residual
            budget()
            pairs = allocation(base, state, coefficients.to(base))
            head, audit = competition.bank_head(bank, pairs, regularizer=metric)
            value = risks(old, head, meta_x, meta_y)
            score = reward(value, baseline, old_count, tail)-.01*action_kl(pairs, base)
            if not torch.isfinite(score):
                raise ValueError('Nonfinite edge reward')
            evaluations += 1
            max_residual = max(max_residual, audit['relative_residual'])
            return float(score), pairs, head, value

        before = self.updates
        samples, losses = [], []
        mean_before = self.mean().detach().clone()
        if self.arm == 'static_pc':
            selected = torch.zeros_like(self.parameter)
            score, pairs, head, value = 0., base.clone(), baseline_head, baseline
            executed = False
        else:
            search_center = self.mean().detach().clone()
            search_best = -math.inf
            for _ in range(4):
                mean = self.mean().detach() if self.arm != 'edge_search' else search_center
                latent = mean[None, :]+.5*torch.randn((4, len(mean)), dtype=mean.dtype, generator=self.generator)
                outcomes = [evaluate(v) for v in latent]
                rewards = torch.tensor([o[0] for o in outcomes], dtype=torch.float64)
                samples.append(dict(coefficients=latent.tolist(), rewards=rewards.tolist()))
                if self.arm == 'edge_search':
                    index = int(rewards.argmax())
                    if float(rewards[index]) > search_best:
                        search_best, search_center = float(rewards[index]), latent[index].clone()
                    continue
                # Static-PC is the zero-return baseline: harmful actions keep negative advantages.
                advantage = rewards/(rewards.square().mean().sqrt()+1e-8)
                old_log = -.5*((latent-mean)/.5).square().sum(1)
                for _ in range(2):
                    now = self.mean()
                    new_log = -.5*((latent-now)/.5).square().sum(1)
                    ratio = (new_log-old_log).exp()
                    kl = ((now-self.reference).square()/(2*.5**2)).mean()
                    loss = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()+.01*kl
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite edge policy update')
                    self.optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_([self.parameter], 1., error_if_nonfinite=True)
                    self.optimizer.step(); self.updates += 1
                    losses.append(float(loss.detach()))
            proposed = search_center if self.arm == 'edge_search' else self.mean().detach()
            score, pairs, head, value = evaluate(proposed)
            executed = score > 1e-8
            selected = proposed if executed else torch.zeros_like(proposed)
            if not executed:
                pairs, head = base.clone(), baseline_head
            if self.arm == 'edge_search':
                # Search state persists just like the actor, but uses no policy gradients.
                with torch.no_grad():
                    self.parameter.copy_(2*torch.atanh((selected/2).clamp(-1+1e-8, 1-1e-8)))
        pair_change = float((pairs-base).abs().sum()/base.sum())
        audit = dict(mode=self.arm, policy_updates=self.updates-before, head_evaluations=evaluations,
            proposed_reward=score, executed=executed, selected_coefficients=selected.tolist(),
            mean_before=mean_before.tolist(), mean_after=self.mean().detach().tolist(),
            samples=samples, optimization_losses=losses,
            original_current_guard=bool((value[old_count:] <= baseline[old_count:]+1e-8).all()),
            original_old_guard=bool((value[:old_count] <= baseline[:old_count]+1e-8).all()),
            baseline_risks=baseline.tolist(), proposed_risks=value.tolist(),
            relative_pair_l1=pair_change, max_solve_residual=max_residual,
            base_pairs=base.tolist(), selected_pairs=pairs.tolist())
        return pairs, head, selected, audit
