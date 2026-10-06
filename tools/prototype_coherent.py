"""Discriminative prototype evidence with coherent class moments and proximal heads.

Prototype similarities are linear in the already unit-normalized encoder output.
The expanded evidence head is folded into an equivalent prototype metric. This
is a learned linear prior, not a nonlinear mixture classifier or a GAT.
"""
import math

import torch
from torch.nn import functional as F

from prototype_graph import partition


def empty(dim, device='cpu'):
    return dict(mu=torch.empty((0, dim), dtype=torch.float64, device=device),
                Q=torch.empty((0, dim, dim), dtype=torch.float64, device=device),
                n=[], components=[])


def seed_components(x, y):
    seeds = []
    for label in sorted(y.unique().tolist()):
        rows = torch.nonzero(y == label).flatten()
        for indices in partition(x[rows].detach().cpu().numpy()):
            sample = x[rows[torch.as_tensor(indices, device=x.device)]]
            center = sample.mean(0)
            seeds.append(dict(label=label, center=center,
                              radius=(sample-center).square().sum(1).mean()))
    return seeds


def memberships(x, y, seeds):
    group = torch.empty(len(x), dtype=torch.long, device=x.device)
    difficulty = torch.empty(len(x), dtype=x.dtype, device=x.device)
    for label in sorted(y.unique().tolist()):
        rows = torch.nonzero(y == label).flatten()
        ids = [i for i, c in enumerate(seeds) if c['label'] == label]
        if not ids:
            raise ValueError('Class has no prototype seed')
        centers = torch.stack([seeds[i]['center'] for i in ids])
        distance = (x[rows, None, :] - centers[None, :, :]).square().sum(2)
        nearest = distance.argmin(1)
        chosen = torch.as_tensor(ids, device=x.device)[nearest]
        radius = torch.stack([seeds[i]['radius'] for i in ids])[nearest]
        group[rows] = chosen
        difficulty[rows] = (distance.gather(1, nearest[:, None]).flatten()
                            / radius.clamp_min(1e-12)).clamp(0., 2.) - 1.
    return group, difficulty


def sample_weights(y, group, difficulty, actions):
    if group.shape != y.shape or difficulty.shape != y.shape or not len(actions):
        raise ValueError('Invalid weighting inputs')
    if not torch.isfinite(actions).all() or (actions.abs() > 1.+1e-10).any():
        raise ValueError('Actions must be finite and in [-1, 1]')
    result = torch.zeros_like(difficulty)
    for label in sorted(y.unique().tolist()):
        rows = y == label
        logits = math.log(2.) * actions[group[rows]] * difficulty[rows]
        # Half of the class mass always stays uniform, including rare modes.
        result[rows] = .5 / int(rows.sum()) + .5 * logits.softmax(0)
    return result


def append(bank, x, y, group, weights):
    old = len(bank['n'])
    classes = sorted(y.unique().tolist())
    if classes != list(range(old, old+len(classes))) or not len(x):
        raise ValueError('Only new contiguous classes can be appended')
    if not torch.isfinite(x).all() or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid features or weights')
    means, seconds, counts, components = [], [], [], list(bank['components'])
    for label in classes:
        rows = y == label; z, w = x[rows], weights[rows]
        if not torch.isclose(w.sum(), w.new_tensor(1.), atol=1e-8, rtol=1e-8):
            raise ValueError('Each class must have unit mass')
        means.append((w[:, None]*z).sum(0))
        seconds.append(z.T @ (w[:, None]*z)); counts.append(len(z))
        for index in sorted(group[rows].unique().tolist()):
            selected = rows & (group == index); a, v = weights[selected], x[selected]
            mass = a.sum(); center = (a[:, None]*v).sum(0)/mass
            components.append(dict(label=label, count=len(v), mass=mass,
                center=center, radius=(a*(v-center).square().sum(1)).sum()/mass))
    return dict(mu=torch.cat([bank['mu'], torch.stack(means)]),
                Q=torch.cat([bank['Q'], torch.stack(seconds)]),
                n=bank['n']+counts, components=components)


def translate(bank, shift):
    mu = bank['mu']
    if shift.shape != (mu.shape[1],) or not torch.isfinite(shift).all():
        raise ValueError('Invalid common translation')
    Q = (bank['Q'] + mu[:, :, None]*shift[None, None, :]
         + shift[None, :, None]*mu[:, None, :] + torch.outer(shift, shift)[None])
    return dict(mu=mu+shift, Q=Q, n=list(bank['n']),
                components=[dict(c, center=c['center']+shift) for c in bank['components']])


def evidence(bank, strength=1.):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Invalid prototype strength')
    dim = bank['mu'].shape[1]
    if not bank['components'] or not strength:
        return bank['mu'].new_empty((dim, 0))
    return strength * torch.stack([
        F.normalize(c['center'], dim=0)*c['mass'].sqrt()
        for c in bank['components']], dim=1)


def metric(bank, strength=1.):
    P = evidence(bank, strength)
    I = torch.eye(bank['mu'].shape[1], dtype=bank['mu'].dtype, device=bank['mu'].device)
    if not P.shape[1]:
        return I
    small = torch.eye(P.shape[1], dtype=P.dtype, device=P.device) + P.T @ P
    R = I-P @ torch.linalg.solve(small, P.T)
    return (R+R.T)/2


def head(bank, strength=1., regularization=.001):
    if not len(bank['n']) or regularization <= 0:
        raise ValueError('Nonempty memory and positive ridge penalty required')
    R = metric(bank, strength)
    A = bank['Q'].sum(0)/len(bank['n']) + regularization*R
    B = bank['mu'].T/len(bank['n'])
    W = torch.linalg.solve(A, B)
    if not torch.isfinite(W).all():
        raise ValueError('Nonfinite analytic head')
    return W, R


def proximal_base(old, R, classes, proximal=.01, regularization=.001):
    if proximal <= 0 or classes < len(old['n']):
        raise ValueError('Invalid proximal objective')
    A = old['Q'].sum(0)/classes + regularization*R
    A = A + proximal*torch.eye(len(R), dtype=R.dtype, device=R.device)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky((A+A.T)/2))
    B = R.new_zeros((len(R), classes)); B[:, :len(old['n'])] = old['mu'].T/classes
    return inverse, B


def proximal_head(x, target, weights, inverse, cross, previous, proximal=.01):
    if (weights <= 0).any() or not torch.isfinite(x).all():
        raise ValueError('Invalid minibatch objective')
    B = cross + x.T @ (weights[:, None]*target) + proximal*previous
    W = inverse @ B
    U = x.T*weights.sqrt()[None, :]
    inverse_U = inverse @ U
    small = torch.eye(len(x), dtype=x.dtype, device=x.device) + U.T @ inverse_U
    W = W-inverse_U @ torch.linalg.solve(small, U.T @ W)
    if not torch.isfinite(W).all():
        raise ValueError('Nonfinite proximal head')
    return W


def class_ce(x, y, W):
    loss = F.cross_entropy(x @ W, y, reduction='none')
    return torch.stack([loss[y == c].mean() for c in sorted(y.unique().tolist())])


def old_square_losses(bank, W):
    if not len(bank['n']):
        return W.new_empty(0)
    value = torch.einsum('dk,cde,ek->c', W, bank['Q'], W)
    value = value - 2.*(bank['mu']*W[:, :len(bank['n'])].T).sum(1) + 1.
    if not torch.isfinite(value).all() or (value.detach() < -1e-6).any():
        raise ValueError('Invalid historical moment loss')
    return value


def effective_samples(y, weights):
    return torch.stack([1./weights[y == c].square().sum() for c in sorted(y.unique().tolist())])


def weight_kl(y, weights):
    return torch.stack([(weights[y == c]*(weights[y == c]*int((y == c).sum())).log()).sum()
                        for c in sorted(y.unique().tolist())]).mean()


def descriptors(x, y, group, seeds, W):
    scores = x @ W
    margin = scores.gather(1, y[:, None]).flatten() - scores.masked_fill(
        F.one_hot(y, W.shape[1]).bool(), -torch.inf).max(1).values
    rows = []
    for k, seed in enumerate(seeds):
        selected = group == k; label = seed['label']
        if not selected.any():
            raise ValueError('Empty current component')
        n, total = int(selected.sum()), int((y == label).sum())
        center = x[selected].mean(0)
        radius = (x[selected]-center).square().sum(1).mean()
        rows.append(torch.stack([x.new_tensor(1.), x.new_tensor(math.log1p(total)/10.),
            x.new_tensor(math.log1p(n)/10.), x.new_tensor(n/total), radius.sqrt(),
            margin[selected].mean().tanh()]))
    return torch.stack(rows).detach()


def policy_mean(parameter, state):
    return 2.*torch.tanh((state @ parameter)/2.)


def advantages(rewards):
    spread = rewards.std(unbiased=False)
    if spread < 1e-8:
        return torch.zeros_like(rewards), False
    return (rewards-rewards.mean())/(spread+1e-6), True


def reward_change(ce, base_ce, old_loss, base_old, mode):
    current = base_ce-ce
    if mode == 'minimum':
        changes = torch.cat([current, base_old-old_loss])
        return changes.min()
    if mode != 'mean':
        raise ValueError('Unknown reward aggregation')
    penalty = (old_loss-base_old).relu().mean() if len(base_old) else ce.new_zeros(())
    return current.mean()-penalty


def controller(old, x, y, group, difficulty, calibration_x, calibration_y,
               seeds, parameter, reference, mode, generator, steps=8, reward_mode='mean'):
    if mode not in ('gradient', 'group') or steps < 2 or steps % 2:
        raise ValueError('Invalid controller configuration')
    zeros = x.new_zeros(len(seeds))
    neutral_weights = sample_weights(y, group, difficulty, zeros)
    neutral_bank = append(old, x, y, group, neutral_weights)
    neutral_head, _ = head(neutral_bank)
    base_ce = class_ce(calibration_x, calibration_y, neutral_head).detach()
    base_old = old_square_losses(old, neutral_head).detach()
    state = descriptors(x, y, group, seeds, neutral_head).detach()
    reference_mean = policy_mean(reference, state).detach()

    def evaluate(actions):
        weights = sample_weights(y, group, difficulty, actions)
        bank = append(old, x, y, group, weights)
        W, _ = head(bank)
        ce = class_ce(calibration_x, calibration_y, W)
        old_loss = old_square_losses(old, W)
        reward = reward_change(ce, base_ce, old_loss, base_old, reward_mode)-.01*weight_kl(y, weights)
        return reward, ce, old_loss, weights

    optimizer = torch.optim.Adam([parameter], lr=.05)
    updates, reward_spreads, clipping, losses = 0, [], [], []
    sigma = .5
    if mode == 'gradient':
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            mean = policy_mean(parameter, state)
            reward, _, _, _ = evaluate(mean.tanh())
            kl = ((mean-reference_mean).square()/(2.*sigma**2)).sum()
            loss = -reward + .01*kl
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite direct controller objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
            optimizer.step(); updates += 1; losses.append(float(loss.detach()))
    else:
        for _ in range(steps//2):
            old_mean = policy_mean(parameter, state).detach()
            latent = old_mean[None, :] + sigma*torch.randn(
                (4, len(seeds)), dtype=x.dtype, device=x.device, generator=generator)
            with torch.no_grad():
                rewards = torch.stack([evaluate(u.tanh())[0] for u in latent])
                A, active = advantages(rewards)
                old_log = -.5*((latent-old_mean)/sigma).square().sum(1)
            reward_spreads.append(float(rewards.std(unbiased=False)))
            if not active:
                continue
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                mean = policy_mean(parameter, state)
                new_log = -.5*((latent-mean)/sigma).square().sum(1)
                ratio = (new_log-old_log).exp()
                kl = ((mean-reference_mean).square()/(2.*sigma**2)).sum()
                loss = -torch.minimum(ratio*A, ratio.clamp(.8, 1.2)*A).mean() + .01*kl
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite group controller objective')
                loss.backward(); torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1; losses.append(float(loss.detach()))
                clipping.append(float(((ratio.detach() < .8) | (ratio.detach() > 1.2)).double().mean()))
    with torch.no_grad():
        proposed = policy_mean(parameter, state).tanh()
        reward, ce, old_loss, weights = evaluate(proposed)
        ess = effective_samples(y, weights)
        counts = x.new_tensor([int((y == c).sum()) for c in sorted(y.unique().tolist())])
        current_pass = bool((ce <= base_ce+1e-8).all())
        old_pass = bool((old_loss <= base_old+1e-8).all())
        ess_pass = bool((ess >= .5*counts).all())
        accepted = current_pass and old_pass and ess_pass
        selected = proposed if accepted else zeros
    audit = dict(mode=mode, reward_mode=reward_mode, optimizer_steps=updates, group_reward_std=reward_spreads,
        clip_fraction=clipping, optimization_losses=losses, proposed_actions=proposed.tolist(),
        selected_actions=selected.tolist(), proposed_reward=float(reward), accepted=accepted,
        current_guard=current_pass, old_moment_guard=old_pass, effective_sample_guard=ess_pass,
        calibration_class_ce=ce.tolist(), neutral_class_ce=base_ce.tolist(),
        historical_class_square_loss=old_loss.tolist(), neutral_historical_loss=base_old.tolist(),
        class_effective_samples=ess.tolist(), class_fit_n=counts.tolist(),
        parameter_norm=float(parameter.detach().norm()))
    return selected, audit
