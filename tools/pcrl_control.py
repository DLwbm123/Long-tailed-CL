"""Small categorical competition scheduler; historical risk is only a proxy."""
import copy
import math
import torch
from torch.nn import functional as F


def masks(k, old, device):
    i = torch.arange(k, device=device)
    h = i < old
    off = i[:, None] != i[None, :]
    return torch.stack([h[:, None] & ~h[None, :],
                        ~h[:, None] & h[None, :],
                        (h[:, None] == h[None, :]) & off])


def allocation(base, old, action):
    if action not in range(7) or not 0 < old < len(base):
        raise ValueError('Invalid scheduling context/action')
    group = masks(len(base), old, base.device)
    mass = (group * base).sum((1, 2))
    if action == 0:
        return base.clone()
    selected = (action - 1) // 2
    if mass[selected] <= 0:
        raise ValueError('Action addresses an empty group')
    odds = torch.ones_like(mass)
    odds[selected] = 2. if action % 2 else .5
    proposed = mass * odds
    proposed *= mass.sum() / proposed.sum()
    target = .5 * mass + .5 * proposed
    ratio = torch.where(mass > 0, target / mass.clamp_min(1e-30), 0.)
    result = base * (group * ratio[:, None, None]).sum(0)
    if not torch.isfinite(result).all() or not torch.allclose(result.sum(), base.sum()):
        raise ValueError('Invalid budget allocation')
    return result


def old_risk(bank, w):
    """Mean softplus of standardized negative margins; no probability claim."""
    values = []
    for c in range(len(bank['n'])):
        d = w[:, c:c+1] - w
        mean = bank['mu'][c] @ d
        variance = ((bank['Q'][c] @ d)*d).sum(0) - mean.square()
        risk = F.softplus(-mean / (variance.clamp_min(0.) + 1e-4).sqrt()).clamp_max(20.)
        values.append(risk[torch.arange(w.shape[1], device=w.device) != c].mean())
    return torch.stack(values)


def risks(bank, w, x, y):
    new = torch.stack([F.cross_entropy(x[y == c] @ w, y[y == c])
                       for c in sorted(y.unique().tolist())])
    value = torch.cat([old_risk(bank, w), new])
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite controller risk')
    return value.detach()


def reward(value, reference, old, tail):
    gain = (reference - value) / reference.clamp_min(.1)
    penalty = gain[:old].clamp_max(0).abs().mean() + gain[old:].clamp_max(0).abs().mean()
    if tail:
        penalty += gain[tail].clamp_max(0).abs().mean()
    return float(gain.mean() - penalty)


def state(bank, w, base, risk, old, tail, progress, shift_norm):
    group = masks(len(base), old, base.device)
    mass = (group * base).sum((1, 2)) / base.sum()
    old_values = risk[:old]
    n = torch.as_tensor(bank['n'], dtype=w.dtype, device=w.device)
    uniform = (1-torch.eye(len(base), device=w.device, dtype=w.dtype))/(len(base)-1)
    # Every input is a bounded aggregate, with no class-id embedding.
    scalars = [1., old/len(base), progress, math.tanh(shift_norm),
               float((n.log().std(unbiased=False)/5).clamp_max(1)),
               float(torch.tanh(old_values.mean())), float(torch.tanh(risk[old:].mean())),
               float(torch.tanh(risk[tail].mean())) if tail else 0.,
               float(torch.tanh(old_values.max())), float((base-uniform).abs().sum()/len(base))]
    return torch.tensor(scalars + mass.tolist(), dtype=torch.float64)


class Policy:
    def __init__(self, seed):
        self.theta = torch.zeros((7, 13), dtype=torch.float64, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.theta], lr=.02)
        self.generator = torch.Generator().manual_seed(seed + 83117)
        self.updates = 0

    def probabilities(self, features, valid):
        logits = (self.theta @ features).masked_fill(~valid, -torch.inf)
        return logits.softmax(0)

    def propose(self, features, valid, greedy=False):
        p = self.probabilities(features, valid).detach()
        sampling = valid.double()/valid.sum() if greedy else p
        return torch.multinomial(sampling, 4, replacement=True, generator=self.generator), p

    def update(self, features, valid, actions, values, previous):
        rewards = torch.as_tensor(values, dtype=torch.float64)
        advantage = (rewards-rewards.mean())/rewards.std(unbiased=False).clamp_min(1e-8)
        if rewards.max()-rewards.min() <= 1e-10:
            return 0
        for _ in range(2):
            p = self.probabilities(features, valid)
            ratio = p[actions]/previous[actions].clamp_min(1e-30)
            surrogate = torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()
            # Reference is the fixed uniform policy over valid actions.
            kl = (p[valid]*(p[valid].log()+valid.sum().double().log())).sum()
            loss = -surrogate+.01*kl
            if not torch.isfinite(loss): raise ValueError('Nonfinite policy objective')
            self.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([self.theta], 1., error_if_nonfinite=True)
            self.optimizer.step(); self.updates += 1
        return 2


class Snapshot:
    """Only adapters and buffers are mutable; frozen weights stay shared."""
    def __init__(self, model, optimizer):
        from next1_support import rng_state
        self.parameters = [(p, p.detach().clone()) for p in model.parameters() if p.requires_grad]
        self.buffers = [(b, b.detach().clone()) for b in model.buffers()]
        self.optimizer = copy.deepcopy(optimizer.state_dict())
        self.rng = rng_state()
        self.modes = [(m, m.training) for m in model.modules()]

    def restore(self, optimizer):
        from next1_support import restore_rng
        with torch.no_grad():
            for live, saved in self.parameters + self.buffers:
                live.copy_(saved)
        optimizer.load_state_dict(copy.deepcopy(self.optimizer))
        optimizer.zero_grad(set_to_none=True)
        for module, training in self.modes: module.training = training
        restore_rng(self.rng)

    def verify(self, optimizer):
        if any(not torch.equal(live, saved) for live, saved in self.parameters+self.buffers):
            raise ValueError('Branch model restore failed')
        def equal(a, b):
            if torch.is_tensor(a): return torch.equal(a, b)
            if isinstance(a, dict): return a.keys()==b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a == b
        if not equal(optimizer.state_dict(), self.optimizer):
            raise ValueError('Branch optimizer restore failed')
