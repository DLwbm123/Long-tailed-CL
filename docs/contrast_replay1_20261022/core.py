"""Actual-stage discriminative feature replay; established contrastive objectives."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ['ER', 'INSTANCE', 'SUPERVISED']


class Model(nn.Module):
    def __init__(self, dimension, rank, means):
        super().__init__()
        self.down = nn.Linear(dimension, rank)
        self.up = nn.Linear(rank, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.head = nn.Parameter(F.normalize(means.clone(), dim=1))

    def expand(self, means):
        self.head = nn.Parameter(torch.cat([self.head.detach(), F.normalize(means, dim=1)]))

    def forward(self, x, scale):
        x = F.normalize(x, dim=1)
        z = F.normalize(x + self.up(F.silu(self.down(x))), dim=1)
        return z, scale * z @ F.normalize(self.head, dim=1).T


def contrastive(z, labels, identities, mode, temperature):
    """All non-self rows in denominator; mean log probability over all positives."""
    n = len(z)
    not_self = ~torch.eye(n, dtype=torch.bool, device=z.device)
    value = z @ z.T / temperature
    logprob = value - torch.logsumexp(value.masked_fill(~not_self, -torch.inf), dim=1)[:, None]
    key = labels if mode == 'SUPERVISED' else identities
    positive = (key[:, None] == key[None]) & not_self
    assert bool((positive.sum(1) > 0).all())
    return -(logprob.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()


def objective(model, views, labels, identities, arm, cfg):
    z, logits = model(views, cfg['logit_scale'])
    ce = F.cross_entropy(logits, labels)
    auxiliary = z.sum() * 0
    if arm != 'ER':
        auxiliary = contrastive(z, labels, identities, arm, cfg['contrastive_temperature'])
    return ce + cfg['contrastive_weight'] * auxiliary, ce, auxiliary


def key(seed, label, group, path=''):
    return hashlib.sha256(f'{seed}:{label}:{group}:{path}'.encode()).digest()


def memory_indices(rows, cap, seed):
    """One image per selected patient, selection fixed by metadata before training."""
    labels = sorted({r['label'] for r in rows})
    chosen = []
    for c in labels:
        patients = {}
        for i, r in enumerate(rows):
            if r['label'] == c:
                patients.setdefault(r['group'], []).append(i)
        for g in sorted(patients, key=lambda g: (key(seed, c, g), g))[:cap]:
            chosen.append(min(patients[g], key=lambda i: (key(seed, c, g, rows[i]['relative_path']), i)))
    return chosen
