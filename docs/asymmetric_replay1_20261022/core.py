"""Two-channel class-balanced acquisition and all-seen consolidation."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ['ER', 'ASYMMETRIC']


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



def objective(model, x, y, incoming_rows, old_count, arm, cfg):
    _, logits = model(x, cfg['logit_scale'])
    k = logits.shape[1]
    current = logits[:incoming_rows]
    targets = y[:incoming_rows]
    if arm == 'ASYMMETRIC':
        current = current[:, old_count:]
        targets = targets - old_count
    else:
        assert arm == 'ER'
    acquire = F.cross_entropy(current, targets)
    consolidate = F.cross_entropy(logits[incoming_rows:], y[incoming_rows:], reduction='none')
    weights = torch.where(y[incoming_rows:] >= old_count, .5, 1.)
    loss = ((k-old_count)/(2*k)) * acquire + (weights * consolidate).mean()
    return loss, acquire, (weights * consolidate).mean()
