"""Bounded birth-function memory: ordinary replay, historical logit matching, all-seen margin preservation."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ['ER', 'LOGIT', 'MARGIN']


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


def auxiliary(logits, labels, targets, valid, arm, scale):
    old = valid.any(1)
    if arm == 'ER' or not bool(old.any()):
        return logits.sum() * 0
    if arm == 'LOGIT':
        # Equal weight per old example, independent of birth-prefix width.
        each = ((logits - targets).square() * valid).sum(1) / valid.sum(1).clamp_min(1)
        return each[old].mean() / scale**2
    assert arm == 'MARGIN'
    own = targets.gather(1, labels[:, None]).squeeze(1)
    rivals = valid.clone()
    rivals.scatter_(1, labels[:, None], False)
    birth_margin = own - targets.masked_fill(~rivals, -torch.inf).max(1).values
    # Teachers with a wrong/tied birth decision create no preservation constraint.
    eligible = old & (birth_margin > 0)
    if not bool(eligible.any()):
        return logits.sum() * 0
    not_label = torch.ones_like(logits, dtype=torch.bool)
    not_label.scatter_(1, labels[:, None], False)
    now_margin = logits.gather(1, labels[:, None]).squeeze(1) - logits.masked_fill(~not_label, -torch.inf).max(1).values
    return F.relu(birth_margin[eligible] - now_margin[eligible]).square().mean() / scale**2


def objective(model, views, labels, history, arm, cfg):
    _, logits = model(views, cfg['logit_scale'])
    ce = F.cross_entropy(logits, labels)
    aux = auxiliary(logits, labels, history['targets'], history['valid'], arm, cfg['logit_scale'])
    return ce + cfg['functional_weight'] * aux, ce, aux
