"""Equal growing capacity with/without old-function freezing.

The structural premise is established by DER (arXiv:2103.16788v1, section3.2).
This feature-level ablation omits DER's pruning, auxiliary classifier and refit;
it is not a faithful reproduction or a novel algorithm claim.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ['ALL_TRAINABLE', 'FROZEN_PAST']


class Branch(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.down = nn.Linear(dimension, rank)
        self.up = nn.Linear(rank, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return F.normalize(x + self.up(F.silu(self.down(x))), dim=1)


class Model(nn.Module):
    def __init__(self, dimension, rank, means):
        super().__init__()
        self.branches = nn.ModuleList([Branch(dimension, rank)])
        self.head = nn.Parameter(F.normalize(means.clone(), dim=1))

    def embedding(self, x):
        x = F.normalize(x, dim=1)
        return F.normalize(torch.cat([branch(x) for branch in self.branches], dim=1), dim=1)

    def append_branch(self, freeze_past):
        # Copy the most recent trained branch, not future data or a future model.
        self.branches.append(copy.deepcopy(self.branches[-1]))
        for i, branch in enumerate(self.branches):
            branch.requires_grad_(not freeze_past or i == len(self.branches)-1)
        extra = self.branches[-1].up.out_features
        self.head = nn.Parameter(torch.cat([self.head.detach(), self.head.new_zeros(len(self.head), extra)], dim=1))

    def append_class_rows(self, current_patient_equal_embedded_means):
        self.head = nn.Parameter(torch.cat([self.head.detach(),
              F.normalize(current_patient_equal_embedded_means, dim=1)], dim=0))

    def forward(self, x, scale):
        z = self.embedding(x)
        return z, scale * z @ F.normalize(self.head, dim=1).T


def objective(model, views, labels, identities, arm, cfg):
    _, logits = model(views, cfg['logit_scale'])
    loss = F.cross_entropy(logits, labels)
    return loss, loss, loss * 0


import hashlib


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
