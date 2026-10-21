"""Frozen episodic comparator and three explicitly different risk reductions."""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ['META', 'WORST', 'PAIRED']


class Comparator(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1536, 128), nn.SiLU(), nn.Linear(128, 64))
        self.h = nn.Sequential(nn.Linear(320, 128), nn.SiLU(), nn.Linear(128, 1))

    def embed(self, x):
        return F.normalize(self.phi(F.normalize(x, dim=-1, eps=1e-12)), dim=-1, eps=1e-12)

    def pairs(self, z, means, variances):
        b, k = len(z), len(means)
        q = z[:, None, None, :].expand(b, k, k, 64)
        c = torch.cat((means, variances), -1)[None, :, None, :].expand(b, k, k, 128)
        d = torch.cat((means, variances), -1)[None, None, :, :].expand(b, k, k, 128)
        h = self.h(torch.cat((q, c, d), -1)).squeeze(-1)
        return torch.tanh(h - h.transpose(1, 2))

    def scores(self, z, means, variances):
        assert len(means) > 1
        return self.pairs(z, means, variances).sum(-1) * (8.0 / (len(means) - 1))


def risks(pair, labels):
    """Each class contributes once after averaging its available pseudo prefixes."""
    values = [[] for _ in range(10)]
    for k in (4, 6, 8, 10):
        active = labels < k
        score = pair[active, :k, :k].sum(-1) * (8.0 / (k - 1))
        loss = F.cross_entropy(score, labels[active], reduction='none')
        for c in range(k):
            values[c].append(loss[labels[active] == c].mean())
    return torch.stack([torch.stack(v).mean() for v in values])


def objective(risk, arm):
    if arm == 'META':
        return risk.mean()
    if arm == 'WORST':
        return risk.mean(0).max()
    assert arm == 'PAIRED'
    return risk.max()


def patient_pools(rows, order, seed):
    """Global patient partition, including cross-class patient membership."""
    by = {c: {} for c in order}
    for i, row in enumerate(rows):
        by[row['label']].setdefault(row['group'], []).append(i)
    key = lambda g: ((int(g) * 1103515245 + seed) % 2147483648, int(g))
    query = set()
    memberships = {}
    for c in order:
        for g in by[c]:
            memberships.setdefault(g, []).append(c)
    remaining = {c: len(by[c]) for c in order}
    query_count = {c: 0 for c in order}

    def movable(g):
        return g not in query and all(remaining[d] > 1 for d in memberships[g])

    def move(g):
        query.add(g)
        for d in memberships[g]:
            remaining[d] -= 1
            query_count[d] += 1

    for c in sorted(order, key=lambda c: len(by[c])):
        if query_count[c]:
            continue
        candidates = sorted(by[c], key=key)
        g = next((g for g in candidates if movable(g)), None)
        assert g is not None, 'No patient-disjoint base support/query partition'
        move(g)
    for g in sorted(memberships, key=key):
        if key(g)[0] % 5 == 0 and movable(g):
            move(g)
    support_by = {c: {g: ids for g, ids in by[c].items() if g not in query} for c in order}
    query_by = {c: {g: ids for g, ids in by[c].items() if g in query} for c in order}
    assert all(support_by[c] and query_by[c] for c in order)
    sp = set().union(*(set(v) for v in support_by.values()))
    qp = set().union(*(set(v) for v in query_by.values()))
    assert not sp & qp
    return support_by, query_by


def episode(support, query, base_order, cfg, index, qualification=False):
    import random
    rng = random.Random(cfg['seed'] + 100003 * index)
    classes = list(base_order)
    rng.shuffle(classes)
    si, qi, y = [], [], []
    for c in classes:
        gs = sorted(support[c], key=int)
        gq = sorted(query[c], key=int)
        rng.shuffle(gs)
        rng.shuffle(gq)
        selected = gs[:1 if qualification else min(4, len(gs))]
        si.append([rng.choice(support[c][g]) for g in selected])
        for g in gq[:1 if qualification else min(2, len(gq))]:
            qi.append(rng.choice(query[c][g]))
            y.append(len(si) - 1)
    views = []
    for pattern in cfg['support_patterns']:
        views.append([ids[:min(len(ids), pattern[c])] for c, ids in enumerate(si)])
    return dict(classes=classes, support=si, views=views, query=qi, labels=y)


def episode_risk(model, features, ep):
    # Reuse embeddings, not training examples across actual stages. All arms do identical work.
    support_ids = list(dict.fromkeys(i for ids in ep['support'] for i in ids))
    selected = support_ids + ep['query']
    mapped = model.embed(features[selected].cuda())
    lookup = {j: i for i, j in enumerate(support_ids)}
    zq = mapped[len(support_ids):]
    labels = torch.tensor(ep['labels'], device=zq.device)
    all_risks = []
    for view in ep['views']:
        states = [mapped[[lookup[i] for i in ids]] for ids in view]
        mean = torch.stack([v.mean(0) for v in states])
        var = torch.stack([(v.square().mean(0) - v.mean(0).square()).clamp_min(0) for v in states])
        all_risks.append(risks(model.pairs(zq, mean, var), labels))
    return torch.stack(all_risks), len(selected), sum(len(v) for view in ep['views'] for v in view), len(ep['query'])


def math_check():
    torch.manual_seed(17)
    m = Comparator().double()
    z = m.embed(torch.randn(3, 1536, dtype=torch.float64))
    means = torch.randn(4, 64, dtype=torch.float64)
    var = torch.rand(4, 64, dtype=torch.float64)
    p = m.pairs(z, means, var)
    assert (p + p.transpose(1, 2)).abs().max() == 0
    perm = torch.tensor([2, 0, 3, 1])
    error = (m.scores(z, means[perm], var[perm]) - m.scores(z, means, var)[:, perm]).abs().max()
    assert error < 1e-12
    extended = m.pairs(z, torch.cat((means, means * 7)), torch.cat((var, var)))
    assert (p - extended[:, :4, :4]).abs().max() < 1e-12
    a = torch.tensor([[5., 1.], [1., 4.], [2., 2.]], requires_grad=True)
    vals = [objective(a, arm) for arm in ARMS]
    grads = [torch.autograd.grad(v, a, retain_graph=True)[0] for v in vals]
    assert vals[0] < vals[1] < vals[2]
    assert not torch.equal(grads[0], grads[1]) and not torch.equal(grads[1], grads[2])
    return dict(antisymmetry=True, permutation_error=float(error.detach()), prefix_isolation=True,
                risk_values=[float(v.detach()) for v in vals], distinct_risk_gradients=True,
                parameters=sum(v.numel() for v in m.parameters()))
