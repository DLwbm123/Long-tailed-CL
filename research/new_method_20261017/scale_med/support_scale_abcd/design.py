"""Deterministic, group-disjoint support-only nested split and scoring helpers."""
import random
from collections import Counter
import torch

SEEDS = [74510, 74511, 74512]
TEMPERATURES = [1/16, 1/8, 1/4, 1/2, 1., 2., 4.]
CLASSES = [c for c in range(13) if c != 1]
EXPECTED = [502, 1, 4, 69, 232, 323, 590, 392, 202, 135, 477, 209, 4]

def make_plan(rows):
    groups = {}
    for i, r in enumerate(rows):
        g = r['identity_component']
        item = groups.setdefault(g, dict(label=r['label'], indices=[]))
        assert item['label'] == r['label']
        item['indices'].append(i)
    assert [sum(v['label'] == c for v in groups.values()) for c in range(13)] == EXPECTED
    assert len({r['relative_path'] for r in rows}) == len(rows) == 3140
    outer = [-1] * len(rows)
    for c in CLASSES:
        keys = sorted(g for g, v in groups.items() if v['label'] == c)
        random.Random(86420 + c).shuffle(keys)
        for j, g in enumerate(keys):
            for i in groups[g]['indices']: outer[i] = j % 4
    folds = []
    for k in range(4):
        train = [i for i, f in enumerate(outer) if f != k]
        test = [i for i, f in enumerate(outer) if f == k]
        inner_assignment = {}
        for c in CLASSES:
            keys = sorted(g for g, v in groups.items() if v['label'] == c and outer[v['indices'][0]] != k)
            random.Random(86420 + 1000*(k+1) + c).shuffle(keys)
            for j, g in enumerate(keys): inner_assignment[g] = j % 3
        inner = []
        for j in range(3):
            va = [i for i in train if inner_assignment.get(rows[i]['identity_component'], -1) == j]
            tr = [i for i in train if inner_assignment.get(rows[i]['identity_component'], -1) != j]
            inner.append(dict(train=tr, validation=va))
        folds.append(dict(train=train, test=test, inner=inner))
    plan = dict(rows=rows, outer_assignment=outer, folds=folds, split_seed=86420)
    validate(plan)
    return plan

def validate(plan):
    rows = plan['rows']
    def gs(ids): return {rows[i]['identity_component'] for i in ids}
    def labels(ids): return {rows[i]['label'] for i in ids}
    tests = []
    for f in plan['folds']:
        tr, te = f['train'], f['test']
        assert not gs(tr) & gs(te) and set(tr) | set(te) == set(range(len(rows)))
        assert labels(tr) == set(range(13)) and labels(te) == set(CLASSES)
        va_all = []
        for inner in f['inner']:
            a, b = inner['train'], inner['validation']
            assert not gs(a) & gs(b) and not (gs(a) | gs(b)) & gs(te)
            assert set(a) | set(b) == set(tr)
            assert labels(a) == set(range(13)) and labels(b) == set(CLASSES)
            va_all += b
        assert len(va_all) == len(set(va_all))
        assert set(va_all) == {i for i in tr if rows[i]['label'] != 1}
        tests += te
    assert len(tests) == len(set(tests)) == len(rows)-1
    assert set(tests) == {i for i, r in enumerate(rows) if r['label'] != 1}

def balanced_weights(y, groups):
    sizes = Counter(groups)
    nc = {c: len({g for g, v in zip(groups, y.tolist()) if v == c}) for c in CLASSES}
    assert set(y.tolist()) == set(CLASSES) and all(nc.values())
    return torch.tensor([1/(12*nc[c]*sizes[g]) for c, g in zip(y.tolist(), groups)], dtype=torch.double)

def score(prob, y, groups):
    assert prob.shape == (len(y), 13) and torch.isfinite(prob).all()
    assert float((prob.sum(1)-1).abs().max()) < 2e-6
    pred = prob.argmax(1); ok = pred == y
    n = torch.bincount(y, minlength=13); correct = torch.bincount(y[ok], minlength=13)
    recall = [None if n[c] == 0 else float(correct[c]/n[c]) for c in range(13)]
    w = balanced_weights(y, groups)
    return dict(BA=sum(recall[c] for c in CLASSES)/12, group_BA=float(w@ok.double()),
                balanced_NLL=float(w@(-prob[torch.arange(len(y)), y].double().clamp_min(1e-12).log())),
                counts=n.tolist(), correct=correct.tolist(), recall=recall,
                confusion=torch.bincount(y*13+pred, minlength=169).reshape(13,13).tolist())

def flows(ncm, adapter, candidate, y):
    n, a, m = [p.argmax(1) == y for p in [ncm, adapter, candidate]]
    return [dict(class_id=c, count=int((y==c).sum()),
                 NCM_exclusive=int((n & ~a & (y==c)).sum()),
                 NCM_exclusive_retained=int((n & ~a & m & (y==c)).sum()),
                 adapter_exclusive=int((~n & a & (y==c)).sum()),
                 adapter_exclusive_retained=int((~n & a & m & (y==c)).sum()),
                 both_wrong_rescued=int((~n & ~a & m & (y==c)).sum()),
                 rescued_vs_adapter=int((~a & m & (y==c)).sum()),
                 harmed_vs_adapter=int((a & ~m & (y==c)).sum())) for c in range(13)]

def select_temperature(ncm_logits, adapter_probs, y, groups):
    w = balanced_weights(y, groups)
    losses = []
    for t in TEMPERATURES:
        mix = .5*(adapter_probs + (ncm_logits/t).softmax(1))
        losses.append(float(w@(-mix[torch.arange(len(y)), y].double().clamp_min(1e-12).log())))
    # Fixed tie rule: prefer the original scale, then the nearest power of two.
    best = min(range(len(losses)), key=lambda i: (losses[i], abs(TEMPERATURES.index(1.)-i), i))
    return TEMPERATURES[best], losses
