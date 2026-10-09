"""Fixed identity partitions and score-distribution diagnostics, not a learner."""
import math
import random

import torch


def split_identities(rows, seed=93003):
    by_identity = {}
    for row in rows:
        by_identity.setdefault(row['identity_component'], set()).add(row['label'])
    assert all(len(v) == 1 for v in by_identity.values()), 'Cross-class identities require a new split design'
    held, covered, support = set(), [], {}
    for label in sorted({r['label'] for r in rows}):
        ids = sorted(k for k, v in by_identity.items() if label in v)
        random.Random(seed+label*2003).shuffle(ids)
        b = set(ids[:len(ids)//2]); held.update(b)
        support[str(label)] = dict(identities=len(ids), A_identities=len(ids)-len(b), B_identities=len(b),
                                  covered=len(ids) >= 4)
        if len(ids) >= 4:
            covered.append(label)
    a = [i for i, r in enumerate(rows) if r['label'] in covered and r['identity_component'] not in held]
    b = [i for i, r in enumerate(rows) if r['label'] in covered and r['identity_component'] in held]
    assert not {rows[i]['identity_component'] for i in a} & {rows[i]['identity_component'] for i in b}
    return a, b, covered, support


def score_factor(scores):
    n, heads, classes = scores.shape
    flat = scores.double().reshape(n, heads*classes); center = flat.mean(0)
    centered = flat-center
    if n <= heads*classes:
        return center, centered/math.sqrt(n), None
    covariance = centered.T @ centered/n
    values, vectors = torch.linalg.eigh((covariance+covariance.T)/2)
    assert float(values.min()) >= -1e-8, 'Score covariance not numerically PSD'
    return center, (vectors*values.clamp_min(0).sqrt()).T, float(values.min())


@torch.no_grad()
def gaussian_scores(scores, target, seed, budget, batches=8, samples=4096):
    assert scores.ndim == 3 and len(scores) >= 2 and torch.isfinite(scores).all()
    center, factor, minimum = score_factor(scores)
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    values = []
    for _ in range(batches):
        budget()
        noise = torch.randn(samples, len(factor), dtype=torch.float64, device=scores.device, generator=generator)
        generated = (noise @ factor+center).reshape(samples, scores.shape[1], scores.shape[2])
        values.append((generated.argmax(2) == target).double().mean(0))
    return torch.stack(values), dict(images=len(scores), factor_dimension=len(factor), minimum_eigenvalue=minimum)


def choose(values, seen, current, tail, candidates):
    """All classes in values are covered; numerical SE is not population uncertainty."""
    groups = dict(all=list(range(len(seen))), old=[i for i,c in enumerate(seen) if c not in current],
                  current=[i for i,c in enumerate(seen) if c in current], tail=[i for i,c in enumerate(seen) if c in tail])
    assert all(groups.values()), 'Missing comparison group'
    audit = {}
    for i in candidates:
        terms = {}
        for group, indices in groups.items():
            delta = (values[:,i,indices]-values[:,0,indices]).mean(1)
            terms[group] = dict(gain=float(delta.mean()), numerical_se=float(delta.std(unbiased=True)/math.sqrt(len(delta))))
        resolution = max(1e-8, 2*terms['all']['numerical_se'])
        protected = all(v['gain'] >= -1e-12 for g,v in terms.items() if g != 'all')
        audit[i] = dict(groups=terms, protection_pass=protected, resolution=resolution,
                        eligible=protected and terms['all']['gain'] > resolution)
    eligible = [i for i in candidates if audit[i]['eligible']]
    return min(eligible, key=lambda i: (-audit[i]['groups']['all']['gain'], i)) if eligible else 0, audit


def self_check():
    torch.set_num_threads(2)
    rows = [dict(label=c, identity_component=f'{c}_{i}') for c,n in [(0,6),(1,2),(2,1)] for i in range(n)]
    a,b,covered,support = split_identities(rows)
    assert covered == [0] and len(a) == len(b) == 3 and not support['1']['covered']
    for n,h,k in [(3,2,2),(10,1,2)]:
        scores = torch.randn(n,h,k,dtype=torch.float64); center,factor,_ = score_factor(scores)
        flat = scores.reshape(n,-1)-center
        assert torch.allclose(factor.T @ factor, flat.T @ flat/n, atol=1e-10, rtol=1e-10)
    z = torch.tensor([.5,1.5],dtype=torch.float64)
    scores = torch.stack([torch.stack([z,-z],1),torch.stack([-z,z],1)],1)
    p,_ = gaussian_scores(scores,0,93003,lambda:None,batches=4,samples=4096)
    assert abs(float(p[:,0].mean())-.5*(1+math.erf(2/math.sqrt(2)))) < .01
    assert torch.equal(p[:,0]+p[:,1],torch.ones(4,dtype=torch.float64))
    values = torch.tensor([[[.5,.5],[.6,.5],[.7,.4]]]*8,dtype=torch.float64)
    assert choose(values,[0,1],[1],[0],[0,1,2])[0] == 1
    assert choose(values,[0,1],[1],[0],[0,2])[0] == 0
