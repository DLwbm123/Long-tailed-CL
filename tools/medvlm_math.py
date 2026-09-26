"""Fixed MedVLM readouts; all statistics are from the current seen prefix."""
import numpy as np
from route_a.semantic_prior import component_reliability, semantic_scale
from route_a.spectral_prior_ridge import ridge

CORE = ('V', 'J', 'J1', 'Z', 'L', 'R')
EXTRA = ('R-no-gate', 'R-identity', 'R-zero-prior', 'L-permute', 'template-0', 'template-1')

def fit(bank, text, text0, text1, *, extensions=False):
    S, M, ids = bank.class_balanced()
    k, da = len(ids), bank.dim_a
    G, R = S / k, M / k
    lam = 5e-4
    sigma, U = np.linalg.eigh((G + G.T) / 2)
    if sigma.min() < -1e-10:
        raise ValueError('NONPSD')
    sigma = np.maximum(sigma, 0)
    ur = U.T @ R
    solve_ridge = lambda value: U @ (ur / (sigma[:, None] + value))
    Gu, Ru = bank.u_stats()
    scale = semantic_scale(text, Gu, Ru, lam=lam)
    stats, counts = bank.component_reliability_inputs()
    reliability, gamma = component_reliability(text, stats, counts, class_ids=ids, a_scale=scale)
    rarity = .001 * 10 / (np.asarray([counts[c] for c in ids]) + 10) if scale else np.zeros(k)
    prior = np.vstack((np.zeros((da, k)), np.sqrt(2) * scale * (text - text.mean(1, keepdims=True))))
    uv = U.T @ prior
    def prior_solve(g, *, identity=False, zero=False):
        p = np.ones_like(sigma) if identity else lam / (sigma + lam)
        rhs = ur + p[:, None] * g * (0 if zero else uv)
        w = U @ (rhs / (sigma[:, None] + lam + p[:, None] * g))
        if not np.isfinite(w).all():
            raise ValueError('NONFINITE_READOUT')
        return w
    if extensions:
        return {'R-no-gate': prior_solve(rarity), 'R-identity': prior_solve(gamma, identity=True),
                'R-zero-prior': prior_solve(gamma, zero=True)}
    return {'V': ridge(Gu, Ru, 1e-3), 'J': solve_ridge(lam), 'J1': solve_ridge(1e-3),
            'F': ridge(2 * G[:da, :da], np.sqrt(2) * R[:da], 1e-3),
            'F2': ridge(2 * G[:da, :da], np.sqrt(2) * R[:da], 2e-3),
            'R': prior_solve(gamma), 'text': text, 'text0': text0, 'text1': text1,
            'gamma': gamma, 'reliability': reliability, 'rarity': rarity, 'scale': np.asarray(scale),
            'class_ids': ids}

def score(weights, method, a, u):
    h = np.concatenate((a, u), axis=1) / np.sqrt(2)
    if method in ('F', 'F2'): return a @ weights[method]
    if method == 'V': return u @ weights['V']
    if method == 'Z': return u @ weights['text']
    if method.startswith('template-'): return u @ weights['text' + method[-1]]
    if method in ('L', 'L-permute'):
        s = h @ weights['J']
        text = weights['text']
        if method == 'L-permute':
            order = np.argsort(weights['class_ids'])
            perm = np.arange(len(order)); perm[order] = np.roll(order, 1)
            text = text[:, perm]
        top = np.argpartition(s, -min(5, s.shape[1]), axis=1)[:, -min(5, s.shape[1]):]
        rows = np.arange(len(s))[:, None]
        s[rows, top] += (u @ text)[rows, top]
        return s
    return h @ weights[method]

def reliability_details(bank, text, scale):
    """Expose the existing formula's margin and uncertainty without changing it."""
    ids=bank.class_order
    stats, counts=bank.component_reliability_inputs()
    means={c:stats[c]['sum']/max(counts[c],1) for c in ids}
    cov={c:stats[c]['second']/max(counts[c],1)-np.outer(means[c],means[c]) for c in ids}
    valid=[cov[c] for c in ids if counts[c]>=2]
    pool=sum(valid,np.zeros_like(next(iter(cov.values()))))/len(valid) if valid else np.eye(bank.dim_u)/bank.dim_u
    reliability,gamma=component_reliability(text,stats,counts,class_ids=ids,a_scale=scale)
    out=[]
    for j,c in enumerate(ids):
        competitors=[q for q in range(len(ids)) if q!=j]
        other=max(competitors,key=lambda q:float(means[c]@text[:,q])) if competitors else j
        diff=text[:,j]-text[:,other] if competitors else text[:,j]
        margin=float(means[c]@diff)
        omega=10/(counts[c]+10)
        var=max(float(diff@((1-omega)*cov[c]+omega*pool)@diff),0)
        lower=margin-np.sqrt(var/max(counts[c],1))
        reason=('scale_zero' if scale==0 else 'component_count_lt_2' if counts[c]<2 else
                'margin_nonpositive' if margin<=0 else 'lower_bound_nonpositive' if lower<=0 else 'active')
        out.append({'class_id':int(c),'component_count':int(counts[c]),'semantic_scale':float(scale),
                    'margin':margin,'margin_lower_bound':float(lower),'reliability':float(reliability[j]),
                    'gamma':float(gamma[j]),'gate_reason':reason})
    return out
