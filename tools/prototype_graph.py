"""Multiple prototypes, reliability-weighted graph drift, and exact mixture moments.

The graph predicts old-component translations; accuracy of that extrapolation is
an empirical hypothesis. Moment propagation is exact only for those translations.
"""
import numpy as np
from prototype_analytic import empty as single_empty, append as single_append


def partition(x):
    """Deterministic Lloyd clustering; at most four components per class."""
    x = np.asarray(x, dtype=np.float64)
    k = min(4, max(1, len(x) // 16))
    centers = [x[np.argmin(np.sum((x - x.mean(0)) ** 2, axis=1))]]
    for _ in range(1, k):
        distance = np.min(np.stack([np.sum((x-c)**2, axis=1) for c in centers]), axis=0)
        if distance.max() <= 1e-12:
            break
        centers.append(x[np.argmax(distance)])
    centers = np.stack(centers)
    previous = None
    for _ in range(20):
        distance = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(centers*centers, axis=1) - 2*x@centers.T, 0.)
        assignment = np.argmin(distance, axis=1)
        if previous is not None and np.array_equal(previous, assignment):
            break
        previous = assignment
        centers = np.stack([x[assignment == j].mean(0) for j in np.unique(assignment)])
    # Relabel after the final assignment so empty components never enter memory.
    return [np.flatnonzero(assignment == j) for j in np.unique(assignment)]


def empty(dim):
    return dict(single_empty(dim), components=[])


def append(bank, x, y, classes):
    result = single_append(bank, x, y, classes)
    components = [dict(c, center=c['center'].copy()) for c in bank['components']]
    for label_index, label in enumerate(classes, len(bank['n'])):
        z = np.asarray(x[y == label], dtype=np.float64)
        for indices in partition(z):
            sample = z[indices]; center = sample.mean(0)
            components.append(dict(label=label_index, center=center, count=len(sample),
                mass=len(sample)/len(z), radius=float(np.mean(np.sum((sample-center)**2, axis=1)))))
    result['components'] = components
    return result


def move(bank, shifts):
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.shape != (len(bank['components']), bank['S'].shape[0]) or not np.isfinite(shifts).all():
        raise ValueError('Invalid component translations')
    S, mu = bank['S'].copy(), bank['mu'].copy()
    components = []
    for c, d in zip(bank['components'], shifts):
        m = c['center']; w = c['mass']
        S += w * (np.outer(m, d) + np.outer(d, m) + np.outer(d, d))
        mu[c['label']] += w * d
        components.append(dict(c, center=m+d))
    return dict(S=S, mu=mu, n=bank['n'].copy(), components=components)


def reliability(deltas):
    """Support shrinkage times drift signal / (signal + estimated mean noise)."""
    n = len(deltas); mean = deltas.mean(0)
    signal = float(mean @ mean)
    if n < 2:
        return 0.
    mean_noise = float(np.sum((deltas-mean)**2) / (n*(n-1)))
    return float(n/(n+8.) * signal/(signal+mean_noise+1e-12))


def transport(bank, before, after, y, classes):
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    current = append(empty(before.shape[1]), before, y, classes)['components']
    observed, confidence = [], []
    # Reuse deterministic memberships of task-start features for paired drifts.
    for label in classes:
        z = before[y == label]; difference = after[y == label] - z
        for indices in partition(z):
            observed.append(difference[indices].mean(0))
            confidence.append(reliability(difference[indices]))
    observed, confidence = np.stack(observed), np.asarray(confidence)
    global_shift = np.mean([np.mean(after[y == c]-before[y == c], axis=0) for c in classes], axis=0)
    old_count = len(bank['components']); nodes = bank['components'] + current
    centers = np.stack([c['center'] for c in nodes]); radii = np.array([c['radius'] for c in nodes])
    masses = np.array([c['mass'] for c in nodes])
    distance = np.maximum(np.sum(centers*centers, axis=1)[:, None] + np.sum(centers*centers, axis=1)[None, :] - 2*centers@centers.T, 0.)
    np.fill_diagonal(distance, np.inf)
    affinity = np.exp(-distance / (radii[:, None]+radii[None, :]+1e-6))
    mask = np.zeros_like(affinity, dtype=bool)
    for i in range(len(nodes)):
        neighbors = np.argsort(distance[i], kind='stable')[:min(3, len(nodes)-1)]
        mask[i, neighbors] = True
    edges = affinity * (mask | mask.T) * np.sqrt(masses[:, None]*masses[None, :])
    laplacian = np.diag(edges.sum(1)) - edges
    anchors = np.r_[np.zeros(old_count), confidence * masses[old_count:]]
    prior = .1 * masses  # Positive global anchor keeps disconnected/noisy graphs well posed.
    A = laplacian + np.diag(anchors + prior)
    B = prior[:, None] * global_shift
    B[old_count:] += anchors[old_count:, None] * observed
    shifts = np.linalg.solve(A, B)
    residual = float(np.linalg.norm(A@shifts-B)/max(np.linalg.norm(B), 1e-15))
    if not np.isfinite(shifts).all() or residual > 1e-7:
        raise ValueError('Graph drift solve failed')
    audit = dict(global_shift_norm=float(np.linalg.norm(global_shift)),
        mean_old_shift_norm=float(np.linalg.norm(shifts[:old_count], axis=1).mean()) if old_count else 0.,
        old_components=old_count, current_components=len(current), graph_edges=int(np.count_nonzero(edges)/2),
        graph_residual=residual, anchor_reliability=confidence.tolist(),
        mean_anchor_reliability=float(confidence.mean()),
        observed_component_shift_norms=np.linalg.norm(observed, axis=1).tolist())
    return move(bank, shifts[:old_count]), audit
