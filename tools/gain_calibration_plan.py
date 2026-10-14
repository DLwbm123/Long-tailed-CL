"""Training-side evidence and sample-feasibility planning. No data/model IO."""
import math
import random
import numpy as np

STEPS = (.125, .25, .5, 1.)


def feasible_support(train_n, independent_n, alpha, events, cap, pseudocount):
    """Optimistic zero-harm lower bound on required independent calibration size.

    Ignores numeric, donor and drift penalties: failure here cannot be repaired
    by those penalties. Augmentations and repeat stages are not new units.
    """
    if any(type(n) is not int or n < 0 for n in (train_n, independent_n, events)) or train_n < 1 or events < 1:
        raise ValueError('Invalid actual support')
    if not all(math.isfinite(x) for x in (alpha, cap, pseudocount)) or not 0 < alpha < 1 or not 0 < cap < 1 or pseudocount <= 0:
        raise ValueError('Invalid frozen risk parameters')
    threshold = cap*train_n/(train_n+pseudocount)
    budget = math.log(2*events/alpha)
    required = math.ceil(budget/-math.log1p(-threshold))
    upper = -math.expm1(-budget/independent_n) if independent_n else 1.
    return dict(train_n=train_n, independent_n=independent_n, harm_cap=threshold,
                ideal_zero_harm_upper=upper, minimum_independent_n=required,
                optimistic_sample_feasible=upper <= threshold+1e-15,
                ignores_numeric_drift_and_borrow=True)


def audit_independence(groups, exposure, split, independent_units_verified):
    """Head-only holdout cannot undo encoder, bank, ray or policy exposure."""
    if split != 'train_only' or type(independent_units_verified) is not bool:
        raise ValueError('Val/test/reserved or ambiguous independence forbidden')
    groups = set(groups)
    expected = {'encoder', 'head', 'bank', 'ray', 'policy'}
    if set(exposure) != expected:
        raise ValueError('Complete exposure lineage required')
    overlap = {key:len(groups & set(values)) for key,values in exposure.items()}
    independent = bool(groups) and independent_units_verified and not any(overlap.values())
    return dict(independent=independent, calibration_units=len(groups),
                exposure_overlap=overlap, patient_independence_claim=independent_units_verified,
                scope='independent_train_only' if independent else 'training_fit_diagnostic_only')


def prospective_split(rows, seed):
    """Existing 20% identity-group rule, planned before any model exposure.

    This changes a future training sample set; it cannot preserve old numerical
    rays/checkpoints and is never applied to a medical manifest in this tool.
    """
    if type(seed) is not int or not rows or any(r.get('split') != 'train_only' for r in rows):
        raise ValueError('Training-only prospective split required')
    held = set()
    for label in sorted({r['label'] for r in rows}):
        groups = sorted({r['group'] for r in rows if r['label'] == label})
        if len(groups) < 2:
            raise ValueError('Class has no independent group holdout')
        random.Random(seed+label*2003).shuffle(groups)
        held.update(groups[:min(len(groups)-1, max(1, round(.2*len(groups))))])
    fit = [r for r in rows if r['group'] not in held]
    calibration = [r for r in rows if r['group'] in held]
    if {r['label'] for r in fit} != {r['label'] for r in rows}:
        raise ValueError('Shared groups remove all fit samples of a class')
    assert not {r['group'] for r in fit} & {r['group'] for r in calibration}
    return fit, calibration


def diagnostic_folds(groups, seed):
    """Two calibrator-only folds; explicitly not an independent classifier test."""
    unique = sorted(set(groups))
    if len(unique) < 2 or type(seed) is not int:
        raise ValueError('At least two diagnostic groups required')
    random.Random(seed).shuffle(unique)
    return {group:i%2 for i,group in enumerate(unique)}


def paired_score_moments(before_features, after_features, projection, common_shift):
    """One paired pass; retain score-space moments rather than old features."""
    x, y, P, s = map(np.asarray, (before_features, after_features, projection, common_shift))
    if x.ndim != 2 or y.shape != x.shape or P.ndim != 2 or P.shape[0] != x.shape[1] or s.shape != (x.shape[1],) or len(x) < 1:
        raise ValueError('Invalid paired feature provenance/dimensions')
    if any(not np.isfinite(z).all() for z in (x,y,P,s)):
        raise ValueError('Nonfinite paired inputs')
    residual = (y-x-s)@P
    mean = residual.mean(0); second = residual.T@residual/len(residual)
    return dict(n=len(x), mean=mean, second=second,
                centered_covariance=second-np.outer(mean,mean),
                paired_current_representation=True, no_old_feature_retention=True)


def candidate_counts(off, changes, labels, guard):
    """All fixed candidates on one feature batch; no candidate selection here."""
    off, labels = np.asarray(off), np.asarray(labels)
    if off.ndim != 2 or labels.shape != (len(off),) or set(changes) != set(STEPS):
        raise ValueError('All unchanged fixed candidates required')
    if not np.isfinite(off).all() or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels >= off.shape[1]).any():
        raise ValueError('Invalid scores or labels')
    before = off.argmax(1); result = {}
    for step in STEPS:
        delta = np.asarray(changes[step])
        if delta.shape != off.shape or not np.isfinite(delta).all():raise ValueError('Invalid fixed score correction')
        on = guard(off, delta).argmax(1)
        result[step] = {int(c):dict(n=int((labels==c).sum()),
                                  rescued=int(((labels==c)&(before!=c)&(on==c)).sum()),
                                  harmed=int(((labels==c)&(before==c)&(on!=c)).sum()))
                        for c in np.unique(labels)}
    return result
