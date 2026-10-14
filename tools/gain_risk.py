"""CPU design: fixed-ray risk admission, with explicit missing-evidence fallback.

No loaders, checkpoint IO, optimizer, launcher or medical calibration entry.
"""
import math
import numpy as np
import torch
import gain_joint as joint

VERSION = 'gain_boundary_transport_risk_v3'
STEPS = joint.STEPS
HEADS = ('RISK', 'RISK_NO_DRIFT', 'RISK_NO_BOUNDARY', 'RISK_NO_DAMAGE')
ALPHA = .01
EVENTS = 18 * 4 * 4 * 3 * 23 * 4 * 23 * 2
HARM_CAP = .01
PSEUDOCOUNT = 16.


def validate_design(p):
    expected = dict(version=VERSION, status='CPU_DESIGN_ONLY', primary_control='local_transport',
                    production_integrated=False, GPU_launch_authorized=False,
                    medical_data_access_authorized=False, checkpoint_access_authorized=False,
                    test_accessed=False, training_feedback=False, validation_retuning=False,
                    automatic_retry=False, optimizer_updates=0, policy_updates=0,
                    seed=74002, direction_changed=False, steps=list(STEPS), heads=list(HEADS),
                    zero_anchor=True, future_gpus=[0, 1], statistical_alpha=ALPHA,
                    statistical_family_events=EVENTS, harm_cap=HARM_CAP,
                    support_pseudocount=PSEUDOCOUNT, missing_evidence='identity',
                    clinical_confidence_calibrated=False)
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Independent synthetic design scope required')


def finite(*xs):
    if any(not np.isfinite(x).all() for x in xs):
        raise ValueError('Nonfinite risk input')


def binomial_bounds(rate, n):
    """One-sided KL-Chernoff inversion with the frozen family error allocation."""
    if not math.isfinite(rate) or not 0 <= rate <= 1 or type(n) is not int or n < 1:
        raise ValueError('Invalid IID binary-event support')
    budget = math.log(2*EVENTS/ALPHA)/n
    def kl(p):
        if p <= 0 or p >= 1:return math.inf
        return (rate*math.log(rate/p) if rate else 0.) + ((1-rate)*math.log((1-rate)/(1-p)) if rate < 1 else 0.)
    if rate == 0:return 0., -math.expm1(-budget)
    if rate == 1:return math.exp(-budget), 1.
    left, right = 0., rate
    for _ in range(64):
        mid = (left+right)/2
        if kl(mid) > budget:left = mid
        else:right = mid
    lower = right; left, right = rate, 1.
    for _ in range(64):
        mid = (left+right)/2
        if kl(mid) > budget:right = mid
        else:left = mid
    return lower, left


def calibration(predicted_rescue, predicted_harm, rescued, harmed, n, independent):
    """One-sided event discrepancies, not a moment-derived TV certificate.

    Coverage is conditional on an independent IID, same-domain calibration set.
    Missing independence is not repaired by a donor or a larger historical n.
    """
    if type(n) is not int or n < 0 or type(independent) is not bool:
        raise ValueError('Invalid calibration support')
    if any(type(v) is not int or v < 0 for v in (rescued, harmed)) or rescued + harmed > n:
        raise ValueError('Invalid disjoint rescue/harm counts')
    finite(predicted_rescue, predicted_harm)
    if not 0 <= predicted_rescue <= 1 or not 0 <= predicted_harm <= 1 or predicted_rescue + predicted_harm > 1 + 1e-12:
        raise ValueError('Invalid predicted event rates')
    if not n or not independent:
        return dict(rescue_error=1., harm_error=1., identified=False, n=n,
                    reason='missing_independent_same_domain_calibration')
    rl, _ = binomial_bounds(rescued/n, n); _, hu = binomial_bounds(harmed/n, n)
    return dict(rescue_error=min(1., max(0., predicted_rescue-rl)),
                harm_error=min(1., max(0., hu-predicted_harm)),
                identified=True, n=n, empirical_rescue_lower=rl, empirical_harm_upper=hu,
                reason='conditional_IID_same_domain_event_discrepancy')


def boundary_prior(covariances, W, ray, truth, supports, ages):
    """Target-competitor score-gap matching; never replace target centers/supports."""
    W, ray = np.asarray(W), np.asarray(ray)
    finite(W, ray)
    k = W.shape[1]
    if ray.shape != W.shape or len(covariances) != k or len(supports) != k or len(ages) != k or type(truth) is not int or not 0 <= truth < k:
        raise ValueError('Boundary dimensions differ')
    if any(type(n) is not int or n < 1 for n in supports) or any(type(a) is not int or a < 0 for a in ages):
        raise ValueError('Invalid actual support/transport age')
    competitors = [j for j in range(k) if j != truth]
    P = np.concatenate((W[:, truth, None] - W[:, competitors],
                        ray[:, truth, None] - ray[:, competitors]), axis=1)
    shapes = []
    for S in covariances:
        S = np.asarray(S); finite(S)
        if S.shape != (W.shape[0], W.shape[0]) or not np.allclose(S, S.T, atol=1e-12, rtol=1e-10):
            raise ValueError('Invalid donor covariance')
        G = P.T @ S @ P
        tol = 64*np.finfo(float).eps*max(1., float(np.abs(G).max()))
        if np.linalg.eigvalsh((G+G.T)/2).min() < -tol:
            raise ValueError('Non-PSD boundary covariance')
        shapes.append(G / max(float(np.trace(G)), np.finfo(float).tiny))
    donors = [j for j in range(k) if j != truth and supports[j] >= 2 and np.trace(shapes[j]) > 0]
    distances = np.asarray([np.square(shapes[j]-shapes[truth]).sum() for j in donors])
    weights = np.asarray([supports[j]/(supports[j]+PSEUDOCOUNT)/(1+ages[j]) for j in donors]) * np.exp(-distances/.1)
    total = weights.sum()
    weights = weights/total if total else np.zeros_like(weights)
    # Similarity is a prior proposal. Its reliability must be supplied separately.
    prior = sum((w*np.asarray(covariances[j])/max(float(np.trace(covariances[j])), np.finfo(float).tiny)
                 for w, j in zip(weights, donors)), np.zeros_like(covariances[truth]))
    eta = .4*PSEUDOCOUNT/(supports[truth]+PSEUDOCOUNT) if total else 0.
    donor_covariances = [np.asarray(covariances[j])/max(float(np.trace(covariances[j])), np.finfo(float).tiny) for j in donors]
    return dict(prior=prior, donor_covariances=donor_covariances, donor_labels=donors, donor_weights=weights,
                boundary_distances=distances, eta=eta, support=supports[truth],
                trust_not_inferred_from_similarity=True)


def borrowed_family(empirical, prior):
    empirical = np.asarray(empirical); finite(empirical)
    trace = float(np.trace(empirical))
    # A distribution mixture admits a transfer bound; a covariance blend does not.
    return [[dict(covariance=empirical.copy(), weight=1-prior['eta']*fraction)] +
            [dict(covariance=S*trace, weight=float(prior['eta']*fraction*w))
             for S,w in zip(prior['donor_covariances'], prior['donor_weights']) if fraction*w]
            for fraction in (0., .5, 1.)]


def borrowing_error(prior, donor_transfer_upper, fraction):
    """Mixture transfer penalty; unsupported donors retain maximal discrepancy."""
    if fraction not in (0., .5, 1.):
        raise ValueError('Frozen borrow fraction differs')
    values = []
    for j in prior['donor_labels']:
        v = donor_transfer_upper.get(j, 1.)
        if not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError('Invalid independent donor transfer bound')
        values.append(v)
    return float(prior['eta'] * fraction * np.dot(prior['donor_weights'], values))


def drift_radius(projection, residual_second_moments):
    """Sum transition RMS envelopes; never assume errors cancel or are IID.

    This is a sensitivity envelope, not an old-class confidence region. Its
    transfer validity and uncovered probability remain separate evidence.
    """
    projection = np.asarray(projection); finite(projection)
    radius = np.zeros(projection.shape[1])
    for S in residual_second_moments:
        S = np.asarray(S); finite(S)
        if S.shape != (projection.shape[0],)*2 or not np.allclose(S, S.T, atol=1e-12, rtol=1e-10):
            raise ValueError('Invalid paired residual second moment')
        G = projection.T @ S @ projection
        if np.linalg.eigvalsh((G+G.T)/2).min() < -1e-10*max(1., float(np.abs(G).max())):
            raise ValueError('Non-PSD transport envelope')
        radius += np.sqrt(np.maximum(0., np.diag(G)))
    return radius


def event_bounds(before, change, radius, old, truth):
    """Outer bounds for coupled OFF/guarded-ON events over a score box."""
    k = before.shape[1]; radius = np.asarray(radius)
    if change.shape != before.shape or radius.shape != (2*k,) or not 0 <= old <= k or not 0 <= truth < k:
        raise ValueError('Invalid event box')
    finite(before, change, radius)
    if (radius < 0).any() or change[:, old:].any() or radius[k+old:].any():
        raise ValueError('New correction is not fixed')
    lo, hi = before-radius[:k], before+radius[:k]
    onlo, onhi = lo+change-radius[k:], hi+change+radius[k:]
    if old:
        caplo = np.maximum(0., onlo[:, :old].max(1)-hi[:, :old].max(1))
        caphi = np.maximum(0., onhi[:, :old].max(1)-lo[:, :old].max(1))
        onlo[:, :old] -= caphi[:, None]; onhi[:, :old] -= caplo[:, None]
    def wins(lower, upper):
        others = [j for j in range(k) if j != truth]
        if not others:return np.ones(len(lower), bool), np.ones(len(lower), bool)
        # np.argmax resolves ties by the smallest index.
        strict = np.asarray(others) < truth
        certain = np.all(np.where(strict, lower[:, truth, None] > upper[:, others],
                                  lower[:, truth, None] >= upper[:, others]), axis=1)
        possible = np.all(np.where(strict, upper[:, truth, None] > lower[:, others],
                                   upper[:, truth, None] >= lower[:, others]), axis=1)
        return certain, possible
    offsure, offpossible = wins(lo, hi); onsure, onpossible = wins(onlo, onhi)
    rl, ru = ~offpossible & onsure, ~offsure & onpossible
    hl, hu = offsure & ~onpossible, offpossible & ~onsure
    if truth >= old:hl[:] = False; hu[:] = False
    return dict(rescue_lower=float(rl.mean()), rescue_upper=float(ru.mean()),
                harm_lower=float(hl.mean()), harm_upper=float(hu.mean()))


def integrate(mean, covariance, radius, old, truth, df, stream):
    """Same nodes/seed as v2; uncertain events replace point winner counts."""
    mean = np.asarray(mean); covariance = np.asarray(covariance); finite(mean, covariance)
    k = len(mean)//2
    if k < 1 or mean.shape != (2*k,) or covariance.shape != (2*k, 2*k) or type(old) is not int or not 0 <= old <= k or type(truth) is not int or not 0 <= truth < k or len(stream) != 2 or df is not None and (type(df) is not int or df < 1):
        raise ValueError('Invalid joint-score inputs')
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10) or mean[k+old:].any() or covariance[k+old:].any():
        raise ValueError('Invalid covariance/new correction')
    eigen, U = np.linalg.eigh((covariance+covariance.T)/2)
    tol = 64*np.finfo(float).eps*len(eigen)*max(np.finfo(float).tiny, float(np.abs(eigen).max()))
    if eigen.min() < -tol:raise ValueError('Non-PSD joint covariance')
    active = eigen >= tol; root = U[:, active]*np.sqrt(eigen[active])
    duplicates = [(j, next(i for i in range(j) if np.array_equal(covariance[j], covariance[i])))
                  for j in range(1, 2*k) if any(np.array_equal(covariance[j], covariance[i]) for i in range(j))]
    rng = np.random.default_rng(np.random.SeedSequence([joint.SEED, *stream]))
    total = joint.NODES if active.any() else 1; sums = dict(rescue_lower=0., rescue_upper=0., harm_lower=0., harm_upper=0.)
    for start in range(0, total, joint.BATCH):
        n = min(joint.BATCH, total-start); noise = rng.standard_normal((n, root.shape[1]))@root.T
        for j, i in duplicates:noise[:, j] = noise[:, i]
        if df is not None and active.any():noise /= np.sqrt(rng.chisquare(df, size=n)/df)[:, None]
        scores = mean+noise
        z = event_bounds(scores[:, :k], scores[:, k:], radius, old, truth)
        for key in sums:sums[key] += n*z[key]
    rates = {key:value/total for key,value in sums.items()}
    rl, _ = binomial_bounds(rates['rescue_lower'], total) if active.any() else (rates['rescue_lower'], rates['rescue_lower'])
    _, hu = binomial_bounds(rates['harm_upper'], total) if active.any() else (rates['harm_upper'], rates['harm_upper'])
    re, he = rates['rescue_lower']-rl, hu-rates['harm_upper']
    return dict(**rates, numerical_error=max(re, he), numerical_rescue_error=re, numerical_harm_error=he,
                numeric_score_nodes=total, conditional_on_supplied_model=True)


def class_risk(component_results, masses, calibration_evidence, transport, borrow_penalty, old_class, mode):
    if mode not in HEADS or len(component_results) != len(masses) or abs(sum(masses)-1.) > 1e-8 or any(m <= 0 for m in masses):
        raise ValueError('Invalid class/mode support')
    if not 0 <= borrow_penalty <= 1:raise ValueError('Invalid borrow penalty')
    calibration_evidence = calibration_evidence or dict(rescue_error=1., harm_error=1., identified=False)
    if transport is not None and (type(transport.get('identified')) is not bool or not 0 <= transport.get('uncovered_upper', -1) <= 1):
        raise ValueError('Invalid independent transport evidence')
    missing_transport = old_class and (transport is None or not transport['identified'])
    de = (1. if missing_transport else transport['uncovered_upper'] if old_class else 0.) if mode != 'RISK_NO_DRIFT' else 0.
    re, he = calibration_evidence['rescue_error'], calibration_evidence['harm_error']
    if not 0 <= re <= 1 or not 0 <= he <= 1 or type(calibration_evidence['identified']) is not bool:
        raise ValueError('Invalid independent calibration evidence')
    if not calibration_evidence['identified']:re = he = 1.
    r = sum(m*max(0., z['rescue_lower']-z.get('numerical_rescue_error', z['numerical_error'])) for m,z in zip(masses,component_results))
    h = sum(m*min(1., z['harm_upper']+z.get('numerical_harm_error', z['numerical_error'])) for m,z in zip(masses,component_results))
    rl, hu = max(0., r-re-de-borrow_penalty), min(1., h+he+de+borrow_penalty)
    if not old_class:hu = 0.
    return dict(rescue_lower=rl, harm_upper=hu, net_lower=rl-hu,
                calibration_rescue_error=re, calibration_harm_error=he,
                transport_uncovered_upper=de, borrow_error=borrow_penalty,
                missing_transport=missing_transport, calibration_identified=calibration_evidence['identified'])


def estimate(models, W, ray, delta, old, supports, ages, calibration_evidence, transports, donor_transfer, mode):
    """Design adapter for existing moment models; supplied evidence is never fitted here.

    Calibration must bind to the exact control/stage/candidate/estimator in a
    future production adapter. This helper accepts only synthetic CPU inputs.
    """
    if mode not in HEADS or W.device.type != 'cpu' or delta.shape != W.shape or ray.shape != W.shape or delta[:, old:].any():
        raise ValueError('Fixed CPU candidate required')
    k = W.shape[1]
    if not all(len(z) == k for z in (models, supports, ages, calibration_evidence, transports, donor_transfer)):
        raise ValueError('Complete class evidence required')
    if not delta.any():
        return dict(robust_lower=0., scenarios=[dict(classes=[dict(rescue_lower=0., harm_upper=0., net_lower=0.) for _ in range(k)]) for _ in range(3)], exact_identity=True)
    projection = torch.cat((W, delta), 1).numpy(); w, r = W.numpy(), ray.numpy()
    covariances = [sum((z['mass']*z['family'][0].numpy() for z in parts), np.zeros((W.shape[0],)*2)) for parts in models]
    priors = [boundary_prior(covariances, w, r, c, supports, ages) for c in range(k)]
    scenarios = []
    for index, fraction in enumerate((0., .5, 1.)):
        classes = []
        for c, parts in enumerate(models):
            if not parts or len(parts) > 4 or abs(sum(z['mass'] for z in parts)-1.) > 1e-8 or any(type(z['count']) is not int or z['count'] < 1 or abs(z['mass']-z['count']/supports[c]) > 1e-8 for z in parts) or sum(z['count'] for z in parts) != supports[c]:
                raise ValueError('Invalid class component masses')
            if calibration_evidence[c] is not None and calibration_evidence[c].get('n', 0) > supports[c]:
                raise ValueError('Calibration support cannot exceed actual class support')
            radius = np.zeros(2*k)
            t = transports[c]
            if c < old and mode != 'RISK_NO_DRIFT' and t is not None:
                radius = drift_radius(projection, t.get('residual_second_moments', []))
            rows = []; masses = []
            for part_index, part in enumerate(parts):
                masses.append(part['mass'])
                if part['count'] < 2 or not part['data_identified']:
                    rows.append(dict(rescue_lower=0., rescue_upper=1., harm_lower=0., harm_upper=float(c < old), numerical_error=0.));continue
                mean = part['center'].numpy()@projection
                if mode == 'RISK_NO_BOUNDARY':
                    mixture = [dict(covariance=part['family'][min(index, len(part['family'])-1)].numpy(), weight=1.)]
                else:
                    mixture = borrowed_family(part['family'][0].numpy(), priors[c])[index]
                row = dict(rescue_lower=0., rescue_upper=0., harm_lower=0., harm_upper=0., numerical_error=0., numerical_rescue_error=0., numerical_harm_error=0.)
                for donor_index, distribution in enumerate(mixture):
                    S = distribution['covariance']*(part['count']+1)/(part['count']-1)
                    z = integrate(mean, projection.T@S@projection, radius, old, c, part['count']-1, (c, part_index))
                    for key in row:row[key] += distribution['weight']*z[key]
                rows.append(row)
            # Unsupported covariance-blend transfer receives the full legacy eta.
            penalty = priors[c]['eta']*fraction if mode == 'RISK_NO_BOUNDARY' else borrowing_error(priors[c], donor_transfer[c], fraction)
            risk = class_risk(rows, masses, calibration_evidence[c], t, penalty, c < old, mode)
            classes.append(dict(risk, actual_support=supports[c], transport_age=ages[c],
                                donor_labels=priors[c]['donor_labels'], donor_weights=priors[c]['donor_weights'].tolist()))
        scenarios.append(dict(classes=classes, net_lower=sum(z['net_lower'] for z in classes)/k))
    return dict(scenarios=scenarios, robust_lower=min(z['net_lower'] for z in scenarios), exact_identity=False,
                conditional_model_and_transfer_assumptions=True, clinical_confidence_calibrated=False)


def worst_classes(report):
    """Keep each class's worst rescue/harm across scenarios before admission."""
    k = len(report['scenarios'][0]['classes']); out = []
    for c in range(k):
        rows = [z['classes'][c] for z in report['scenarios']]
        r = min(z['rescue_lower'] for z in rows); h = max(z['harm_upper'] for z in rows)
        out.append(dict(rescue_lower=r, harm_upper=h, net_lower=r-h))
    return out


def select(candidate_classes, supports, old, mode):
    """Identity competes with exactly four unchanged candidates. No val input."""
    if mode not in HEADS or len(candidate_classes) != 4 or type(old) is not int or not 0 <= old <= len(supports) or any(type(n) is not int or n < 1 for n in supports):
        raise ValueError('Invalid selection geometry')
    rows = []; best = 1e-12; chosen = 0.
    for step, classes in zip(STEPS, candidate_classes):
        if len(classes) != len(supports):raise ValueError('Class coverage differs')
        for c in classes:
            finite(c['rescue_lower'], c['harm_upper'], c['net_lower'])
            if not 0 <= c['rescue_lower'] <= 1 or not 0 <= c['harm_upper'] <= 1 or abs(c['net_lower']-c['rescue_lower']+c['harm_upper']) > 1e-12:
                raise ValueError('Invalid risk bounds')
        net = sum(c['net_lower'] for c in classes)/len(classes)
        caps = [HARM_CAP*n/(n+PSEUDOCOUNT) for n in supports[:old]]
        failures = [i for i,cap in enumerate(caps) if classes[i]['harm_upper'] > cap+1e-12]
        admitted = net > 1e-12 and (not failures or mode == 'RISK_NO_DAMAGE')
        rows.append(dict(scale=step, net_lower=net, admitted=admitted, damage_failed_classes=failures, harm_caps=caps))
        if admitted and net > best:best, chosen = net, step
    return dict(chosen_scale=chosen, candidates=rows, mode=mode, direction_changed=False,
                clinical_confidence_calibrated=False, evidence_conditional=True)


def selected_delta(ray, selection, old):
    if ray.device.type != 'cpu' or not torch.isfinite(ray).all() or type(old) is not int or not 0 <= old <= ray.shape[1] or ray[:, old:].any() or selection['chosen_scale'] not in (0.,)+STEPS:
        raise ValueError('Unchanged CPU old-only ray required')
    tolerance = 1e-8+torch.finfo(torch.float32).eps*ray[:, :old].abs().sum(1)
    if (ray[:, :old].sum(1).abs() > tolerance).any():raise ValueError('Discriminative old-only ray required')
    return (selection['chosen_scale']*ray).float().double()
