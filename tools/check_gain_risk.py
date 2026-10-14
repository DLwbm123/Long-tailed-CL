"""Synthetic CPU design checks. Never imports a medical worker or loader."""
import copy
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
import gain_risk as risk
import gain_joint as joint
import prototype_coherent as native
import core1_competition as core
import module_additions as modules


def reject(f):
    try:f()
    except (ValueError, ArithmeticError):return
    raise AssertionError('Illegal input accepted')


def run(protocol):
    start = time.monotonic(); torch.set_num_threads(2); torch.manual_seed(74002)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    p = json.loads(Path(protocol).read_text()); risk.validate_design(p)
    for key in ('GPU_launch_authorized', 'medical_data_access_authorized', 'checkpoint_access_authorized', 'test_accessed', 'production_integrated', 'validation_retuning'):
        reject(lambda key=key:risk.validate_design(dict(p, **{key:True})))
    reject(lambda:risk.validate_design(dict(p, steps=[.1, .25, .5, 1.])))
    reject(lambda:risk.validate_design(dict(p, harm_cap=.02)))
    rng = np.random.default_rng(74002)
    before = rng.normal(size=(400, 23)); change = rng.normal(0, .1, size=(400, 23)); change[:, 21:] = 0.
    change[:, :21] -= change[:, :21].mean(1, keepdims=True)
    radius = rng.uniform(.001, .03, size=46); radius[44:] = 0.
    for truth in (0, 13, 20, 21, 22):
        bounds = risk.event_bounds(before, change, radius, 21, truth)
        for _ in range(8):
            noise = rng.uniform(-1, 1, (400, 46))*radius
            off = before+noise[:, :23]; on = joint.guard_scores(off, change+noise[:, 23:], 21)
            rescued = ((off.argmax(1) != truth)&(on.argmax(1) == truth)).mean()
            harmed = ((off.argmax(1) == truth)&(on.argmax(1) != truth)).mean()
            assert bounds['rescue_lower'] <= rescued <= bounds['rescue_upper']
            assert bounds['harm_lower'] <= harmed <= bounds['harm_upper']
        larger = risk.event_bounds(before, change, radius*2, 21, truth)
        assert larger['rescue_lower'] <= bounds['rescue_lower'] and larger['harm_upper'] >= bounds['harm_upper']
        if truth >= 21:assert bounds['harm_upper'] == 0.
    for truth in range(3):
        off = np.zeros((4, 3)); delta = np.zeros_like(off)
        z = risk.event_bounds(off, delta, np.zeros(6), 2, truth)
        assert all(v == 0. for v in z.values())
    # Zero uncertainty and identical streams recover the existing categorical model.
    mean = np.array([.05, -.05, -1., .1, -.1, 0.]); v = np.array([.1, -.1, 0., 0., 0., 0.]); covariance = np.outer(v, v)
    point = joint.integrate(mean, covariance, 2, 0, 24, (0, 0))
    bounded = risk.integrate(mean, covariance, np.zeros(6), 2, 0, 24, (0, 0))
    assert bounded['rescue_lower'] == bounded['rescue_upper'] == point['rescue']
    assert bounded['harm_lower'] == bounded['harm_upper'] == point['harm']
    # Numerical harm error must shrink with zero observed harm, not impose a
    # symmetric Hoeffding floor above every frozen damage cap.
    assert bounded['numerical_harm_error'] < .001
    large_cal = risk.calibration(point['rescue'], point['harm'],
                                 round(point['rescue']*1000000), round(point['harm']*1000000), 1000000, True)
    safe = risk.class_risk([bounded], [1.], large_cal,
                          dict(identified=True, uncovered_upper=0.), 0., True, 'RISK')
    zero_component = dict(rescue_lower=0., harm_upper=0., numerical_error=0.)
    old_safe = risk.class_risk([zero_component], [1.], risk.calibration(0., 0., 0, 0, 1000000, True),
                              dict(identified=True, uncovered_upper=0.), 0., True, 'RISK')
    new_safe = dict(rescue_lower=0., harm_upper=0., net_lower=0.)
    assert risk.select([[safe, old_safe, new_safe]]*4, [1000000]*3, 2, 'RISK')['chosen_scale'] == .125
    for n in (5, 25, 100):
        for probability in (.1, .5, .9):
            violated = 0.
            for count in range(n+1):
                lower, upper = risk.binomial_bounds(count/n, n)
                if probability < lower or probability > upper:
                    violated += math.comb(n, count)*probability**count*(1-probability)**(n-count)
            assert violated <= risk.ALPHA/risk.EVENTS + 1e-14
    shifted = risk.integrate(mean, covariance, np.array([.2, .2, 0., .1, .1, 0.]), 2, 0, 24, (0, 0))
    assert shifted['rescue_lower'] <= bounded['rescue_lower'] and shifted['harm_upper'] >= bounded['harm_upper']
    reject(lambda:risk.integrate(mean, -np.eye(6), np.zeros(6), 2, 0, None, (0, 0)))
    reject(lambda:risk.event_bounds(before, change, np.ones(46), 21, 0))
    missing = risk.calibration(.1, .001, 0, 0, 25, False)
    assert not missing['identified'] and missing['harm_error'] == 1.
    harm = risk.calibration(.1, .001, 0, 2, 25, True)
    assert harm['harm_error'] >= 2/25-.001 and harm['rescue_error'] >= .1
    reject(lambda:risk.calibration(.1, .1, 4, 4, 5, True))
    assert risk.calibration(0., 0., 0, 0, 0, True)['harm_error'] == 1.
    positive = [dict(rescue_lower=.15, harm_upper=.001, net_lower=.149),
                dict(rescue_lower=0., harm_upper=.001, net_lower=-.001),
                dict(rescue_lower=.01, harm_upper=0., net_lower=.01)]
    support = [25, 1000, 1000]; candidates = [copy.deepcopy(positive) for _ in risk.STEPS]
    selected = risk.select(candidates, support, 2, 'RISK'); assert selected['chosen_scale'] == .125
    rare_damage = copy.deepcopy(candidates)
    for classes in rare_damage:classes[0] = dict(rescue_lower=.15, harm_upper=.02, net_lower=.13)
    assert risk.select(rare_damage, support, 2, 'RISK')['chosen_scale'] == 0.
    assert risk.select(rare_damage, support, 2, 'RISK_NO_DAMAGE')['chosen_scale'] == .125
    for n in (1, 3, 23, 25, 34, 1000):
        cap = risk.select(candidates, [n, 1000, 1000], 2, 'RISK')['candidates'][0]['harm_caps'][0]
        assert 0 < cap < risk.HARM_CAP
    component = dict(rescue_lower=.3, rescue_upper=.3, harm_lower=0., harm_upper=0., numerical_error=0.)
    calibrated = risk.calibration(.3, 0., 30000, 0, 100000, True)
    transport = dict(identified=True, uncovered_upper=0., residual_second_moments=[])
    good = risk.class_risk([component], [1.], calibrated, transport, 0., True, 'RISK'); assert good['net_lower'] > 0.
    unknown = risk.class_risk([component], [1.], calibrated, None, 0., True, 'RISK'); assert unknown['net_lower'] == -1.
    unidentified = risk.class_risk([component], [1.], missing, transport, 0., True, 'RISK'); assert unidentified['net_lower'] == -1.
    ignored = risk.class_risk([component], [1.], calibrated, None, 0., True, 'RISK_NO_DRIFT'); assert ignored['net_lower'] > 0.
    # Boundary donor selection changes when the candidate's actual pair directions change.
    W = np.array([[1., -1., 0.], [0., 0., 0.], [0., 0., 0.]])
    ray = np.array([[0., 0., 0.], [.1, -.1, 0.], [0., 0., 0.]])
    covs = [np.diag([1., 1., 2.]), np.diag([1., 1., 20.]), np.diag([1., 20., 2.])]
    donor = risk.boundary_prior(covs, W, ray, 0, [25, 1000, 1000], [2, 0, 0])
    assert donor['donor_weights'][0] > donor['donor_weights'][1]
    assert risk.borrowing_error(donor, {}, 1.) == donor['eta']
    assert risk.borrowing_error(donor, {1:0., 2:0.}, 1.) == 0.
    family = risk.borrowed_family(covs[0], donor)
    assert len(family[0]) == 1 and all(abs(sum(z['weight'] for z in f)-1.) < 1e-12 for f in family)
    for f in family:assert abs(sum(z['weight']*np.trace(z['covariance']) for z in f)-np.trace(covs[0])) < 1e-10
    P = np.concatenate((W, ray), 1); residual = np.diag([.01, .02, .03])
    one = risk.drift_radius(P, [residual]); two = risk.drift_radius(P, [residual, residual])
    np.testing.assert_allclose(two, 2*one)
    assert not risk.drift_radius(P, []).any()
    # Same-state adapter exercises all contribution modes without medical evidence.
    tW = torch.tensor(W, dtype=torch.float64); tR = torch.tensor(ray, dtype=torch.float64)
    stats = [[dict(center=torch.tensor([1. if c == 0 else -1., 1., 1.], dtype=torch.float64),
                   count=25, mass=1., family=[torch.tensor(covs[c], dtype=torch.float64)], data_identified=True)] for c in range(3)]
    snap = copy.deepcopy(stats); adapted = {}
    for mode in risk.HEADS:
        reports = [risk.estimate(stats, tW, tR, (step*tR).float().double(), 2,
                                [25]*3, [1, 1, 0], [None]*3, [None]*3, [{}]*3, mode) for step in risk.STEPS]
        choice = risk.select([risk.worst_classes(z) for z in reports], [25]*3, 2, mode)
        assert choice['chosen_scale'] == 0.; adapted[mode] = dict(missing_evidence_identity=True)
        for a,b in zip(stats, snap):assert torch.equal(a[0]['center'], b[0]['center']) and torch.equal(a[0]['family'][0], b[0]['family'][0])
    # Full native Hessian/reference and frozen ray at the actual feature dimension.
    d, k, old = 1536, 23, 21; mu = torch.eye(d, dtype=torch.float64)[:k]*.2; mu[:, -1] = 1.
    S = torch.eye(d, dtype=mu.dtype)*.05; counts = [2048]*k; counts[1] = 3
    bank = dict(mu=mu, Q=torch.stack([torch.outer(m,m)+S for m in mu]), n=counts,
                components=[dict(label=c, center=mu[c], count=counts[c], mass=mu.new_tensor(1.), radius=S.trace()) for c in range(k)])
    snapshot = copy.deepcopy(bank); initial,_ = native.head(bank); pairs = core.competition(bank, initial); highdim = []
    for arm in ('base', 'local_transport'):
        R,_ = modules.metric(bank, arm, 'HK', list(range(k)), None if arm == 'base' else torch.randn(k,4,d,dtype=mu.dtype))
        head, qa = core.bank_head(bank, pairs, regularizer=R)
        frozen_ray, ra = joint.frozen_ray(bank, pairs, R, head, old, head.float())
        assert frozen_ray.any() and qa['relative_residual'] <= 1e-8 and ra['relative_residual'] <= 1e-8
        for scale in risk.STEPS:
            delta = risk.selected_delta(frozen_ray, dict(chosen_scale=scale), old)
            assert torch.equal(delta, (scale*frozen_ray).float().double()) and not delta[:, old:].any()
        projection = torch.cat((head.float().double(), frozen_ray), 1).numpy()
        projected = risk.drift_radius(projection, [S.numpy()])
        assert projected.shape == (46,) and np.isfinite(projected).all() and projected[:23].sum() > 0 and not projected[44:].any()
        assert torch.equal(bank['Q'], snapshot['Q']) and torch.equal(bank['mu'], snapshot['mu']) and bank['n'] == snapshot['n']
        assert not risk.selected_delta(frozen_ray, dict(chosen_scale=0.), old).any()
        highdim.append(dict(control=arm, dimensions=[d,k], native_residual=qa['relative_residual'],
                            ray_residual=ra['relative_residual'], nonzero_frozen_ray=True,
                            four_candidate_bits_exact=True, native_statistics_unchanged=True,
                            full_dimension_transport_projection=True))
    return dict(status='PASS', synthetic_only=True, production_integrated=False,
                GPU_processes=0, medical_images_read=0, medical_checkpoint_reads=0,
                medical_val_reads=0, optimizer_updates=0, policy_updates=0, test_accessed=False,
                full_highdimensional_native_checks=highdim, contribution_modes=adapted,
                positive_fixture_admitted=True, rare_damage_positive_macro_refused=True,
                integrated_positive_with_synthetic_calibration_admitted=True,
                zero_numeric_harm_not_structurally_vetoed=True, binomial_tail_oracle_checked=True,
                missing_drift_and_calibration_identity=True, nodewise_box_bounds_checked=True,
                repeated_drift_no_cancellation=True, covariance_borrow_is_distribution_mixture=True,
                donor_similarity_not_confidence=True, singleton_support_not_increased=True,
                joint_v2_zero_uncertainty_probabilities_exact=True, new_guard_and_ties_checked=True,
                scope_and_parameter_mutations_refused=True, clinical_confidence_calibrated=False,
                pipeline_seconds=time.monotonic()-start)


if __name__ == '__main__':
    print(json.dumps(run(os.environ['Q128_PROTOCOL']), indent=2, allow_nan=False))
