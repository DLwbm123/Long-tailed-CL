"""Synthetic CPU prototype: conditional risk for an unchanged candidate ray.

No medical loaders, checkpoint readers, optimizer or production entry point.
Predictive bands below are model assumptions, not empirical coverage bounds.
"""
import numpy as np

VERSION = 'gain_conditional_reservoir_boundary_v4'
STEPS = (.125, .25, .5, 1.)
HEADS = ('CRISK', 'NO_RESERVOIR', 'NO_DRIFT', 'NO_DONOR_TRUST', 'NO_EDGE_GUARD')
DRAWS = 4096
SEED = 1074005
PSEUDOCOUNT = 16.
PRIOR_CAP = 2.
LOWER_QUANTILE = .1
EDGE_BUDGET = .25


def finite(*values):
    if any(not np.isfinite(v).all() for v in values):
        raise ValueError('Nonfinite conditional risk input')


def validate_design(p):
    expected = dict(version=VERSION, status='CPU_DESIGN_ONLY',
                    direction_changed=False, steps=list(STEPS), heads=list(HEADS),
                    production_integrated=False, GPU_launch_authorized=False,
                    medical_data_access_authorized=False, checkpoint_access_authorized=False,
                    validation_retuning=False, test_accessed=False,
                    optimizer_updates=0, policy_updates=0, primary_control='local_transport',
                    posterior_draws=DRAWS, seed=SEED, jeffreys_count=.5,
                    model_prior_cap=PRIOR_CAP, support_pseudocount=PSEUDOCOUNT,
                    utility_quantile=LOWER_QUANTILE, relative_edge_budget=EDGE_BUDGET,
                    independent_calibration_ready=False,
                    risk_interpretation='conditional_predictive_not_certificate')
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen synthetic design scope differs')


def rates(model):
    a = np.asarray([model[k] for k in ('off_error', 'rescue_given_wrong', 'harm_given_correct')], dtype=float)
    finite(a)
    if ((a < 0) | (a > 1)).any():
        raise ValueError('Invalid conditional probabilities')
    return a


def provenance(evidence, expected):
    if evidence.get('metadata') != expected or evidence.get('basis') not in ('synthetic', 'training_inner', 'independent_train_only'):
        raise ValueError('Evidence must bind exact source/control/task/class/candidate')
    if evidence.get('val_accessed') or evidence.get('test_accessed'):
        raise ValueError('Val/test cannot fit conditional risk')
    independent = evidence.get('classifier_independent')
    if type(independent) is not bool or independent != (evidence['basis'] == 'independent_train_only'):
        raise ValueError('Training exposure cannot become independent calibration')
    if independent and (evidence.get('lineage') != dict(encoder_exposed=False, head_exposed=False, bank_exposed=False, ray_exposed=False)):
        raise ValueError('Independent provenance requires full exposure exclusion')


def boundary_summary(before, after, labels, truth, old):
    """Measure relative slack erosion on OFF-correct target rows, not mean drift."""
    before, after, labels = np.asarray(before), np.asarray(after), np.asarray(labels)
    finite(before, after)
    if before.ndim != 2 or after.shape != before.shape or labels.shape != (len(before),) or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels >= before.shape[1]).any() or type(truth) is not int or not 0 <= truth < before.shape[1] or type(old) is not int or not 0 <= old <= before.shape[1]:
        raise ValueError('Invalid score/class dimensions')
    if not np.array_equal(before[:, old:], after[:, old:]):
        raise ValueError('New logits must remain fixed')
    if old and (after[:, :old].max(1) > before[:, :old].max(1) + 64*np.finfo(float).eps*np.maximum(1., np.abs(before).max(1))).any():
        raise ValueError('Original new-class guard required')
    mask = (labels == truth) & (before.argmax(1) == truth)
    if not mask.any():
        return dict(correct_support=0, erosion_sum=0., erosion_square_sum=0., tied_correct=0)
    competitors = [j for j in range(before.shape[1]) if j != truth]
    if not competitors:
        return dict(correct_support=int(mask.sum()), erosion_sum=0., erosion_square_sum=0., tied_correct=0)
    off = before[mask, truth, None] - before[mask][:, competitors]
    on = after[mask, truth, None] - after[mask][:, competitors]
    scale = np.maximum(np.abs(before[mask]).max(1), np.finfo(float).tiny)
    floor = np.maximum(np.finfo(float).tiny, 64*np.finfo(float).eps*scale[:, None])
    erosion = np.clip((off-on)/np.maximum(off, floor), 0., 1.).max(1)
    # Losing a tie is maximal erosion; an unchanged exact tie contributes zero.
    return dict(correct_support=int(mask.sum()), erosion_sum=float(erosion.sum()),
                erosion_square_sum=float(np.square(erosion).sum()),
                tied_correct=int((off.min(1) <= floor[:, 0]).sum()))


def donor_model(target, donors, mode):
    base = rates(target); total = sum(d['weight'] for d in donors)
    if donors and (not np.isfinite(total) or total <= 0):
        raise ValueError('Invalid donor proposal weights')
    r, h, used = base[1], base[2], []
    for donor in donors:
        dm = rates(donor)
        if type(donor['support']) is not int or donor['support'] < 1 or type(donor['age']) is not int or donor['age'] < 0:
            raise ValueError('Invalid actual donor support/age')
        finite(donor['weight'], donor['boundary_mismatch_z'], donor['drift_z'])
        if donor['weight'] < 0 or donor['boundary_mismatch_z'] < 0 or donor['drift_z'] < 0 or type(donor['evidence_available']) is not bool:
            raise ValueError('Invalid boundary transfer evidence')
        rho = (donor['support']/(donor['support']+PSEUDOCOUNT)/(1+donor['age']) /
               (1+donor['boundary_mismatch_z']**2+donor['drift_z']**2)) if donor['evidence_available'] else 0.
        if mode == 'NO_DONOR_TRUST':rho = 1.
        weight = .4*donor['weight']/total*rho
        if donor.get('rescue_prior_available', True):r += weight*(dm[1]-base[1])
        if donor.get('harm_prior_available', True):h += weight*(dm[2]-base[2])
        used.append(dict(weight=weight, credibility_score=rho, actual_support=donor['support']))
    return dict(off_error=base[0], rescue_given_wrong=r, harm_given_correct=h,
                **{k:target[k] for k in ('rescue_prior_available','harm_prior_available') if k in target}), used


def estimate_class(evidence, model, donors, drift, edge, expected, old_class, mode='CRISK'):
    if mode not in HEADS or type(old_class) is not bool:
        raise ValueError('Invalid contribution mode/class role')
    provenance(evidence, expected); model, donor_audit = donor_model(model, donors, mode)
    n, correct, rescued, harmed = [evidence[k] for k in ('n', 'correct_off', 'rescued', 'harmed')]
    if any(type(v) is not int or v < 0 for v in (n, correct, rescued, harmed)) or not n or expected['support'] != n or correct > n or rescued > n-correct or harmed > correct or not old_class and harmed:
        raise ValueError('Events exceed OFF error/correct reservoirs')
    wrong = n-correct
    if type(drift.get('available')) is not bool:
        raise ValueError('Drift evidence availability required')
    if old_class and mode != 'NO_DRIFT' and not drift['available']:
        return dict(ready=False, reason='missing_current_state_drift_sensitivity')
    sq, sr, sh = (drift.get('off_error_sensitivity', 0.), drift.get('rescue_sensitivity', 0.), drift.get('harm_sensitivity', 0.))
    finite(sq, sr, sh)
    if not 0 <= sq <= 1 or not 0 <= sr <= 1 or not 0 <= sh <= 1:
        raise ValueError('Invalid model-conditional drift sensitivity')
    if mode == 'NO_DRIFT' or not old_class:sq = sr = sh = 0.
    if old_class and mode != 'NO_EDGE_GUARD' and edge is None:
        return dict(ready=False, reason='missing_OFF_correct_joint_slack_statistics')
    edge_score = 0.
    if edge is not None:
        if edge['correct_support'] != correct:
            raise ValueError('Boundary support must match OFF-correct reservoir')
        es, es2 = edge['erosion_sum'], edge['erosion_square_sum']; finite(es, es2)
        if not 0 <= es <= correct or not 0 <= es2 <= es+1e-12 or correct and es2+1e-12 < es*es/correct:
            raise ValueError('Invalid bounded relative erosion moments')
        if correct:
            mean = es/correct; variance = max(0., es2/correct-mean*mean)
            edge_score = float(min(1., mean+np.sqrt(variance/correct)))
    strength = PRIOR_CAP*n/(n+PSEUDOCOUNT)
    rescue_strength = strength if model.get('rescue_prior_available', True) else 0.
    harm_strength = strength if model.get('harm_prior_available', True) else 0.
    # Reservoir and conditional likelihoods are separate; model nodes are not n.
    stream = [SEED, expected['task'], expected['label']]
    q = np.random.default_rng(np.random.SeedSequence(stream+[0])).beta(wrong+.5, correct+.5, DRAWS)
    if mode == 'NO_RESERVOIR':q[:] = model['off_error']
    r = np.random.default_rng(np.random.SeedSequence(stream+[1])).beta(rescued+.5+rescue_strength*model['rescue_given_wrong'], wrong-rescued+.5+rescue_strength*(1-model['rescue_given_wrong']), DRAWS)
    h = np.random.default_rng(np.random.SeedSequence(stream+[2])).beta(harmed+.5+harm_strength*model['harm_given_correct'], correct-harmed+.5+harm_strength*(1-model['harm_given_correct']), DRAWS)
    if not old_class:h[:] = 0.
    rescue, harm = q*r, (1-q)*h
    adverse_q = np.maximum(0., q-sq)
    conservative_rescue = adverse_q*np.maximum(0., r-sr)
    conservative_harm = (1-adverse_q)*np.minimum(1., h+sh)
    return dict(ready=True, rescue_mean=float(rescue.mean()), harm_mean=float(harm.mean()),
                off_error_mean=float(q.mean()), empirical_off_error=wrong/n,
                reservoir_wrong_n=wrong, reservoir_correct_n=correct,
                rescue_conditional_n=wrong, harm_conditional_n=correct,
                rescue_data_identified=wrong > 0, harm_data_identified=correct > 0,
                model_prior_pseudocount=strength, target_actual_n=n, donor_audit=donor_audit,
                rescue_prior_pseudocount=rescue_strength, harm_prior_pseudocount=harm_strength,
                drift_off_error_sensitivity=sq, drift_rescue_sensitivity=sr, drift_harm_sensitivity=sh,
                relative_edge_score=edge_score, edge_budget=EDGE_BUDGET*n/(n+PSEUDOCOUNT),
                edge_failed=bool(old_class and mode != 'NO_EDGE_GUARD' and edge_score > EDGE_BUDGET*n/(n+PSEUDOCOUNT)),
                net_samples=conservative_rescue-conservative_harm,
                classifier_independent=evidence['classifier_independent'],
                conditional_predictive_not_certificate=True)


def select(candidates, old, mode='CRISK'):
    if mode not in HEADS or len(candidates) != len(STEPS) or type(old) is not int or not candidates or not 0 <= old <= len(candidates[0]):
        raise ValueError('Exactly four frozen candidates required')
    k = len(candidates[0]); rows = []; chosen = 0.; best = 1e-12
    if old == 0:
        return dict(chosen_scale=0., candidates=[dict(scale=s, ready=True, admitted=False, reason='no_old_classes_exact_identity') for s in STEPS],
                    status='MODEL_CONDITIONAL_SELECTION', clinical_or_coverage_certificate=False)
    for step, classes in zip(STEPS, candidates):
        if len(classes) != k:raise ValueError('Class coverage differs')
        if any(not c['ready'] for c in classes):
            rows.append(dict(scale=step, ready=False, admitted=False, reason='missing_evidence'));continue
        draws = np.stack([c['net_samples'] for c in classes]);finite(draws)
        if draws.shape != (k, DRAWS):raise ValueError('Posterior draw dimensions differ')
        macro = draws.mean(0); utility = float(np.quantile(macro, LOWER_QUANTILE))
        failed = [i for i,c in enumerate(classes[:old]) if c['edge_failed']]
        admitted = utility > 1e-12 and (not failed or mode == 'NO_EDGE_GUARD')
        rows.append(dict(scale=step, ready=True, admitted=admitted, predictive_net_mean=float(macro.mean()),
                         conditional_utility=utility, edge_failed_classes=failed))
        if admitted and utility > best:best, chosen = utility, step
    return dict(chosen_scale=chosen, candidates=rows,
                status='NOT_READY' if any(not r['ready'] for r in rows) else 'MODEL_CONDITIONAL_SELECTION',
                clinical_or_coverage_certificate=False)


def selected_delta(ray, choice, old):
    ray = np.asarray(ray); finite(ray)
    if ray.ndim != 2 or type(old) is not int or not 0 <= old <= ray.shape[1] or ray[:, old:].any() or choice['chosen_scale'] not in (0.,)+STEPS:
        raise ValueError('Fixed-ray/new-column scope differs')
    tolerance = 1e-8+np.finfo(np.float32).eps*np.abs(ray[:, :old]).sum(1)
    if (np.abs(ray[:, :old].sum(1)) > tolerance).any():
        raise ValueError('Old common component differs')
    if choice.get('status') == 'NOT_READY':raise ValueError('Missing evidence cannot produce a medical sidecar')
    return (choice['chosen_scale']*ray).astype(np.float32).astype(np.float64)
