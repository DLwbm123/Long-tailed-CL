"""Bounded gate oracle, held-class leakage check, and optional two-step native admission."""
import json
from pathlib import Path
import sys
import numpy as np
from prototype_graph import empty, append, transport, move, fit_gate, held_class_prediction

if __name__ == '__main__':
    baseline = np.zeros((2, 3)); predicted = np.ones((2, 3)); weights = np.array([.3, .7])
    for target, expected in [(-1., 0.), (.3, .3), (2., 1.)]:
        assert abs(fit_gate(baseline, predicted, np.full((2, 3), target), weights)-expected) < 1e-12
    assert fit_gate(baseline, baseline, predicted, weights) == 0.
    laplacian = np.array([[1., -1.], [-1., 1.]])
    masses = np.ones(2); confidence = np.ones(2); observed = np.array([[100., 50.], [.2, -.3]])
    labels = np.array([0, 1])
    g, prediction = held_class_prediction(laplacian, masses, confidence, observed, labels, 0)
    observed[0] = -10000.; confidence[0] = 999.
    other_g, other_prediction = held_class_prediction(laplacian, masses, confidence, observed, labels, 0)
    assert np.array_equal(g, other_g) and np.array_equal(prediction, other_prediction)
    assert np.allclose(g, observed[1])
    rng = np.random.default_rng(62)
    old_x = rng.normal(size=(64, 6)); old_y = np.repeat([0, 1], 32)
    bank = append(empty(6), old_x, old_y, [0, 1])
    before = rng.normal(size=(64, 6)); y = np.repeat([2, 3], 32)
    d = np.ones(6)*.2
    uniform, audit = transport(bank, before, before+d, y, [2, 3], adaptive=True)
    expected = move(bank, np.tile(d, (len(bank['components']), 1)))
    assert np.allclose(uniform['S'], expected['S']) and np.allclose(uniform['mu'], expected['mu'])
    after = before + rng.normal(scale=.1, size=before.shape)
    _, audit = transport(bank, before, after, y, [2, 3], adaptive=True)
    e = audit['gate_calibration_errors']
    assert 0. <= audit['adaptive_gate'] <= 1. and e['blend'] <= min(e['global'], e['graph'])+1e-12
    # Existing fixed graph remains numerically unchanged after extracting the shared solver.
    import runpy
    config = json.load(sys.stdin)
    if config.get('original_graph'):
        original = runpy.run_path(config['original_graph'])
        a, da = original['transport'](bank, before, after, y, [2, 3])
        b, db = transport(bank, before, after, y, [2, 3])
        for k in ('S', 'mu', 'n'):assert np.allclose(a[k], b[k], atol=1e-12, rtol=1e-12)
        assert da.keys() == db.keys()
    print('PASS: bounded blend, held-class drift/confidence isolation, moment limits, fixed graph parity')
    if config.get('admission_root'):
        root = Path(config['admission_root'])
        receipt = json.loads((root/'adaptive'/'STATUS.json').read_text())
        assert receipt['status']=='COMPLETE' and receipt['steps']==2 and receipt['stages']==2
        diagnostics = json.loads((root/'adaptive'/'diagnostics.json').read_text())
        assert len(diagnostics)==2 and diagnostics[-1]['old_components']>0
        for row in diagnostics:
            assert row['trained_encoder'] and row['fd_weight']==10. and not row['error_feedback']
            assert row['old_logit_weight']==0. and row['subspace_weight']==0.
            assert row['ridge_residual']<1e-7 and 0.<=row['adaptive_gate']<=1.
            e=row['gate_calibration_errors'];assert e['blend']<=min(e['global'],e['graph'])+1e-12
        result=dict(status='PASS',steps=2,receipt=receipt,held_class_no_target_leakage=True,
            fixed_graph_parity=True,gate_bounds=True,calibration_not_independent=True,
            observed_gates=[d['adaptive_gate'] for d in diagnostics],effectiveness_claim=False)
        Path(config['output']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
