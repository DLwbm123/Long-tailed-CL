import numpy as np
import torch

ARMS = ['F', 'L', 'P', 'C', 'Q', 'R']
CONTROLS = ['F', 'C', 'Q', 'R', 'L']


def points(pred, y):
    recall = [float((pred[y == c] == c).double().mean()) for c in range(4)]
    return dict(BA=sum(recall)/4, recall=recall,
                correct=[int(((pred == c) & (y == c)).sum()) for c in range(4)],
                counts=[int((y == c).sum()) for c in range(4)])


def pairs(base, pred, y):
    rows = []
    for c in range(4):
        b, p = base[y == c], pred[y == c]
        bo, po = b == c, p == c
        r, h = int((~bo & po).sum()), int((bo & ~po).sum())
        rows.append(dict(class_index=c, count=len(b), rescued=r, harmed=h, net_correct=r-h,
                         both_correct=int((bo & po).sum()), both_wrong=int((~bo & ~po).sum()),
                         harmed_destination=torch.bincount(p[bo & ~po], minlength=4).tolist()))
    gain = 25*sum(v['net_correct']/v['count'] for v in rows)
    assert abs(gain-100*(points(pred, y)['BA']-points(base, y)['BA'])) < 1e-10
    return dict(BA_gain_pp=gain, classes=rows)


def gate(tab):
    if not all(a in tab for a in ARMS):
        return dict(pass_all=False, missing=True)
    p, f = tab['P'], tab['F']
    dr = np.array(p['recall'])-f['recall']
    contrasts = {a: dict(BA_gain_pp=100*(p['BA']-tab[a]['BA']),
                        class2_gain_pp=100*(p['recall'][2]-tab[a]['recall'][2])) for a in CONTROLS}
    checks = dict(BA_vs_F_at_least1=p['BA']-f['BA'] >= .01-1e-12,
                  class2_vs_F_at_least5=dr[2] >= .05-1e-12,
                  other_class_cost_within1=min(dr[[0, 1, 3]]) >= -.01-1e-12,
                  BA_all_controls_positive=all(contrasts[a]['BA_gain_pp'] > 0 for a in CONTROLS[1:]),
                  class2_all_controls_nonnegative=all(contrasts[a]['class2_gain_pp'] >= -1e-12 for a in CONTROLS[1:]),
                  class2_vs_R_at_least5=contrasts['R']['class2_gain_pp'] >= 5-1e-12)
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(**checks, pass_all=all(checks.values()), contrasts=contrasts)


def evaluate(pred, y, groups):
    assert all(a in pred for a in ARMS)
    summary = {a: points(pred[a], y) for a in ARMS}
    g = gate(summary)
    result = dict(summary=summary, development_gate=g,
                  paired_P_vs={a: pairs(pred[a], pred['P'], y) for a in CONTROLS},
                  method_success=False, test_used=False, future_used=False,
                  development_exposed=True, not_independent_confirmation=True)
    if not g['pass_all']:
        return dict(**result, decision='STOP_READOUT_CONSISTENT_DEVELOPMENT', mechanism_bootstrap_performed=False)
    rng = np.random.default_rng(74271)
    rec = {a: np.zeros((10000, 4)) for a in ARMS}
    yy = y.numpy()
    for c in range(4):
        ids = np.where(yy == c)[0]
        keys = sorted({groups[i] for i in ids})
        members = [[i for i in ids if groups[i] == key] for key in keys]
        sizes = np.array([len(v) for v in members])
        draw = rng.integers(0, len(keys), (10000, len(keys)))
        den = sizes[draw].sum(1)
        for a in ARMS:
            ok = pred[a].numpy() == yy
            hits = np.array([ok[v].sum() for v in members])
            rec[a][:, c] = hits[draw].sum(1)/den
    contrasts = {}
    for a in CONTROLS:
        for name, ds in [('BA', 100*(rec['P'].mean(1)-rec[a].mean(1))),
                         ('class2', 100*(rec['P'][:, 2]-rec[a][:, 2]))]:
            contrasts[name+'_vs_'+a] = dict(CI95_pp=np.quantile(ds, [.025, .975]).tolist(),
                                            CI10_adjusted_pp=np.quantile(ds, [.05/20, 1-.05/20]).tolist())
    support = all(v['CI10_adjusted_pp'][0] > 0 for v in contrasts.values())
    return dict(**result, decision='DEVELOPMENT_READOUT_CONSISTENCY_SIGNAL' if support else
                'STOP_READOUT_CONSISTENCY_EVIDENCE_INSUFFICIENT', mechanism_bootstrap_performed=True,
                contrasts=contrasts, bootstrap=dict(seed=74271, draws=10000, family=10))


def synthetic():
    y = torch.arange(80) % 4
    groups = ['toy'+str(i) for i in range(80)]
    bad = y.clone()
    bad[::3] = (bad[::3]+1) % 4
    pred = {a: bad.clone() for a in ARMS}
    pred['P'] = y.clone()
    good = evaluate(pred, y, groups)
    assert good['development_gate']['pass_all'] and len(good['contrasts']) == 10
    pred['P'] = bad.clone()
    failed = evaluate(pred, y, groups)
    assert not failed['development_gate']['pass_all'] and not gate({})['pass_all']
    import json
    json.dumps([good, failed], allow_nan=False)
    return dict(ten_contrasts_success_failure_missing_JSON=True, paired_identity=True, discrete_gate=True)
