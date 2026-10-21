"""Independent NumPy objectives, gradients, geometry and stored-query acceptance."""
import json
import os
import time
from pathlib import Path
import numpy as np
from collections import Counter
from scipy.special import logsumexp, softmax


def numpy_objective(t, x, y, radius, mass, cfg, gradient=False):
    w, b = t[:, :-1], t[:, -1]
    logits = x @ w.T + b
    delta = w[None] - w[y, None]
    length = np.sqrt((delta**2).sum(2) + cfg['norm_smoothing']**2)
    length[np.arange(len(x)), y] = 0
    rel = logits - logits[np.arange(len(x)), y, None] + radius[:, None] * length
    value = float(mass @ logsumexp(rel, axis=1) + cfg['ridge']/2 * (t*t).sum())
    if not gradient: return value
    p = softmax(rel, axis=1)
    e = p.copy(); e[np.arange(len(x)), y] -= 1
    out = np.empty_like(t)
    out[:, :-1] = (mass[:, None] * e).T @ x
    out[:, -1] = mass @ e
    safe = np.where(length == 0, 1., length)
    change = mass[:, None, None] * radius[:, None, None] * p[:, :, None] * delta / safe[:, :, None]
    out[:, :-1] += change.sum(0)
    np.add.at(out[:, :-1], y, -change.sum(1))
    out += cfg['ridge'] * t
    return value, out


def geometry_audit(z, groups, cover, chosen, assignment):
    centers = cover['centers'].numpy()
    points = z.numpy()
    counts = Counter(groups)
    mass = np.asarray([1/counts[g] for g in groups]); mass /= mass.sum()
    weights = np.zeros(len(centers)); radii = np.zeros(len(centers))
    max_assigned_violation = 0.
    for start in range(0, len(z), 256):
        block = points[start:start+256]
        distance = np.sqrt(((block[:, None]-centers[None])**2).sum(2))
        ids = assignment[start:start+len(block)].numpy()
        actual = distance[np.arange(len(block)), ids]
        assert np.max(actual-distance.min(1)) <= 1e-10
        np.add.at(weights, ids, mass[start:start+len(block)])
        np.maximum.at(radii, ids, actual)
        max_assigned_violation = max(max_assigned_violation, float(np.max(actual-cover['radii'].numpy()[ids])))
    error = float(max(np.max(abs(weights-cover['weights'].numpy())),
                      np.max(abs(radii + 1e-12-cover['radii'].numpy()))))
    assert error <= 1e-9 and max_assigned_violation <= 1e-10
    assert len({groups[i] for i in chosen}) == len(chosen)
    return dict(state='PASS', independent_geometry_error=error,
                coverage_violation=max_assigned_violation, covered_rows=len(z),
                covered_patients=len(counts), centers=len(centers),
                unique_center_patients=len(chosen), mass_sum=float(weights.sum()))


def main(cfg):
    import torch
    torch.set_num_threads(4)
    tick = time.time(); root = Path(cfg['output'])
    read = lambda n: json.loads((root/n).read_text())
    load = lambda n: torch.load(root/n, map_location='cpu', weights_only=False)
    assert read('QUALIFICATION.json')['state'] == 'PASS'
    covers = load('covers.private.pt')
    assert all(v['state']=='PASS' for v in read('GEOMETRY_AUDITS.json').values())
    if (root/'STOP_THEOREM.private.json').exists():
        t = read('STOP_THEOREM.private.json')
        point = np.asarray(t['common_point'])
        slack = min(float(np.min(v['radii'].numpy()-np.linalg.norm(v['centers'].numpy()-point,axis=1))) for v in covers.values())
        assert slack >= -1e-10
        result = dict(state='PASS', theorem_stop_verified=True, common_intersection_slack=slack,
                      no_query_open=True, new_encodes=0, new_training=0, wall_seconds=time.time()-tick)
    else:
        query = load('query.private.pt')
        x = query['features'].double().numpy(); labels = query['labels'].numpy()
        assert read('FIT_SEAL.json')['at'] < read('QUERY_OPEN.private.json')['at']
        assert read('COSTS.json')['science_train_encodes']==68058 and read('COSTS.json')['validation_encodes']==300
        max_error = 0.; mismatches = 0; grad_error = 0.; largest_grad=0.
        for task in range(6):
            seen = cfg['order'][:sum(cfg['task_sizes'][:task+1])]
            pack = load(f'stage_{task}.private.pt')
            assert pack['seen']==seen and set(pack['covers'])==set(seen)
            for c in seen:
                for key in ['centers','weights','radii']:
                    assert torch.equal(covers[c][key],pack['covers'][c][key])
            idx = np.flatnonzero(np.isin(labels,seen))
            for arm in cfg['arms']:
                t=pack['models'][arm].numpy()
                score=x[idx]@t[:,:-1].T+t[:,-1]
                saved=load(f'scores_{task}_{arm}.private.pt').numpy()
                error=float(np.max(abs(score-saved))); max_error=max(max_error,error)
                mismatches+=int(np.sum(score.argmax(1)!=saved.argmax(1)))
                xs=[];ys=[];rs=[];ms=[]
                for j,c in enumerate(seen):
                    v=pack['covers'][c]; n=len(v['weights']);xs.append(v['centers'].numpy())
                    ys.extend([j]*n); radius=v['radii'].numpy().copy()
                    if arm=='POINT':radius[:]=0
                    if arm=='CLASS':radius[:]=radius.max()
                    rs.append(radius);ms.append(v['weights'].numpy()/len(seen))
                value, gradient=numpy_objective(t,np.concatenate(xs),np.asarray(ys),np.concatenate(rs),np.concatenate(ms),cfg,True)
                fit=pack['fits'][arm]
                assert abs(value-fit['loss'])<=1e-9
                independent_grad=float(np.max(abs(gradient)));largest_grad=max(largest_grad,independent_grad)
                grad_error=max(grad_error,abs(independent_grad-fit['gradient_inf']))
                assert fit['converged'] and independent_grad<=cfg['accept_gradient_inf']+1e-9
                assert error<=cfg['score_absolute_tolerance']
        assert mismatches==0 and grad_error<=1e-9
        from data import read_rows
        train_rows, val_rows = read_rows(cfg, 'train'), read_rows(cfg, 'val')
        assert not {r['group'] for r in train_rows} & {r['group'] for r in val_rows}
        assert read('COVERAGE.json')['patient_overlap']==0
        result=dict(state='PASS',max_score_error=max_error,prediction_mismatches=mismatches,
                    independent_gradient_error=grad_error,largest_gradient_inf=largest_grad,
                    geometry_independently_checked_at_arrival=True,geometry_audit_same_worker=True,
                    score_tables=18,patient_overlap=0,new_encodes=0,new_training=0,wall_seconds=time.time()-tick)
    (root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    state=read('status.private.json');state.update(state='COMPLETE_VERIFIED',verification=result)
    (root/'status.private.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
