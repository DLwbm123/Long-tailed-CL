"""Bounded read-only acceptance of a stopped, incomplete experiment; no rescue."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
import json
import signal
import time
import resource
from pathlib import Path
import numpy as np
import torch
from verify import numpy_objective


def main(cfg):
    tick = time.time()
    remaining = int(cfg['hard_deadline_epoch'] - tick - 60)
    assert remaining > 0
    signal.alarm(min(30, remaining))
    torch.set_num_threads(4)
    root = Path(cfg['output'])
    read = lambda n: json.loads((root/n).read_text())
    pack = torch.load(root/'partial_stage_0.private.pt', map_location='cpu', weights_only=False)
    cost = read('COSTS.private.json')
    assert read('QUALIFICATION.json')['state'] == 'PASS'
    assert cost['science_train_encodes'] == 60129 and cost['validation_encodes'] == 0
    assert not (root/'QUERY_OPEN.private.json').exists() and not (root/'query.private.pt').exists()
    assert not list(root.glob('scores_*.private.pt'))
    assert pack['seen'] == cfg['order'][:10] and set(pack['covers']) == set(pack['seen'])
    assert set(pack['models']) == {'POINT', 'CLASS'} and cost['optimizer_updates']['CELL'] == 0
    audits = read('GEOMETRY_AUDITS.json')
    assert sum(v['covered_rows'] for v in audits.values()) == 60129
    assert all(v['state'] == 'PASS' and v['coverage_violation'] <= 1e-10 for v in audits.values())
    checks = {}
    for arm, model in pack['models'].items():
        xs, ys, rs, ms = [], [], [], []
        for j, c in enumerate(pack['seen']):
            v = pack['covers'][c]
            x, mass, radius = (v[k].numpy() for k in ['centers', 'weights', 'radii'])
            assert abs(float(mass.sum()) - 1) <= 1e-9
            assert len(x) <= 32 and np.all(radius >= 0)
            xs.append(x); ys.extend([j] * len(x)); ms.append(mass / 10)
            rs.append(np.zeros_like(radius) if arm == 'POINT' else np.full_like(radius, radius.max()))
        theta = model.numpy()
        value, gradient = numpy_objective(theta, np.concatenate(xs), np.asarray(ys),
            np.concatenate(rs), np.concatenate(ms), cfg, True)
        inf = float(abs(gradient).max())
        saved = pack['fits'][arm]
        objective_error = abs(value - saved['loss'])
        gradient_error = abs(inf - saved['gradient_inf'])
        assert objective_error < 1e-9 and gradient_error < 1e-9
        checks[arm] = dict(objective=value, gradient_inf=inf,
            saved_objective_error=objective_error, saved_gradient_inf_error=gradient_error,
            passes_original_gradient_gate=bool(inf <= cfg['accept_gradient_inf']),
            coefficient_norm=float(np.linalg.norm(theta)), parameters=int(theta.size))
    assert checks['POINT']['passes_original_gradient_gate']
    assert not checks['CLASS']['passes_original_gradient_gate']
    result = dict(state='PASS_PARTIAL_FAILURE_RECORD_ONLY', checks=checks,
        scientific_outcome='FIXED_SOLVER_GATE_FAILED_INCOMPLETE', CELL_fitted=False,
        six_stage_result_available=False, new_encodes=0, new_training=0,
        new_validation_scores=0, independent_objective_gradient_evaluations=2,
        objective_feature_rows=sum(len(v['weights']) for v in pack['covers'].values()) * 2,
        geometry_audit_reused_from_independent_arrival_check=True,
        wall_seconds=time.time()-tick, RSS_max_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        started=tick, finished=time.time())
    (root/'PARTIAL_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
