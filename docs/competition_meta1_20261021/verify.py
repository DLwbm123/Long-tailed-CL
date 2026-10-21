"""Independent CPU NumPy MLP, moment reconstruction and prefix metrics."""
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch


def embed(x, state):
    x = np.asarray(x, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    x = x @ state['phi.0.weight'].T + state['phi.0.bias']
    x = x / (1 + np.exp(-x))
    x = x @ state['phi.2.weight'].T + state['phi.2.bias']
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def scores(z, mean, var, state):
    b, k = len(z), len(mean)
    result = np.zeros((b, k), dtype=np.float64)
    for c in range(k):
        for d in range(c):
            both = []
            for i, j in [(c, d), (d, c)]:
                inputs = np.concatenate((z, np.broadcast_to(np.r_[mean[i], var[i], mean[j], var[j]], (b, 256))), 1)
                h = inputs @ state['h.0.weight'].T + state['h.0.bias']
                h = h / (1 + np.exp(-h))
                both.append((h @ state['h.2.weight'].T + state['h.2.bias'])[:, 0])
            a = np.tanh(both[0] - both[1]) * (8 / (k - 1))
            result[:, c] += a
            result[:, d] -= a
    return result


def risk_objectives(risk):
    return dict(META=float(risk.mean()), WORST=float(risk.mean(0).max()), PAIRED=float(risk.max()))


def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    torch.set_num_threads(4)
    tick = time.time()
    root = Path(cfg['output'])
    from data import read_rows
    train, val = read_rows(cfg, 'train'), read_rows(cfg, 'val')
    patient_counts = Counter((r['label'], r['group']) for r in train)
    assert len(train) == 68058 and len(val) == 300
    assert not {r['group'] for r in train} & {r['group'] for r in val}
    pools = json.loads((root / 'PATIENT_POOLS.private.json').read_text())
    assert not set(pools['support_patients']) & set(pools['query_patients'])
    base_patients = {r['group'] for r in train if r['label'] in cfg['order'][:10]}
    assert set(pools['support_patients']) | set(pools['query_patients']) == base_patients
    model_seal = json.loads((root / 'MODEL_SEAL.json').read_text())
    fit_seal = json.loads((root / 'FIT_SEAL.json').read_text())
    opened = json.loads((root / 'QUERY_OPEN.private.json').read_text())
    assert model_seal['at'] < fit_seal['at'] < opened['at']
    assert model_seal['future_class_encodes'] == model_seal['validation_encodes'] == 0
    assert model_seal['base_train_encodes'] == 60129
    assert model_seal['steps_per_arm'] == 3000
    curves = json.loads((root / 'TRAINING_CURVES.json').read_text())
    assert curves[0]['step'] == 1 and curves[-1]['step'] == 3000
    for record in curves:
        for arm, details in record['arms'].items():
            expected_risk = risk_objectives(np.array(details['risk_grid'], dtype=np.float64))[arm]
            assert abs(details['loss'] - expected_risk) <= 2e-6
    seals = json.loads((root / 'STAGE_SEALS.json').read_text())
    expected_prefixes = [sum(cfg['task_sizes'][:i + 1]) for i in range(6)]
    assert [s['classes'] for s in seals] == [cfg['order'][:n] for n in expected_prefixes]
    assert all(s['validation_encodes'] == 0 for s in seals)
    query = torch.load(root / 'query.private.pt', map_location='cpu', weights_only=True)
    saved = torch.load(root / 'scores.private.pt', map_location='cpu', weights_only=True)
    labels = query['labels'].numpy()
    assert labels.tolist() == [c for c in cfg['order'] for _ in range(15)]
    assert torch.equal(query['labels'], saved['labels'])
    all_scores = {}
    err = 0.
    mismatches = 0
    state_error = 0.
    for arm in cfg['arms']:
        assert time.time() < cfg['hard_deadline_epoch'] - 60
        state = {k: v.numpy().astype(np.float64) for k, v in torch.load(root / (arm + '.private.pt'), map_location='cpu', weights_only=True).items()}
        moments = torch.load(root / (arm + '_moments.private.pt'), weights_only=True, map_location='cpu')
        qz = embed(query['features'].numpy(), state)
        means = {s: [] for s in cfg['evaluation_support_settings']}
        variances = {s: [] for s in means}
        for label in cfg['order']:
            # Private per-patient aggregate audit ledger never enters any optimizer.
            audit = torch.load(root / ('audit_' + arm + '_' + str(label) + '.private.pt'), weights_only=True, map_location='cpu')
            groups = sorted({r['group'] for r in train if r['label'] == label}, key=int)
            assert audit['groups'] == groups
            counts = np.array([patient_counts[label, g] for g in groups])
            assert counts.tolist() == audit['counts'].tolist()
            ranked = sorted(range(len(groups)), key=lambda j: ((int(groups[j]) * 1103515245 + cfg['seed']) % 2147483648, int(groups[j])))
            for setting in means:
                chosen = list(range(len(groups))) if setting == 'all' else ranked[:1 if setting == 'one' else 4]
                mean = (audit['sum'].numpy()[chosen] / counts[chosen, None]).mean(0)
                second = (audit['sumsq'].numpy()[chosen] / counts[chosen, None]).mean(0)
                var = np.maximum(second - mean ** 2, 0)
                stored = moments[setting][str(label)]
                state_error = max(state_error, float(np.max(np.abs(mean - stored['mean'].numpy()))), float(np.max(np.abs(var - stored['variance'].numpy()))))
                means[setting].append(mean)
                variances[setting].append(var)
        assert state_error <= 1e-10
        all_scores[arm] = {}
        for setting in means:
            all_scores[arm][setting] = []
            for task, n in enumerate(expected_prefixes):
                snapshot = torch.load(root / ('stage_' + str(task) + '_' + arm + '.private.pt'), weights_only=True, map_location='cpu')
                for c in cfg['order'][:n]:
                    assert torch.equal(snapshot[setting][str(c)]['mean'], moments[setting][str(c)]['mean'])
                    assert torch.equal(snapshot[setting][str(c)]['variance'], moments[setting][str(c)]['variance'])
                score = scores(qz, np.array(means[setting][:n]), np.array(variances[setting][:n]), state)
                err = max(err, float(np.max(np.abs(score - saved['scores'][arm][setting][task].numpy()))))
                mismatches += int(np.sum(score.argmax(1) != saved['scores'][arm][setting][task].numpy().argmax(1)))
                all_scores[arm][setting].append(score)
    assert err <= cfg['score_absolute_tolerance'] and mismatches == 0
    stages = []
    for t, n in enumerate(expected_prefixes):
        seen = cfg['order'][:n]
        mask = np.isin(labels, seen)
        metric = {}
        for a in cfg['arms']:
            prediction = np.array(seen)[all_scores[a]['all'][t].argmax(1)]
            correct = {str(c): int(np.sum((labels == c) & (prediction == c))) for c in seen}
            assert sum(np.sum(labels[mask] == c) for c in seen) == 15 * n
            metric[a] = dict(correct=correct, total_correct=sum(correct.values()), BA_percent=sum(correct.values()) / (15 * n) * 100)
        stages.append(dict(task=t, seen=seen, metrics=metric))
    mean = {a: sum(s['metrics'][a]['BA_percent'] for s in stages) / 6 for a in cfg['arms']}
    groups = {a: {name: sum(stages[-1]['metrics'][a]['correct'][str(c)] for c in cs) for name, cs in dict(tail4=[7, 14, 15, 16], tail7=[15, 16, 7, 14, 18, 12, 5], initial10=cfg['order'][:10], later10=cfg['order'][10:]).items()} for a in cfg['arms']}
    gates = {}
    stress = {}
    for a in cfg['arms']:
        class_risk = []
        for setting in cfg['evaluation_support_settings']:
            for c in cfg['order']:
                losses = []
                for t, n in enumerate(expected_prefixes):
                    if c not in cfg['order'][:n]:
                        continue
                    score = all_scores[a][setting][t][labels == c]
                    shifted = score - score.max(1, keepdims=True)
                    loss = np.log(np.exp(shifted).sum(1)) - shifted[:, cfg['order'].index(c)]
                    losses.append(float(loss.mean()))
                class_risk.append(dict(setting=setting, class_id=c, trajectory_CE=float(np.mean(losses))))
        stress[a] = dict(class_risks=class_risk, worst_class_trajectory_CE=max(v['trajectory_CE'] for v in class_risk))
    for a in ['META', 'WORST']:
        gates[a] = dict(final=stages[-1]['metrics']['PAIRED']['total_correct'] >= stages[-1]['metrics'][a]['total_correct'] + 6,
                        mean=mean['PAIRED'] >= mean[a] + 1,
                        tail4=groups['PAIRED']['tail4'] >= groups[a]['tail4'] + 2,
                        initial10=groups['PAIRED']['initial10'] >= groups[a]['initial10'],
                        later10=groups['PAIRED']['later10'] >= groups[a]['later10'],
                        support_stress=stress['PAIRED']['worst_class_trajectory_CE'] <= stress[a]['worst_class_trajectory_CE'] - .02)
    cost = json.loads((root / 'COSTS.json').read_text())
    assert cost['qualification_encodes'] == 20 and cost['science_train_encodes'] == 68058 and cost['validation_encodes'] == 300
    assert cost['qualification_updates'] == 3 and all(v == 3000 for v in cost['meta_updates'].values())
    assert all(v == 68058 for v in cost['state_embedding_rows'].values()) and all(v == 5400 for v in cost['validation_embedding_rows'].values())
    assert len(set(cost['embedding_rows'].values())) == len(set(cost['support_view_rows'].values())) == len(set(cost['query_rows'].values())) == 1
    resources = json.loads((root.parent / 'guardian_resources.private.json').read_text())
    assert resources['peak_output_bytes'] < cfg['output_budget_bytes'] and resources['RSS_sampled_peak_bytes'] < cfg['RSS_budget_bytes']
    output = [('RESULTS.json', dict(stages=stages, mean_BA_percent=mean, frequency=groups)),
              ('SUPPORT_STRESS.json', stress),
              ('DECISION.json', dict(gates=gates, proceed=all(v for arm in gates.values() for v in arm.values()), no_parameter_rescue=True)),
              ('VERIFICATION.json', dict(state='PASS', max_score_error=err, prediction_mismatches=mismatches, moment_error=state_error, patient_overlap=0, new_encodes=0, new_model_training=0, wall_seconds=time.time() - tick))]
    for name, obj in output:
        (root / name).write_text(json.dumps(obj, indent=2) + '\n')
    (root / 'status.private.json').write_text(json.dumps(dict(state='COMPLETE_VERIFIED', method_gate_pass=output[2][1]['proceed'], delivery_pending=True, updated=time.time())))


if __name__ == '__main__':
    main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
