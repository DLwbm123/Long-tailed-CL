"""Separate CPU NumPy forward/loss and independent stage/memory/count acceptance."""
import hashlib
import json
import os
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def numpy_forward(x, state, scale):
    x = normalize(np.asarray(x, dtype=np.float64))
    branches = []
    count = sum(k.startswith('branches.') and k.endswith('down.weight') for k in state)
    for i in range(count):
        prefix = f'branches.{i}.'
        h = x @ state[prefix+'down.weight'].T + state[prefix+'down.bias']
        h = h / (1 + np.exp(-h))
        branches.append(normalize(x + h @ state[prefix+'up.weight'].T + state[prefix+'up.bias']))
    z = normalize(np.concatenate(branches, axis=1))
    return z, scale * z @ normalize(state['head']).T


def numpy_loss(x, y, ids, state, arm, cfg):
    _, score = numpy_forward(x, state, cfg['logit_scale'])
    score = score - score.max(1, keepdims=True)
    return float((np.log(np.exp(score).sum(1)) - score[np.arange(len(y)), y]).mean())


def summarize(stages, cfg, counts):
    arms = cfg['arms']
    mean = {a: float(np.mean([s['metrics'][a]['BA_percent'] for s in stages])) for a in arms}
    groups = dict(tail4=[7, 14, 15, 16], tail7=[c for c, n in counts.items() if n < 250],
                  initial10=cfg['order'][:10], later10=cfg['order'][10:])
    frequency = {a: {g: sum(stages[-1]['metrics'][a]['correct'][str(c)] for c in cs)
                     for g, cs in groups.items()} for a in arms}
    main = 'FROZEN_PAST'
    final = lambda a: stages[-1]['metrics'][a]['total_correct']
    gates = dict(final=final(main) >= 91, mean=mean[main] >= cfg['gates']['mean_min_percent'],
                 tail7=frequency[main]['tail7'] >= 25, initial10=frequency[main]['initial10'] >= 44,
                 later10=frequency[main]['later10'] >= 41)
    a = 'ALL_TRAINABLE'
    gates.update(vs_control_final=final(main) >= final(a) + 4,
                 vs_control_mean=mean[main] >= mean[a] + 1,
                 vs_control_tail4=frequency[main]['tail4'] >= frequency[a]['tail4'] + 2,
                 vs_control_initial=frequency[main]['initial10'] >= frequency[a]['initial10'],
                 vs_control_later=frequency[main]['later10'] >= frequency[a]['later10'])
    return dict(stages=stages, mean_BA_percent=mean, frequency=frequency), dict(
        gates=gates, proceed=all(gates.values()), no_parameter_rescue=True,
        evidence_scope='Equal growing architecture with/without old branch freezing; DER-inspired feature adaptation, not a novel method or faithful full DER reproduction.')


def main(cfg):
    from threadpoolctl import threadpool_limits
    from data import read_rows
    threadpool_limits(4)
    torch.set_num_threads(4)
    start = time.time()
    root = Path(cfg['output'])
    train, val = read_rows(cfg, 'train'), read_rows(cfg, 'val')
    assert not {r['group'] for r in train} & {r['group'] for r in val}
    seals = json.loads((root / 'STAGE_SEALS.json').read_text())
    fit = json.loads((root / 'FIT_SEAL.json').read_text())
    opened = json.loads((root / 'QUERY_OPEN.private.json').read_text())
    assert len(seals) == 6 and all(s['at'] < fit['at'] < opened['at'] for s in seals)
    assert all(s['validation_encodes'] == 0 for s in seals)
    query = torch.load(root / 'query.private.pt', map_location='cpu', weights_only=False)
    assert query['labels'].tolist() == [r['label'] for r in val]
    labels = query['labels'].numpy()
    error, mismatches, stages, previous = 0., 0, [], {}
    for task in range(6):
        assert time.time() < cfg['hard_deadline_epoch'] - 60
        seen = cfg['order'][:sum(cfg['task_sizes'][:task+1])]
        pack = torch.load(root / f'stage_{task}.private.pt', map_location='cpu', weights_only=False)
        assert pack['seen'] == seals[task]['seen'] == seen
        assert seals[task]['science_train_encodes'] == sum(r['label'] in seen for r in train)
        expected_steps = cfg['base_updates'] + task * cfg['later_updates']
        assert all(s == expected_steps for s in seals[task]['updates'].values())
        selected = []
        for c in seen:
            patients = {}
            for i, r in enumerate(train):
                if r['label'] == c:
                    patients.setdefault(r['group'], []).append(i)
            def rank(g, path=''):
                return hashlib.sha256(f"{cfg['seed']}:{c}:{g}:{path}".encode()).digest()
            for g in sorted(patients, key=lambda g: (rank(g), g))[:cfg['memory_per_class']]:
                selected.append(min(patients[g], key=lambda i: (rank(g, train[i]['relative_path']), i)))
        rows = pack['memory_rows']
        assert sorted(r['id'] for r in rows) == sorted(selected)
        assert len(rows) == len({(r['label'], r['group']) for r in rows})
        for row, vector in zip(rows, pack['memory_features']):
            i = row['id']
            assert row['relative_path'] == train[i]['relative_path'] and row['label'] in seen
            if i in previous:
                assert np.array_equal(previous[i], vector.numpy())
            previous[i] = vector.numpy().copy()
        assert seals[task]['parameters_per_arm'] == (task+1) * 394880 + len(seen)*1536*(task+1)
        expected_trainable = 394880 + len(seen)*1536*(task+1)
        assert seals[task]['trainable_parameters']['FROZEN_PAST'] == expected_trainable
        assert seals[task]['trainable_parameters']['ALL_TRAINABLE'] == seals[task]['parameters_per_arm']
        if task == 0:
            for k in pack['models']['FROZEN_PAST']:
                assert torch.equal(pack['models']['FROZEN_PAST'][k], pack['models']['ALL_TRAINABLE'][k])
        else:
            prior = torch.load(root / f'stage_{task-1}.private.pt', map_location='cpu', weights_only=False)
            for k, old_value in prior['models']['FROZEN_PAST'].items():
                if k.startswith('branches.'):
                    assert torch.equal(old_value, pack['models']['FROZEN_PAST'][k])
        mask = np.isin(labels, seen)
        metrics = {}
        for a in cfg['arms']:
            state = {k: v.numpy().astype(np.float64) for k, v in pack['models'][a].items()}
            assert state['head'].shape == (len(seen), cfg['input_dimension'] * (task+1))
            _, score = numpy_forward(query['features'].numpy()[mask], state, cfg['logit_scale'])
            saved = torch.load(root / f'scores_{task}_{a}.private.pt', map_location='cpu', weights_only=True).numpy()
            error = max(error, float(np.max(np.abs(score - saved))))
            mismatches += int((score.argmax(1) != saved.argmax(1)).sum())
            pred = np.array(seen)[score.argmax(1)]
            correct = {str(c): int(((pred == c) & (labels[mask] == c)).sum()) for c in seen}
            metrics[a] = dict(correct=correct, total_correct=sum(correct.values()),
                              BA_percent=100 * sum(correct.values()) / mask.sum())
        stages.append(dict(task=task, seen=seen, metrics=metrics))
    assert error <= cfg['score_absolute_tolerance'] and mismatches == 0
    cost = json.loads((root / 'COSTS.json').read_text())
    assert cost['science_train_encodes'] == 68058 and cost['qualification_encodes'] == 20
    assert cost['validation_encodes'] == 300 and cost['qualification_updates'] == 2
    total = cfg['base_updates'] + 5 * cfg['later_updates']
    assert all(v == total for v in cost['updates'].values())
    assert all(v == total * 64 for v in cost['sampled_image_rows'].values())
    assert all(v == total * 128 for v in cost['training_view_rows'].values())
    assert all(v == 1350 for v in cost['validation_model_rows'].values())
    assert cost['encoder_parameter_updates'] == 0
    assert all(v == 7929 for v in cost['head_initialization_embedding_rows'].values())
    assert all(v == 31780 for v in cost['head_initialization_branch_rows'].values())
    assert all(v == 1536000 for v in cost['training_branch_rows'].values())
    assert cost['gradient_branch_rows'] == dict(ALL_TRAINABLE=1536000, FROZEN_PAST=576000)
    assert all(v == 5250 for v in cost['validation_branch_rows'].values())
    assert cost['current_class_input_mean_rows'] == 68058
    assert all(v == 0 for v in cost['contrastive_similarity_pairs'].values())
    resource = json.loads((root.parent / 'guardian_resources.private.json').read_text())
    assert resource['peak_output_bytes'] < cfg['output_budget_bytes']
    assert resource['RSS_sampled_peak_bytes'] < cfg['RSS_budget_bytes']
    results, decision = summarize(stages, cfg, Counter(r['label'] for r in train))
    for name, value in [('RESULTS.json', results), ('DECISION.json', decision),
                        ('VERIFICATION.json', dict(state='PASS', score_groups=12, max_score_error=error,
                         prediction_mismatches=mismatches, patient_overlap=0, memory_reconstruction=True,
                         future_prefix_checks=True, frozen_old_branch_parameter_equality=True, new_encodes=0, new_optimization=0, wall_seconds=time.time()-start))]:
        (root / name).write_text(json.dumps(value, indent=2) + '\n')
    (root / 'status.private.json').write_text(json.dumps(dict(state='COMPLETE_VERIFIED',
        method_gate_pass=decision['proceed'], delivery_pending=True, updated=time.time())))


if __name__ == '__main__':
    main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
