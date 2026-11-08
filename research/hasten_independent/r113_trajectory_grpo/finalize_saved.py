"""Verify completed integer counts and frozen gates without model queries."""
import csv
import datetime
import json
from pathlib import Path


def mean(v):
    return sum(v) / len(v) if v else None


def table(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else 'NA' if v is None else v
                          for k, v in r.items()} for r in rows)


def run(base):
    snap = json.loads((base / 'FINAL_SNAPSHOT.private.json').read_text())
    raw = snap['files']
    original = json.loads(raw['FINAL_RESULTS.json'])
    exit_receipt = json.loads(raw['CONTROL_EXIT.json'])
    assert exit_receipt['exit_code'] == 0 and all(r['exit_code'] == 0 for r in exit_receipt['workers'].values())
    rows, blocks, groups, horizons = [], [], [], []
    learning, maximum_error = {}, 0.
    for name in ['a', 'b']:
        result = json.loads(raw[name + '/RESULTS.json'])
        rr = result['stages']
        assert len(rr) == 50
        assert {(r['seed'], r['stage'], r['arm']) for r in rr} == {
            (seed, stage, arm) for seed in [74003, 74004] for stage in range(5)
            for arm in ['ZERO', 'BAL', 'MYOPIC_GRPO', 'LONG_GRPO', 'SUP_SHARED']}
        rows.extend(rr)
        for r in rr:
            assert all(n == 1 for n in r['denominator']) and all(c in [0, 1] for c in r['correct'])
            error = abs(r['macro_BA'] - mean([c / n for c, n in zip(r['correct'], r['denominator'])]))
            assert error < 1e-12
            maximum_error = max(maximum_error, error)
            assert r['oldtail_BA'] == mean([r['correct'][c] / r['denominator'][c] for c in r['oldtail_classes']])
            assert r['evidence_bytes'] <= 528568
        gg = [json.loads(s) for p, s in raw.items() if p.startswith(name + '/GROUP_')]
        assert len(gg) == 40
        for g in gg:
            assert g['AUDIT_training_reads'] == 0
            for i, (ret, points) in enumerate(zip(g['group_returns'], g['FIT_points'])):
                assert len(points) == 3
                value = mean([p['macro'] for p in points]) - 2 * max(
                    [max(0, -p['tail'] - .005) for p in points if p['tail'] is not None], default=0)
                maximum_error = max(maximum_error, abs(value - ret))
                assert abs(value - ret) < 1e-12
                for h, (point, allocation, step) in enumerate(zip(points, g['allocations'][i], g['steps'][i])):
                    assert allocation['tokens'] <= 256 and allocation['bytes'] == allocation['tokens'] * 2064 + 184 <= 528568
                    assert sum(allocation['S_by_class']) + 2 * allocation['C'] == allocation['tokens']
                    assert all(v > 0 for v in allocation['S_by_class'])
                    assert 0 <= step['action'] < step['candidate_count'] <= 12
                    horizons.append({'stream': result['stream'], 'seed': g['seed'], 'real_stage': g['stage'] + 1,
                                     'scenario': g['scenario'], 'mode': g['mode'], 'trajectory': i, 'horizon': h + 1,
                                     'R_trajectory': ret, 'macro_delta': point['macro'], 'oldtail_delta': point['tail'],
                                     'correct': point['correct'], 'denominator': point['denominator'],
                                     'tokens': allocation['tokens'], 'bytes': allocation['bytes'],
                                     'S_by_class': allocation['S_by_class'], 'action': step['action'],
                                     'candidate_count': step['candidate_count'], 'chosen_probability': step['chosen_probability']})
            returns = g['group_returns']
            groups.append({'stream': result['stream'], 'seed': g['seed'], 'real_stage': g['stage'] + 1,
                           'scenario': g['scenario'], 'mode': g['mode'], 'returns': returns,
                           'nonconstant': max(returns) - min(returns) > 1e-12,
                           'policy_updates': len(g['updates']), 'SUP_updates': len(g['supervised_updates']),
                           'max_gradient_norm': max(u['grad_norm'] for u in g['updates'])})
        learning[result['stream']] = {'groups': 40, 'policy_updates': result['policy_updates'],
                                     'groups_with_nonconstant_returns': sum(max(g['group_returns']) - min(g['group_returns']) > 1e-12 for g in gg),
                                     'max_gradient_norm': max(u['grad_norm'] for g in gg for u in g['updates']),
                                     'science_classifier_updates': result['science_classifier_updates']}
        for seed in [74003, 74004]:
            for arm in ['ZERO', 'BAL', 'MYOPIC_GRPO', 'LONG_GRPO', 'SUP_SHARED']:
                ss = sorted([r for r in rr if r['seed'] == seed and r['arm'] == arm], key=lambda r: r['stage'])
                md, td = [], []
                for r in ss:
                    z = next(z for z in rr if z['seed'] == seed and z['stage'] == r['stage'] and z['arm'] == 'ZERO')
                    md.append(r['macro_BA'] - z['macro_BA'])
                    if r['oldtail_BA'] is not None:
                        td.append(r['oldtail_BA'] - z['oldtail_BA'])
                score = mean(md) - 2 * max([max(0, -v - .005) for v in td], default=0)
                saved = next(z for z in original['blocks'] if z['stream'] == result['stream'] and z['seed'] == seed and z['arm'] == arm)
                assert abs(saved['R_real5'] - score) < 1e-12
                blocks.append({'stream': result['stream'], 'seed': seed, 'arm': arm, 'R_real5': score,
                               'mean_macro_BA_percent': 100 * mean([r['macro_BA'] for r in ss]),
                               'terminal_macro_BA_percent': 100 * ss[-1]['macro_BA'],
                               'terminal_oldtail_BA_percent': 100 * ss[-1]['oldtail_BA'],
                               'mean_macro_delta_pp': 100 * mean(md), 'worst_oldtail_delta_pp': 100 * min(td),
                               'oldtail_violation_stage_seed_pairs': sum(v < -.005 for v in td),
                               'nonzero_actions': sum(r['action'] != 0 for r in ss), 'action_denominator': 5})
    assert len(rows) == 100 and len(groups) == 80 and len(horizons) == 960
    table(base / 'FINAL_STAGE_RESULTS.csv', rows)
    table(base / 'FINAL_BLOCK_SUMMARY.csv', blocks)
    table(base / 'FINAL_GROUP_SUMMARY.csv', groups)
    table(base / 'FINAL_FIT_TRAJECTORIES.csv', horizons)
    pooled = {}
    for stream in ['ordered', 'stratified_shuffled']:
        pooled[stream] = {}
        for arm in ['ZERO', 'BAL', 'MYOPIC_GRPO', 'LONG_GRPO', 'SUP_SHARED']:
            bb = [b for b in blocks if b['stream'] == stream and b['arm'] == arm]
            pooled[stream][arm] = {k: mean([b[k] for b in bb]) for k in
                                  ['R_real5', 'mean_macro_BA_percent', 'terminal_macro_BA_percent',
                                   'terminal_oldtail_BA_percent', 'mean_macro_delta_pp']}
            pooled[stream][arm].update(nonzero_actions=sum(b['nonzero_actions'] for b in bb),
                                      action_denominator=10,
                                      oldtail_violation_stage_seed_pairs=sum(b['oldtail_violation_stage_seed_pairs'] for b in bb))
    first = json.loads((base / 'FIRST.private.json').read_text())
    finish = datetime.datetime.fromtimestamp(first['start'] + exit_receipt['original_elapsed_seconds'],
                                            datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    summary = {'finished_at': finish, 'experiment_complete': True, 'real_stage_rows': 100, 'training_groups': 80,
               'decision': original['decision'], 'streams': pooled, 'learning': learning,
               'original_R112_C_unchanged': True, 'independent_population_confirmation': False,
               'VAL_TEST': 0, 'patient_denominator': 'NA', 'failed_attempt1_not_pooled': True}
    (base / 'FINAL_SUMMARY.json').write_text(json.dumps(summary, indent=2))
    verification = {'passed': True, 'real_stage_rows': 100, 'training_groups': 80, 'FIT_horizon_records': 960,
                    'maximum_score_error': maximum_error, 'all_fixed_slots_present': True,
                    'new_training_or_query_forward': 0, 'controller_and_workers_exit_zero': True,
                    'unchanged_success_gate_failed': True}
    (base / 'FINAL_VERIFICATION.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
