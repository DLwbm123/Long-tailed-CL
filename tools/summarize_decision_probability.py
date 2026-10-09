"""Validate and summarize frozen terminal decisions from anonymous aggregates."""
import json
import math
from pathlib import Path
import statistics
import sys
import time

from summarize_feedback_audit import correlation


KEYS = ['edge_%02d' % i for i in range(17)]+['norm_%02d' % i for i in range(1, 17)]
mean = statistics.mean


def select(record, indices):
    seen, current = record['seen'], record['current']
    groups = dict(all=seen, old=[c for c in seen if c not in current], current=current)
    if record['tail']:
        groups['tail'] = record['tail']
    audit = {}
    for i in indices:
        values = {}
        for group, classes in groups.items():
            deltas = [mean([(record['meta'][KEYS[i]]['per_class'][str(c)]['recall']-
                            record['meta'][KEYS[0]]['per_class'][str(c)]['recall']) if c in current else
                           b[i][seen.index(c)]-b[0][seen.index(c)] for c in classes])
                      for b in record['gaussian_batches']]
            values[group] = dict(mean_gain=mean(deltas), integration_se=statistics.stdev(deltas)/math.sqrt(len(deltas)))
        protected = all(v['mean_gain'] >= -1e-12 for g, v in values.items() if g != 'all')
        resolution = max(1e-8, 2*values['all']['integration_se'])
        audit[KEYS[i]] = dict(groups=values, protection_pass=protected, numerical_resolution=resolution,
                             eligible=protected and values['all']['mean_gain'] > resolution)
    candidates = [k for k, v in audit.items() if v['eligible']]
    return min(candidates, key=lambda k: (-audit[k]['groups']['all']['mean_gain'], KEYS.index(k))) if candidates else KEYS[0], audit


def development(record, partition):
    raw = record['development'][partition]; zero = raw['edge_00']['metrics']['per_class']; rows = {}
    groups = dict(all=record['seen'], old=[c for c in record['seen'] if c not in record['current']],
                  current=record['current'], tail=record['tail'])
    for key in KEYS:
        per = raw[key]['metrics']['per_class']; pairs = raw[key]['paired_to_zero']
        delta = {str(c): per[str(c)]['recall']-zero[str(c)]['recall'] for c in record['seen']}
        for c, v in pairs.items():
            assert abs(delta[c]-(v['helped']-v['hurt'])/v['n']) < 1e-12
        gains = {g: mean(delta[str(c)] for c in classes) if classes else None for g, classes in groups.items()}
        assert abs(gains['all']-(raw[key]['metrics']['views']['all']['BA']-raw['edge_00']['metrics']['views']['all']['BA'])) < 1e-12
        rows[key] = dict(BA_gain=gains, BA=raw[key]['metrics']['views']['all']['BA'], per_class_recall_gain=delta,
            helped=sum(v['helped'] for v in pairs.values()), hurt=sum(v['hurt'] for v in pairs.values()),
            prediction_changed=sum(v['prediction_changed'] for v in pairs.values()),
            group_protection=all(v >= -1e-12 for g, v in gains.items() if g != 'all' and v is not None),
            per_class_protection=min(delta.values()) >= -1e-12,
            per_sample_protection=all(v['hurt'] == 0 for v in pairs.values()))
    return rows


def calibration(record):
    current = record['current']; seen = record['seen']; per_class = {}
    for c in current:
        j = seen.index(c); truth = [record['meta'][k]['per_class'][str(c)]['recall'] for k in KEYS]
        predicted = [r[j] for r in record['gaussian_mean']]
        dg = [v-predicted[0] for v in predicted]; dt = [v-truth[0] for v in truth]
        sign = lambda v: (v > 1e-12)-(v < -1e-12)
        per_class[str(c)] = dict(support=record['current_support'][str(c)], zero_gaussian=predicted[0], zero_meta=truth[0],
            edge_accuracy_MAE=mean(abs(predicted[i]-truth[i]) for i in range(17)),
            edge_delta_MAE=mean(abs(dg[i]-dt[i]) for i in range(1, 17)),
            edge_three_way_sign_agreement=sum(sign(dg[i]) == sign(dt[i]) for i in range(1, 17)),
            edge_nonzero_meta_deltas=sum(sign(dt[i]) != 0 for i in range(1, 17)),
            edge_delta_correlation=correlation(dg[:17], dt[:17]))
    return dict(current_classes=per_class, edge_accuracy_MAE=mean(v['edge_accuracy_MAE'] for v in per_class.values()),
                edge_delta_MAE=mean(v['edge_delta_MAE'] for v in per_class.values()),
                interpretation='Training-side descriptive calibration; the same current meta also selects heads. No independent generalization or old-class transport test.')


def summarize(data, lock, repair):
    assert len(data['results']) == len(data['statuses']) == len(data['suites']) == 3
    output = {}; comparisons = {}; checks = 0; all_states = []
    for job in lock['jobs']:
        name = job['name']; status = data['statuses'][name]; expected = [s['name'] for s in job['states']]
        assert status['status'] == data['suites'][name]['status'] == 'COMPLETE'
        assert status['completed_states'] == status['evaluated_states'] == expected
        assert status['adapter_updates'] == status['policy_updates'] == 0 and not status['test_accessed']
        assert status['maximum_solve_residual'] <= lock['max_solve_residual']
        barrier = data['barriers'][name]
        assert barrier['states'] == expected and not barrier['development_features_accessed']
        for key, record in data['results'][name].items():
            all_states.append(key); assert key in expected
            assert record['development_after_all_selections'] and record['current_meta_excluded_from_fit']
            assert record['zero_all_head_reconstruction_error'] <= lock['reconstruction_atol']
            assert len(record['gaussian_batches']) == 8
            assert all(len(b) == 33 and all(len(p) == len(record['seen']) for p in b) for b in record['gaussian_batches'])
            assert all(0 <= x <= 1 for b in record['gaussian_batches'] for p in b for x in p)
            for i in range(33):
                for c in range(len(record['seen'])):
                    assert abs(record['gaussian_mean'][i][c]-mean(b[i][c] for b in record['gaussian_batches'])) < 1e-12
            for label, indices, audit_key in [('decision', range(17), 'decision_audit'),
                    ('norm_decision', [0]+list(range(17, 33)), 'norm_decision_audit')]:
                chosen, audit = select(record, indices); assert chosen == record['selected'][label]
                for k, value in audit.items():
                    saved = record[audit_key][k]
                    assert value['eligible'] == saved['eligible'] and value['protection_pass'] == saved['protection_pass']
                    for group, terms in value['groups'].items():
                        assert all(abs(v-saved['groups'][group][term]) < 1e-12 for term, v in terms.items())
                    checks += 1
            rewards = record['original_rewards']; best = min(rewards, key=lambda k: (-rewards[k], KEYS.index(k)))
            assert record['selected']['original_reward'] == (best if rewards[best] > 1e-8 else KEYS[0])
            summary = dict(dataset=record['dataset'], task=record['task'], cohort=record['cohort'], selected=record['selected'],
                           calibration=calibration(record), partitions={})
            comparisons[key] = {}
            for partition in ('fit', 'all'):
                assert set(record['development'][partition]) == set(KEYS)
                rows = development(record, partition); comparisons[key][partition] = rows
                oracles = {}
                for family, keys in [('edge', KEYS[:17]), ('norm', [KEYS[0]]+KEYS[17:])]:
                    oracles[family] = {}
                    for tier in ('unconstrained', 'group', 'per_class', 'per_sample'):
                        eligible = [k for k in keys if tier == 'unconstrained' or rows[k][tier+'_protection']]
                        best = min(eligible, key=lambda k: (-rows[k]['BA_gain']['all'], KEYS.index(k)))
                        oracles[family][tier] = dict(key=best, **rows[best])
                summary['partitions'][partition] = dict(selected={name: dict(key=k, **rows[k]) for name, k in record['selected'].items()},
                    oracles=oracles, positive_edge_candidates=[k for k in KEYS[:17] if rows[k]['BA_gain']['all'] > 1e-12],
                    correlations=dict(hybrid_edge_vs_BA=correlation([record['decision_audit'][k]['groups']['all']['mean_gain'] for k in KEYS[:17]],
                                                                  [rows[k]['BA_gain']['all'] for k in KEYS[:17]]),
                                      original_reward_vs_BA=correlation([rewards[k] for k in KEYS[:17]], [rows[k]['BA_gain']['all'] for k in KEYS[:17]])))
            output[key] = summary
    assert len(set(all_states)) == lock['states'] == 5
    assert all(data['cached_selection_exact_match'].values())
    costs = {k: v['elapsed_seconds'] for k, v in data['suites'].items()}
    failed_costs = {}
    for attempt, failure in data['failed_attempts'].items():
        assert all(v['status'] == 'INCOMPLETE' for v in failure['suites'].values())
        assert all(not v['evaluated_states'] and not v['test_accessed'] for v in failure['statuses'].values())
        failed_costs[attempt] = {k: v['elapsed_seconds'] for k, v in failure['suites'].items()}
    assert abs(sum(failed_costs['initial'].values())-repair['failed_total_suite_seconds']) < 1e-9
    failed = sum(sum(v.values()) for v in failed_costs.values()); total = sum(costs.values())+failed
    return dict(summary=output, comparisons=comparisons,
        audit=dict(status='PASS', states=len(all_states), head_readouts=5*33*2, selector_candidate_checks=checks,
                   maximum_solve_residual=max(s['maximum_solve_residual'] for s in data['statuses'].values()),
                   zero_head_reconstruction_checks=5, adapter_updates=0, policy_updates=0, test_accessed=False,
                   image_rows=sum(s['image_rows'] for s in data['statuses'].values()),
                   cached_selection_exact_match=data['cached_selection_exact_match'],
                   selection_barrier='All three SELECT_COMPLETE receipts present; all five choices reproduced from pre-DEV aggregates.'),
        budget=dict(prior_closed_gpu_seconds=lock['prior_closed_gpu_seconds'], failed_attempt_suite_seconds=failed_costs,
                    repair_suite_seconds=costs, round_gpu_seconds=total, cumulative_gpu_seconds=lock['prior_closed_gpu_seconds']+total,
                    worker_cpu_seconds={k: v['cpu_process_seconds'] for k, v in data['statuses'].items()},
                    cpu_import_preflight_seconds=repair['cpu_import_preflight']['elapsed_seconds'], original_deadline=lock['original_deadline'],
                    note='Suite residence counted once, including barriers; worker CPU separate and not added. Failed attempt retained.'))


def self_check():
    record = dict(seen=[0, 1], current=[1], tail=[0], gaussian_batches=[[[.5, .5], [.6, .5]]]*8,
                  meta={k: {'per_class': {'1': {'recall': .5}}} for k in KEYS[:2]})
    assert select(record, [0, 1])[0] == 'edge_01'
    record['meta']['edge_01']['per_class']['1']['recall'] = .4
    assert select(record, [0, 1])[0] == 'edge_00'
    record['meta']['edge_01']['per_class']['1']['recall'] = .5
    for i in range(8):
        record['gaussian_batches'][i] = [[.5, .5], [.4 if i % 2 else .65, .5]]
    assert select(record, [0, 1])[0] == 'edge_00'


if __name__ == '__main__':
    config = json.load(sys.stdin); started = time.process_time(); self_check()
    if config.get('self_check'):
        print('PASS: hybrid choice, protected fallback, numerical uncertainty fallback')
    else:
        data = json.loads(Path(config['input']).read_text()); lock = json.loads(Path(config['lock']).read_text())
        repair = json.loads(Path(config['repair']).read_text()); result = summarize(data, lock, repair)
        out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
        result['budget']['readout_cpu_seconds'] = time.process_time()-started
        for name, value in [('RESULTS', data['results']), ('SUMMARY', result['summary']), ('COMPARISONS', result['comparisons']),
                            ('AUDIT', result['audit']), ('BUDGET_LEDGER', result['budget'])]:
            text = json.dumps(value, indent=2, allow_nan=False)+'\n'
            assert not any(s in text for s in ('/remote-home/', '/Users/', 'identity_component'))
            (out/(name+'.json')).write_text(text)
        print(json.dumps(result['audit']))
