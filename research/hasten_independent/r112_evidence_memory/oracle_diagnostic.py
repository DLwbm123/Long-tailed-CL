"""Post-hoc oracle decomposition of saved R112 predictions; never a policy."""
import csv
import json
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def qualified(tapes, role, i, tail):
    return not tail or all(
        t[role][i]['full_oldtail_evaluable'] and t[role][i]['qualified']
        for t in tapes
    )


def best(ids, scores, allowed):
    eligible = [i for i in ids if i == 0 or allowed(i)]
    i = max(eligible, key=lambda i: (scores[i], -i))
    return i if scores[i] > 0 else 0


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concentration(values, observed):
    positive = sorted((v for v in values.values() if v > 1e-12), reverse=True)
    gross = sum(positive)
    return {
        'distinct_AUDIT_components_observed': len(observed),
        'net_positive_components': len(positive),
        'net_negative_components': sum(v < -1e-12 for v in values.values()),
        'net_R_increment_sum': sum(values.values()),
        'gross_positive_component_contribution': gross,
        'gross_negative_component_contribution': -sum(v for v in values.values() if v < 0),
        **{f'top{k}_share_of_gross_positive': sum(positive[:k]) / gross if gross else None
           for k in [1, 3, 5]},
    }


def run(base):
    # These checks cover a FIT exclusion and the mandatory ZERO/tie fallback.
    assert best(range(3), [0, .2, .3], lambda i: i == 1) == 1
    assert best(range(3), [0, 0, -.1], lambda i: True) == 0
    bindings = json.loads((base / 'ORACLE_BINDINGS.private.json').read_text())
    saved = list(csv.DictReader((base / 'P1C_RULE_SELECTION.csv').open()))
    actual = {(r['stream'], int(r['seed']), int(r['stage']), int(r['scenario']), r['rule']): r
              for r in saved}
    rows = []
    candidate_scores = []
    contributions = defaultdict(lambda: defaultdict(float))
    observed = defaultdict(set)
    binding_checks = 0
    for name in ['a', 'b']:
        data = json.loads((base / 'final_results.private' / name / 'science/RESULTS.json').read_text())
        raw = {(r['seed'], r['stage'], r['scenario']): r['AUDIT'] for r in bindings[name]}
        for rec in data['roots']:
            key = (rec['seed'], rec['stage'], rec['scenario'])
            tapes, candidates = rec['tapes'], rec['candidates']
            tail = bool(rec['metadata']['real_oldtail_shadow'])
            if tail:
                assert all(t[role][i]['full_oldtail_evaluable']
                           for t in tapes for role in ['FIT', 'AUDIT'] for i in range(len(candidates)))
            audit = [mean([t['AUDIT'][i]['R'] for t in tapes]) for i in range(len(candidates))]
            fit = [mean([t['FIT'][i]['R'] for t in tapes]) for i in range(len(candidates))]
            anchors = [i for i, c in enumerate(candidates) if c['family'] == 'anchor']
            assert 0 in anchors and abs(audit[0]) < 1e-12 and abs(fit[0]) < 1e-12
            safe = lambda i: qualified(tapes, 'AUDIT', i, tail)
            both = lambda i: safe(i) and qualified(tapes, 'FIT', i, tail)
            for i, candidate in enumerate(candidates):
                candidate_scores.append({
                    'stream': rec['stream'], 'seed': rec['seed'], 'stage': rec['stage'],
                    'scenario': rec['scenario'], 'candidate': i, 'family': candidate['family'],
                    'anchor': candidate['anchor'], 'oldtail_gate_applicable': tail,
                    'mean_FIT_R': fit[i], 'mean_AUDIT_R': audit[i],
                    'FIT_admissible': qualified(tapes, 'FIT', i, tail), 'AUDIT_admissible': safe(i),
                })
            choices = {
                'ANCHOR_AUDIT': best(anchors, audit, safe),
                'ALL_AUDIT': best(range(len(candidates)), audit, safe),
                'ANCHOR_FIT_AUDIT': best(anchors, audit, both),
                'ALL_FIT_AUDIT': best(range(len(candidates)), audit, both),
            }
            actual_ids = {}
            for rule, ids in [('E-ANCHOR', anchors), ('E-LOCAL', range(len(candidates)))]:
                selected = best(ids, fit, lambda i: qualified(tapes, 'FIT', i, tail))
                a = actual[(rec['stream'], *key, rule)]
                assert selected == int(a['selected_candidate'])
                assert abs(audit[selected] - float(a['R'])) < 1e-12
                actual_ids[rule] = selected
            vals = {k + '_R': audit[i] for k, i in choices.items()}
            extra = vals['ALL_AUDIT_R'] - vals['ANCHOR_AUDIT_R']
            lost = vals['ALL_AUDIT_R'] - vals['ALL_FIT_AUDIT_R']
            assert extra >= -1e-12 and lost >= -1e-12
            action = len(candidates) > 1 and rec['metadata']['S_root_tokens'] > rec['metadata']['S_capacity']
            row = {'stream': rec['stream'], 'seed': rec['seed'], 'stage': rec['stage'],
                   'scenario': rec['scenario'], 'informative': tail and action,
                   'candidate_count': len(candidates), **choices, **vals,
                   'local_oracle_increment_R': extra, 'FIT_exclusion_loss_R': lost,
                   'local_increment_after_FIT_R': vals['ALL_FIT_AUDIT_R'] - vals['ANCHOR_FIT_AUDIT_R'],
                   'actual_LOCAL_R': audit[actual_ids['E-LOCAL']],
                   'actual_ANCHOR_R': audit[actual_ids['E-ANCHOR']],
                   'actual_LOCAL': actual_ids['E-LOCAL'],
                   'actual_LOCAL_AUDIT_admissible': safe(actual_ids['E-LOCAL']),
                   'ALL_oracle_choice_passes_FIT': qualified(tapes, 'FIT', choices['ALL_AUDIT'], tail)}
            rows.append(row)
            # Safe-oracle R has zero tail penalty, so its increment is additive over probes.
            root_contrib = defaultdict(float)
            for ti, tape in enumerate(tapes):
                assert len(raw[key][ti]) == len(candidates)
                for i in range(len(candidates)):
                    for h, point in enumerate(raw[key][ti][i]):
                        assert point['ids'] == raw[key][ti][0][h]['ids']
                        scored = tape['AUDIT'][i]['endpoints'][h]
                        assert len(point['ids']) == sum(scored['denominator'])
                        assert sum(point['good']) == sum(scored['correct'])
                        binding_checks += 1
                for h in range(3):
                    left = raw[key][ti][choices['ALL_AUDIT']][h]
                    right = raw[key][ti][choices['ANCHOR_AUDIT']][h]
                    weight = 1 / (6 * len(left['ids']))
                    for identity, a, b in zip(left['ids'], left['good'], right['good']):
                        root_contrib[identity] += (int(a) - int(b)) * weight
            assert abs(sum(root_contrib.values()) - extra) < 1e-12
            groups = [(rec['stream'], rec['seed'], 'all'), (rec['stream'], 0, 'all')]
            if row['informative']:
                groups += [(rec['stream'], rec['seed'], 'informative'),
                           (rec['stream'], 0, 'informative')]
            for group in groups:
                observed[group].update(root_contrib)
                for identity, value in root_contrib.items():
                    contributions[group][identity] += value
    assert len(rows) == 64
    blocks = []
    for stream in sorted({r['stream'] for r in rows}):
        for seed in [0, 74003, 74004]:
            for scope in ['all', 'informative']:
                rr = [r for r in rows if r['stream'] == stream and (not seed or r['seed'] == seed)
                      and (scope == 'all' or r['informative'])]
                block = {'stream': stream, 'seed': seed or 'pooled', 'scope': scope, 'roots': len(rr)}
                for metric in ['ANCHOR_AUDIT_R', 'ALL_AUDIT_R', 'ANCHOR_FIT_AUDIT_R',
                               'ALL_FIT_AUDIT_R', 'local_oracle_increment_R', 'FIT_exclusion_loss_R',
                               'local_increment_after_FIT_R', 'actual_LOCAL_R', 'actual_ANCHOR_R']:
                    block['mean_' + metric] = mean([r[metric] for r in rr])
                block['positive_local_increment_roots'] = sum(r['local_oracle_increment_R'] > 1e-12 for r in rr)
                block['positive_local_increment_snapshots'] = len({r['stage'] for r in rr if r['local_oracle_increment_R'] > 1e-12})
                block['nonzero_ALL_oracle_choices'] = sum(r['ALL_AUDIT'] != 0 for r in rr)
                block.update(concentration(contributions[(stream, seed, scope)], observed[(stream, seed, scope)]))
                assert abs(block['net_R_increment_sum'] - sum(r['local_oracle_increment_R'] for r in rr)) < 1e-12
                blocks.append(block)
    table(base / 'ORACLE_ROOTS.csv', rows)
    table(base / 'ORACLE_BLOCKS.csv', blocks)
    table(base / 'ORACLE_CANDIDATE_SCORES.csv', candidate_scores)
    result = {'post_hoc_oracle_only': True, 'original_R112_decision_unchanged': True,
              'new_training_or_query_forward': 0, 'VAL_TEST': 0,
              'same_probes_repeated_not_independent_patients': True,
              'candidate_H3_AUDIT_bindings_checked': binding_checks, 'blocks': blocks}
    (base / 'ORACLE_RESULTS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps([r for r in blocks if r['seed'] == 'pooled'], indent=2))


if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
