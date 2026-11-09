"""Descriptive aggregate checks on sealed records; no inference or new selection."""
import json
from pathlib import Path

from .core import write


def run(cfg):
    root = Path(cfg['output'])
    assert json.loads((root / 'FINAL_VERIFICATION.json').read_text())['passed']
    rows = json.loads((root / 'MEASUREMENTS.private.json').read_text())
    assert len(rows) == 175
    correct = lambda row, method: row['results'][method]['recognition']['prediction'] == row['class']
    methods = ['GLOBAL_REPEAT', 'CENTER_DETAIL']
    strata = {}
    for global_valid in [False, True]:
        for center_valid in [False, True]:
            selected = [r for r in rows
                        if r['results'][methods[0]]['proposal']['valid'] == global_valid
                        and r['results'][methods[1]]['proposal']['valid'] == center_valid]
            strata[f'global_{global_valid}_center_{center_valid}'] = {
                'components': len(selected),
                'ROOT_correct': sum(correct(r, 'ROOT') for r in selected),
                'GLOBAL_correct': sum(correct(r, methods[0]) for r in selected),
                'CENTER_correct': sum(correct(r, methods[1]) for r in selected),
                'CENTER_rescues_GLOBAL_errors': sum(not correct(r, methods[0]) and correct(r, methods[1]) for r in selected),
                'CENTER_loses_GLOBAL_correct': sum(correct(r, methods[0]) and not correct(r, methods[1]) for r in selected),
            }
    assert sum(s['components'] for s in strata.values()) == 175
    policy = {}
    for method in methods:
        valid_rows = [r for r in rows if r['results'][method]['proposal']['valid']]
        areas = [((r['results'][method]['proposal']['box'][2] - r['results'][method]['proposal']['box'][0])
                  * (r['results'][method]['proposal']['box'][3] - r['results'][method]['proposal']['box'][1])) / 1e6
                 for r in valid_rows]
        lost = sum(correct(r, 'ROOT') and not correct(r, method) for r in rows)
        rescued = sum(not correct(r, 'ROOT') and correct(r, method) for r in rows)
        assert sum(correct(r, method) for r in rows) - sum(correct(r, 'ROOT') for r in rows) == rescued - lost
        policy[method] = {
            'valid': len(valid_rows), 'invalid': len(rows) - len(valid_rows),
            'correct_valid': sum(correct(r, method) for r in valid_rows),
            'ROOT_correct_on_valid': sum(correct(r, 'ROOT') for r in valid_rows),
            'rescues_ROOT_errors': rescued, 'loses_ROOT_correct': lost,
            'mean_valid_normalized_box_area': sum(areas) / len(areas),
            'median_valid_normalized_box_area': sorted(areas)[len(areas) // 2],
            'valid_area_at_most_quarter': sum(a <= .25 for a in areas),
        }
    output = {'scope': 'posthoc_saved_only_descriptive_not_new_gates', 'new_model_queries_updates': 0,
              'components': len(rows), 'ROOT_correct': sum(correct(r, 'ROOT') for r in rows),
              'validity_strata': strata, 'policy_vs_ROOT': policy,
              'limitations': ['Validity strata are post-intervention selections, not randomized subgroups.',
                             'ROOT is cheaper; these counts diagnose retained information, not matched-compute superiority.',
                             'No localization ground truth; box area is descriptive, not localization quality.',
                             'The recognizer was fit on TRAIN; no independent clinical or RL-learning claim.']}
    write(root / 'SAVED_MECHANISM_ANALYSIS.json', output)
    return output
