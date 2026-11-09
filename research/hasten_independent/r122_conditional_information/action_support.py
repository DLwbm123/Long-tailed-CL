"""Negative feasibility bound for selectors over fixed saved terminal readouts."""
import json
from pathlib import Path

from .core import write


def run(cfg):
    root = Path(cfg['output'])
    assert json.loads((root / 'FINAL_VERIFICATION.json').read_text())['passed']
    rows = json.loads((root / 'MEASUREMENTS.private.json').read_text())
    definitions = json.loads((root / 'CLASS_DEFINITIONS.json').read_text())
    groups = {'macro': list(range(23)),
              'head': [c for c, n in enumerate(definitions['counts']) if n >= 250],
              'tail': [c for c, n in enumerate(definitions['counts']) if n < 250],
              'tiny': [c for c, n in enumerate(definitions['counts']) if n < 25]}
    den = [sum(r['class'] == c for r in rows) for c in range(23)]
    assert len(rows) == 175 and all(den)
    is_correct = lambda row, method: row['results'][method]['recognition']['prediction'] == row['class']
    root_good = [sum(r['class'] == c and is_correct(r, 'ROOT') for r in rows) for c in range(23)]
    output = {'scope': 'fixed_saved_terminal_library_only_not_new_joint_readout',
              'new_model_queries_updates': 0, 'components': 175,
              'oracle_is_not_deployable': True, 'ROOT_correct': sum(root_good), 'libraries': {}}
    for name, methods in {
        'STOP_or_GLOBAL': ['ROOT', 'GLOBAL_REPEAT'],
        'STOP_or_DETAIL': ['ROOT', 'CENTER_DETAIL'],
        'STOP_or_either_proposed_crop': ['ROOT', 'GLOBAL_REPEAT', 'CENTER_DETAIL'],
        'STOP_or_any_saved_view': ['ROOT', 'CENTER', 'GLOBAL_REPEAT', 'CENTER_DETAIL'],
    }.items():
        union = [sum(r['class'] == c and any(is_correct(r, m) for m in methods) for r in rows) for c in range(23)]
        assert all(a >= b for a, b in zip(union, root_good))
        increments = {group: 100 * sum((union[c] - root_good[c]) / den[c] for c in ids) / len(ids)
                      for group, ids in groups.items()}
        output['libraries'][name] = {'methods': methods, 'oracle_correct': sum(union),
            'rescues_over_ROOT': sum(union) - sum(root_good), 'increment_upper_bound_pp': increments,
            'class_rescues': [{'class': c, 'name': definitions['class_names'][str(c)],
                               'rescues': union[c] - root_good[c], 'denominator': den[c]}
                              for c in range(23) if union[c] > root_good[c]],
            'tiny_correct_upper_bound': sum(union[c] for c in groups['tiny']),
            'tiny_components': sum(den[c] for c in groups['tiny'])}
    output['limitations'] = ['Uses labels only to exclude an incapable fixed library, never as a deployable selector.',
        'Does not bound new regions, new stochastic generations, changed weights, or joint global-plus-crop observations.',
        'No new experiment, optimizer, patient-validation result, or change to the frozen ten gates.']
    write(root / 'ACTION_SUPPORT_AUDIT.json', output)
    return output
