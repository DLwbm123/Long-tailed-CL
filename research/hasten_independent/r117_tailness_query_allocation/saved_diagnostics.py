"""CPU-only descriptive audits of frozen predictions and the saved query tape."""
import csv
import json
from collections import Counter
from pathlib import Path

import torch


def transition_counts(labels, before, after):
    assert len(labels) == len(before) == len(after)
    rows = []
    for c in range(23):
        ids = labels == c
        old, new = before[ids], after[ids]
        good_old, good_new = old == c, new == c
        rescued = int((~good_old & good_new).sum())
        harmed = int((good_old & ~good_new).sum())
        assert int(good_new.sum() - good_old.sum()) == rescued - harmed
        pairs = Counter(zip(old.tolist(), new.tolist()))
        rows.append({'class': c, 'denominator': int(ids.sum()),
                     'baseline_correct': int(good_old.sum()), 'endpoint_correct': int(good_new.sum()),
                     'rescued': rescued, 'harmed': harmed,
                     'prediction_transitions': [{'before': a, 'after': b, 'count': n}
                                                for (a, b), n in sorted(pairs.items())]})
    assert sum(r['denominator'] for r in rows) == len(labels)
    return rows


def run(cfg):
    torch.set_num_threads(1)
    root, base = Path(cfg['output']), Path(cfg['base_source'])
    load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    sft = load(base / 'SFT_PREDICTIONS.private.pt')
    info = json.loads((base / 'TARS_WEIGHTS.json').read_text())
    sources = [(base, 'TARS_GRPO'),
               (base.parent / 'r115_head_protected_tars', 'PROJECTED_TARS'),
               (base.parent / 'r115_head_protected_tars', 'SHRINK_TARS'),
               (Path(cfg['natural_control_source']), 'KL_PROJECTED_TARS'),
               (Path(cfg['natural_control_source']), 'KL_SHRINK_TARS')]
    complete = (root / 'FINAL_RESULTS.json').exists() and (root / 'CONTROL_EXIT.json').exists()
    if complete:
        assert json.loads((root / 'CONTROL_EXIT.json').read_text())['exit_code'] == 0
        sources += [(root, a) for a in cfg['arms']]
    transitions = {}
    for folder, arm in sources:
        saved = load(folder / (arm + '_PREDICTIONS.private.pt'))
        assert torch.equal(sft['labels'], saved['labels'])
        transitions[arm] = transition_counts(sft['labels'], sft['predictions'], saved['predictions'])
    train = list(csv.DictReader((Path(cfg['manifest']) / 'train.csv').open()))
    tape = load(root / 'QUERY_TAPE.private.pt').flatten().tolist()
    assert len(train) == 6821 and len(tape) == 2048
    allocation = json.loads((root / 'ALLOCATION.json').read_text())
    exposure = []
    for c in range(23):
        ids = [i for i in tape if int(train[i]['original_label']) == c]
        assert len(ids) == allocation['realized_queries_by_class'][c]
        exposure.append({'class': c, 'TRAIN_count': info['counts'][c], 'queries': len(ids),
                         'unique_TRAIN_rows': len(set(ids)),
                         'unique_TRAIN_identity_components': len({train[i]['identity_component'] for i in ids})})
    rewards = {}
    if complete:
        curves = json.loads((root / 'TRAINING_CURVES.json').read_text())
        for arm in cfg['arms']:
            if arm == 'ALLOC_CE_FULL_IS':
                continue
            rr = [r for r in curves if r['arm'] == arm]
            assert len(rr) == 256
            class_rows = []
            for c in range(23):
                successes = sum(r['correct_by_class'][c] for r in rr)
                responses = sum(r['responses_by_class'][c] for r in rr)
                assert responses == 8 * exposure[c]['queries'] and successes == int(successes)
                # Per-class group compositions were not persisted by the frozen worker.
                class_rows.append({'class': c, 'responses': responses, 'successful_responses': int(successes)})
            rewards[arm] = {'classes': class_rows, 'groups': sum(r['groups'] for r in rr),
                            'constant_groups': sum(r['constant_groups'] for r in rr),
                            'mixed_groups': sum(r['groups'] - r['constant_groups'] for r in rr)}
    result = {'scope': 'saved_only_descriptive_no_selection_or_new_model_queries',
              'R117_complete': complete, 'new_model_queries_updates': 0,
              'class_names': info['class_names'], 'transitions_from_SFT': transitions,
              'shared_R117_query_exposure': exposure, 'R117_reward_counts': rewards,
              'limitations': ['Counts refer to validation rows, not independent patients.',
                             'Repeated TRAIN queries are not independent new evidence.',
                             'Frozen curves retain class success totals and global constant groups only; exact per-class all-wrong/all-right/mixed groups cannot be recovered.',
                             'Descriptive VAL transitions do not prove a causal semantic-boundary mechanism or authorize tuning.']}
    (root / 'SAVED_DIAGNOSTICS.json').write_text(json.dumps(result, indent=2) + '\n')
    return {'passed': True, 'endpoints': len(transitions), 'R117_complete': complete,
            'new_model_queries_updates': 0}
