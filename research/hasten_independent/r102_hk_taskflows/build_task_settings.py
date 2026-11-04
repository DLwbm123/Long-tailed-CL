#!/usr/bin/env python3
"""Build HK LT-CIL task specifications; never train or access VAL/TEST.

No flags: write unbound specifications using legacy arrival indices C0...C22.
Binding: supply BOTH --legacy-config and --train-manifest to resolve original
labels and validate all 6,821 TRAIN counts. The output is a task specification,
not a complete training-launch configuration.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

COUNTS = (626, 3, 6, 85, 288, 403, 736, 488, 252, 167, 595, 261,
          6, 34, 83, 25, 23, 635, 127, 662, 18, 654, 644)
SIZES = (5, 5, 5, 4, 4)
ORDER_SEEDS = (0, 1, 2)
SOURCE = ('https://github.com/DLwbm123/Long-tailed-CL/blob/'
          '0a5b5aea078d582be406442bcdfaa1026920bf20/'
          'research/hasten_independent/tars_rlap_review_20261102/REPORT_ZH.md')
RANKED = sorted(range(23), key=lambda c: (-COUNTS[c], c))


def ordered_tasks() -> list[list[int]]:
    result, offset = [], 0
    for size in SIZES:
        result.append(RANKED[offset:offset + size])
        offset += size
    return result


def stratified_tasks(seed: int) -> list[list[int]]:
    # One class per task from ranks 1-5, 6-10, 11-15, and 19-23.
    # Ranks 16-18 add the fifth class to T1-T3 (same sizes as Ordered).
    rng = random.Random(seed)
    result: list[list[int]] = [[] for _ in SIZES]
    for band in (RANKED[0:5], RANKED[5:10], RANKED[10:15], RANKED[18:23]):
        shuffled = list(band)
        rng.shuffle(shuffled)
        for task, class_id in zip(result, shuffled):
            task.append(class_id)
    extras = list(RANKED[15:18])
    rng.shuffle(extras)
    for task, class_id in zip(result[:3], extras):
        task.append(class_id)
    # Within a simultaneous task, display and index in descending frequency.
    for task in result:
        task.sort(key=lambda c: (-COUNTS[c], c))
    return result


def check_tasks(tasks: list[list[int]], mode: str) -> None:
    flat = [c for task in tasks for c in task]
    if tuple(map(len, tasks)) != SIZES or sorted(flat) != list(range(23)):
        raise ValueError('Tasks must cover each of the 23 classes exactly once.')
    if sum(COUNTS[c] for c in flat) != 6821:
        raise ValueError('Unexpected training count total.')
    if mode == 'ordered' and flat != RANKED:
        raise ValueError('Ordered stream is not globally frequency descending.')
    if mode == 'stratified_shuffled':
        for task in tasks:
            if max(COUNTS[c] for c in task) < 635 or min(COUNTS[c] for c in task) > 23:
                raise ValueError('Frequency-band construction invariant failed.')


def make_spec(name: str, tasks: list[list[int]], seed: int | None,
              original_order: list[int] | None) -> dict[str, Any]:
    mode = 'ordered' if seed is None else 'stratified_shuffled'
    check_tasks(tasks, mode)
    flat = [c for task in tasks for c in task]
    records, start = [], 0
    for task_number, classes in enumerate(tasks, 1):
        n = [COUNTS[c] for c in classes]
        record = {
            'task_number_1based': task_number,
            'stage_index_0based': task_number - 1,
            'legacy_arrival_indices': classes,
            'train_counts': n,
            'train_total': sum(n),
            'within_task_imbalance_ratio_max_over_min': max(n) / min(n),
            'new_classifier_indices': list(range(start, start + len(classes))),
            'cumulative_seen_classes': start + len(classes),
        }
        if original_order is not None:
            record['original_labels'] = [original_order[c] for c in classes]
        records.append(record)
        start += len(classes)
    spec = {
        'name': name, 'setting': mode, 'status': 'PROPOSED_NOT_TRAINED',
        'is_complete_training_config': False,
        'binding_status': 'BOUND_AND_TRAIN_COUNTS_CHECKED' if original_order is not None else 'UNBOUND_LEGACY_INDICES_ONLY',
        'order_seed': seed, 'task_sizes': list(SIZES), 'tasks': records,
        'legacy_order_new_arrival': flat,
        'stages_new_classifier_indices': [r['new_classifier_indices'] for r in records],
        'train_counts_new_arrival': [COUNTS[c] for c in flat],
        'new_classifier_index_to_legacy_index': flat,
        'legacy_index_to_new_classifier_index': {str(c): i for i, c in enumerate(flat)},
        'requires_task_id_at_inference': False,
        'preserve_original_image_splits': True,
        'preserve_all_23_classes_and_6821_train_images': True,
        'extra_large_base_task': False,
        'reuse_legacy_hk_supervised_checkpoints': False,
        'all_history_teacher_is_not_bounded_memory_cl': True,
        'global_tail_definition_for_reporting': 'original TRAIN count < 250; fixed per class',
        'tail_metrics_when_no_tail_class_has_arrived': None,
        'memory_budget': None,
        'memory_budget_note': 'Freeze one identical budget before training; this task spec does not choose it.',
    }
    if original_order is not None:
        order = [original_order[c] for c in flat]
        spec['order_original_labels'] = order
        spec['original_label_to_new_classifier_index'] = {str(label): i for i, label in enumerate(order)}
    return spec


def resolve_labels(config_path: Path, manifest_path: Path) -> tuple[list[int], dict[str, Any]]:
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or 'order' not in cfg:
        raise ValueError('Use the original HK dataset configuration containing order; wrappers must be resolved first.')
    order = cfg['order']
    if not isinstance(order, list) or len(order) != 23 or any(type(v) is not int for v in order) or len(set(order)) != 23:
        raise ValueError('Expected 23 distinct integer original labels in legacy order.')
    for key in ('train_counts', 'train_counts_ordered'):
        if key in cfg and list(cfg[key]) != list(COUNTS):
            raise ValueError(f'{key} is not the documented legacy HK count sequence.')
    observed: Counter[int] = Counter()
    with manifest_path.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {'split', 'original_label'} <= set(reader.fieldnames):
            raise ValueError('TRAIN CSV must contain split and original_label columns.')
        for row in reader:
            if row['split'] != 'train':
                raise ValueError('Only TRAIN rows are accepted; do not pass VAL/TEST.')
            observed[int(row['original_label'])] += 1
    if set(observed) != set(order) or [observed[c] for c in order] != list(COUNTS):
        raise ValueError('Manifest counts do not match the legacy class-ID mapping; no output was written.')
    evidence = {
        'train_manifest_count_binding': True,
        'legacy_config_label_binding': True,
        'checked_rows': sum(observed.values()),
        'identity_or_patient_independence_checked': False,
        'val_test_read': False,
    }
    return order, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-config', type=Path)
    parser.add_argument('--train-manifest', type=Path)
    args = parser.parse_args()
    if (args.legacy_config is None) != (args.train_manifest is None):
        parser.error('Supply both --legacy-config and --train-manifest, or neither.')
    original_order, evidence = None, None
    if args.legacy_config is not None:
        original_order, evidence = resolve_labels(args.legacy_config, args.train_manifest)
    streams = [make_spec('HK_Ordered_NoLargeBase_5Tasks', ordered_tasks(), None, original_order)]
    streams.extend(make_spec(f'HK_StratifiedShuffled_NoLargeBase_order{s}', stratified_tasks(s), s, original_order)
                   for s in ORDER_SEEDS)
    payload = {
        'schema_version': '1.0', 'source_train_counts': SOURCE,
        'legacy_index_definition': 'C_i is the arrival index in the documented OLD HK 13+2+2+2+2+2 stream, NOT necessarily original_label.',
        'legacy_train_counts': list(COUNTS),
        'setting_B_note': 'Frequency-stratified shuffled, not unconstrained random Shuffled LT-CIL. No performance-based seed selection.',
        'construction_bands_rank_1based': [[1,5],[6,10],[11,15],[19,23]],
        'extra_band_rank_1based': [16,18],
        'extra_class_tasks_1based': [1,2,3],
        'generation_python_version': sys.version.split()[0],
        'binding_evidence': evidence,
        'streams': streams,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'hk_task_settings.json'
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for spec in streams:
        print(spec['name'])
        for record in spec['tasks']:
            print(f"  T{record['task_number_1based']} ids={record['legacy_arrival_indices']} counts={record['train_counts']} total={record['train_total']} IR={record['within_task_imbalance_ratio_max_over_min']:.6f}")
    print(f'Wrote {destination}; no training, VAL/TEST access, or repository modification.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f'Validation failed: {error}') from error
