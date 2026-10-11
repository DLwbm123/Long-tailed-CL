"""Train-only RFMiD task allocation; image identities stay in the private plan."""
import csv
import math
import random

from prepare_rfmid_ml import BENCHMARK_LABELS


def read_mapped(path, expected):
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['ID', *BENCHMARK_LABELS]:
            raise ValueError('RFMiD mapped label schema differs')
        rows = list(reader)
    ids = [r['ID'] for r in rows]
    if len(rows) != expected or len(set(ids)) != expected or any(
            not s.isascii() or not s.isdigit() or str(int(s)) != s or int(s) < 1 for s in ids):
        raise ValueError('Wrong RFMiD row count or noncanonical/duplicate ID')
    for row in rows:
        if any(row[k] not in ('0', '1') for k in BENCHMARK_LABELS):
            raise ValueError('RFMiD targets must all be known binary values')
        if int(row['Normal']) != int(not any(int(row[k]) for k in BENCHMARK_LABELS[1:])):
            raise ValueError('Normal contradicts disease OR')
    return sorted(rows, key=lambda r: int(r['ID']))


def build_plan(rows):
    labels = list(BENCHMARK_LABELS)
    if len({r['ID'] for r in rows}) != len(rows) or not rows:
        raise ValueError('Empty or duplicated training identities')
    counts = {k: sum(int(r[k]) for r in rows) for k in labels}
    if any(not 0 < n < len(rows) for n in counts.values()):
        raise ValueError('Every frozen RFMiD label needs train positive and negative support')
    order = labels.copy()
    random.Random(74004).shuffle(order)
    order.sort(key=lambda k: counts[k])
    introduced = [order[:9], order[9:14], order[14:19], order[19:24], order[24:]]
    label_task = {k: t for t, new in enumerate(introduced) for k in new}
    assigned = [[] for _ in introduced]
    for row in rows:
        task = min(label_task[k] for k in labels if int(row[k]))
        assigned[task].append(row)
    tasks, audit_tasks, seen = [], [], []
    cumulative = {k: [0, 0] for k in labels}
    missed_positive = {k: 0 for k in labels}
    for t, (new, current) in enumerate(zip(introduced, assigned), 1):
        if not current:
            raise ValueError('Empty RFMiD task; do not choose a replacement seed')
        seen += new
        positive = {k: sum(int(r[k]) for r in current) for k in seen}
        negative = {k: len(current) - positive[k] for k in seen}
        if any(positive[k] == 0 or negative[k] == 0 for k in new):
            raise ValueError('Introduced RFMiD label lacks a sign in its current task')
        for k in seen:
            cumulative[k][0] += negative[k]
            cumulative[k][1] += positive[k]
        for k in labels:
            if k not in seen:
                missed_positive[k] += sum(int(r[k]) for r in current)
        steps = 2 * math.ceil(len(current) / 64)
        tasks.append(dict(task=t, introduced_labels=new, visible_labels=seen.copy(),
                          train_ids=[r['ID'] for r in current], expected_steps=steps))
        audit_tasks.append(dict(task=t, introduced_labels=new, visible_labels=seen.copy(),
                               train_images=len(current), fit_images=len(current), meta_images=0,
                               expected_steps=steps, current_positive=positive, current_negative=negative,
                               cumulative_observed_counts={k: cumulative[k].copy() for k in seen}))
    if sum(len(t['train_ids']) for t in tasks) != len(rows):
        raise AssertionError('Training images lost or replayed')
    descending = order[::-1]
    groups = dict(head=descending[:10], middle=descending[10:20], tail=descending[20:])
    return (dict(labels=labels, tasks=tasks, metric_groups=groups,
                 train_seed=74002, order_tie_seed=74004, internal_meta=False),
            dict(status='PASS_TRAIN_ONLY_PLAN', train_images=len(rows), tasks=audit_tasks,
                 label_order=order, metric_groups=groups, train_positive_counts=counts,
                 hidden_future_positive_occurrences=missed_positive,
                 observed_positive_counts={k: cumulative[k][1] for k in labels},
                 image_replay=False, samples_assigned_once=True, test_accessed=False,
                 validation_used_for_plan=False,
                 allocation='earliest task containing any positive label; cumulative known-label mask',
                 fit_meta='native non-RL multi-label A uses all current images; no internal meta',
                 expected_steps_per_arm=sum(t['expected_steps'] for t in tasks)))
