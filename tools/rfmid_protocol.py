"""Train-only RFMiD task allocation; image identities stay in the private plan."""
import csv
import math
import random

from prepare_rfmid_ml import BENCHMARK_LABELS


RECURRENT_INTRODUCED = [
    ['Normal', 'ARMD', 'ERM', 'MS', 'AH', 'PT', 'RT', 'EDN', 'MHL'],
    ['DR', 'DN', 'CSR', 'TV', 'RP'],
    ['MH', 'MYA', 'LS', 'CRVO', 'ST'],
    ['BRVO', 'ODC', 'ODP', 'AION', 'OTHER'],
    ['TSLN', 'ODE', 'RS', 'CRS', 'RPEC'],
]

COVERAGE_INTRODUCED = [
    ['Normal', 'DR', 'MH', 'TSLN', 'LS', 'CSR', 'ODC', 'ODP', 'AION'],
    ['ARMD', 'TV', 'PT', 'MHL', 'RP'],
    ['DN', 'ERM', 'ST', 'RT', 'EDN'],
    ['MYA', 'MS', 'AH', 'ODE', 'OTHER'],
    ['BRVO', 'CRVO', 'RS', 'CRS', 'RPEC'],
]
COVERAGE_FOCUS = ('AION', 'ODP', 'LS', 'CSR')
COVERAGE_ANCHORS = ('Normal', 'DR', 'MH', 'ODC', 'TSLN')


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


def recurrent_plan(rows, proposal, variant='balanced_recurrent_v2'):
    """Validate the approved private assignment without solving or changing it."""
    labels = list(BENCHMARK_LABELS)
    if len(rows) != 1920 or proposal.get('labels') != labels:
        raise ValueError('Recurrent RFMiD row count or label view differs')
    if variant not in ('balanced_recurrent_v2', 'negative_coverage_v3'):
        raise ValueError('Unapproved recurrent variant')
    introduced = COVERAGE_INTRODUCED if variant == 'negative_coverage_v3' else RECURRENT_INTRODUCED
    if [t['introduced_labels'] for t in proposal['tasks']] != introduced:
        raise ValueError('Recurrent RFMiD introduction order differs')
    by_id = {r['ID']: r for r in rows}
    identities = [i for t in proposal['tasks'] for i in t['train_ids']]
    if len(by_id) != 1920 or len(identities) != 1920 or len(set(identities)) != 1920 or set(identities) != set(by_id):
        raise ValueError('Recurrent RFMiD images lost, repeated or unknown')
    counts = {k: sum(int(r[k]) for r in rows) for k in labels}
    metric_groups = build_plan(rows)[0]['metric_groups']
    cumulative = {k: [0, 0] for k in labels}
    tasks, audits, seen = [], [], []
    pairs = 0
    for index, t in enumerate(proposal['tasks'], 1):
        if t['task'] != index or len(t['train_ids']) != 384 or t['expected_steps'] != 12:
            raise ValueError('Recurrent task identity, size or budget differs')
        new = t['introduced_labels']; old = seen.copy(); seen += new
        current = [by_id[i] for i in t['train_ids']]
        if any(int(r[k]) for r in current for k in labels if k not in seen):
            raise ValueError('Future positive supervision would be lost')
        pos = {k: sum(int(r[k]) for r in current) for k in seen}
        neg = {k: 384 - pos[k] for k in seen}
        if any(pos[k] < min(5, math.ceil(counts[k] / 2)) or neg[k] < 5 for k in new):
            raise ValueError('Recurrent new label lacks its frozen sign support')
        if any(pos[k] < 1 or neg[k] < 1 for k in old):
            raise ValueError('Recurrent old label lacks a sign')
        if variant == 'negative_coverage_v3':
            if pos['Normal'] != (81 if index == 1 else 80):
                raise ValueError('Normal coverage differs')
            for k in COVERAGE_ANCHORS + COVERAGE_FOCUS:
                minimum = counts[k] // 5 if k in COVERAGE_FOCUS else math.floor(counts[k] / 5 * .8)
                if pos[k] < minimum:
                    raise ValueError('Focus/anchor recurrence differs')
        for k in old:
            intro = next(i for i, group in enumerate(introduced, 1) if k in group)
            minimum = 20 if k in ('Normal', 'DR', 'MH', 'ODC', 'TSLN') else max(
                1, min(5, (counts[k] - min(5, math.ceil(counts[k] / 2))) // (5 - intro)))
            if pos[k] < minimum:
                raise ValueError('Recurrent old positive quota differs')
        for k in seen:
            cumulative[k][0] += neg[k]; cumulative[k][1] += pos[k]
        pairs += sum(sum(int(r[k]) for k in seen) * (sum(int(r[k]) for k in seen) - 1) // 2 for r in current)
        tasks.append(dict(task=index, introduced_labels=new, visible_labels=seen.copy(),
                          train_ids=t['train_ids'], expected_steps=12))
        audits.append(dict(task=index, introduced_labels=new, visible_labels=seen.copy(),
            train_images=384, fit_images=384, meta_images=0, expected_steps=12,
            current_positive=pos, current_negative=neg,
            cumulative_observed_counts={k: cumulative[k].copy() for k in seen},
            old_positive_occurrences=sum(pos[k] for k in old),
            images_with_old_positive=sum(any(int(r[k]) for k in old) for r in current)))
    if sum(counts.values()) != 2470 or pairs != 650 or any(cumulative[k][1] != counts[k] for k in labels):
        raise ValueError('Recurrent positive or comorbidity conservation differs')
    plan = dict(labels=labels, tasks=tasks, metric_groups=metric_groups,
                train_seed=74002, order_tie_seed=74004, internal_meta=False)
    support = dict(status='PASS_TRAIN_ONLY_RECURRENT_PLAN', train_images=1920, tasks=audits,
        label_order=seen, metric_groups=metric_groups, train_positive_counts=counts,
        hidden_future_positive_occurrences={k: 0 for k in labels},
        observed_positive_counts={k: cumulative[k][1] for k in labels},
        image_replay=False, samples_assigned_once=True, test_accessed=False,
        validation_used_for_plan=False, allocation='frozen ' + variant + ' assignment; cumulative known mask',
        fit_meta='native A, all current images, no internal meta', expected_steps_per_arm=60,
        observed_positive_occurrences=2470, jointly_supervised_positive_pairs=650,
        old_positive_occurrences=sum(t['old_positive_occurrences'] for t in audits),
        observed_negative_occurrences=sum(v[0] for v in cumulative.values()))
    # Preserve the already frozen v2 aggregate schema for historical callers.
    if variant == 'balanced_recurrent_v2':
        support['allocation'] = 'frozen balanced recurrent assignment; cumulative known mask'
    else:
        if support['old_positive_occurrences'] != 1478 or support['observed_negative_occurrences'] != 34010:
            raise ValueError('Coverage frozen totals differ')
        for k in COVERAGE_FOCUS + COVERAGE_ANCHORS:
            if cumulative[k] != [1920 - counts[k], counts[k]]:
                raise ValueError('First-task labels lack full train sign coverage')
        support['normal_per_task'] = [t['current_positive']['Normal'] for t in audits]
        support['focus_full_train_negative_coverage'] = True
    return plan, support
