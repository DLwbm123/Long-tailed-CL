"""Frozen class schedules, LT sampling and reporting for the three-dataset run."""
import numpy as np


def task_blocks(order, sizes):
    if (not sizes or any(type(n) is not int or n < 1 for n in sizes)
            or sum(sizes) != len(order) or sorted(order) != list(range(len(order)))):
        raise ValueError('Task sizes must partition every contiguous dataset label exactly once')
    edges = np.cumsum([0] + sizes)
    return [order[a:b] for a, b in zip(edges[:-1], edges[1:])]


def cifar_indices(labels):
    labels = np.asarray(labels)
    if len(labels) != 50000 or not np.array_equal(np.bincount(labels), np.full(100, 500)):
        raise ValueError('Expected the original balanced CIFAR-100 training split')
    # Match the reference LT-CIL loader: floor counts and legacy RandomState(0).
    rng = np.random.RandomState(0)
    selected = []
    for c in range(100):
        indices = np.flatnonzero(labels == c)
        rng.shuffle(indices)
        selected.extend(indices[:int(500 * .01 ** (c / 99.))].tolist())
    return selected


def benchmark_metrics(stages, tasks, train_counts):
    last = stages[-1]
    groups = dict(many=[c for c, n in train_counts.items() if n > 100],
                  medium=[c for c, n in train_counts.items() if 20 <= n <= 100],
                  few=[c for c, n in train_counts.items() if n < 20])
    frequency = {}
    for name, classes in groups.items():
        count = sum(last['per_class_n'][str(c)] for c in classes)
        correct = sum(last['per_class_n'][str(c)] * last['per_class_recall'][str(c)] for c in classes)
        frequency[name] = dict(classes=classes, n=count, accuracy=correct/count if count else None,
            macro_recall=float(np.mean([last['per_class_recall'][str(c)] for c in classes])) if classes else None)
    matrix = []
    for stage in stages:
        row = []
        for classes in tasks:
            if not set(classes).issubset(stage['seen']):
                row.append(None)
                continue
            count = sum(stage['per_class_n'][str(c)] for c in classes)
            row.append(sum(stage['per_class_n'][str(c)] * stage['per_class_recall'][str(c)] for c in classes)/count)
        matrix.append(row)
    drops = [max(row[t] for row in matrix[t:-1]) - matrix[-1][t] for t in range(len(tasks)-1)]
    return dict(average_incremental_accuracy=float(np.mean([s['accuracy'] for s in stages])),
                final_accuracy=last['accuracy'], task_accuracy_matrix=matrix,
                task_forgetting=float(np.mean(drops)) if drops else None,
                class_frequency=frequency)


def check():
    labels = np.arange(100).repeat(500)
    ids = cifar_indices(labels)
    counts = np.bincount(labels[ids])
    assert len(set(ids)) == len(ids) == 10847 and counts[0] == 500 and counts[-1] == 5
    assert np.all(counts[:-1] >= counts[1:]) and ids == cifar_indices(labels)
    assert list(map(len, task_blocks(list(range(100)), [50, 10, 10, 10, 10, 10]))) == [50, 10, 10, 10, 10, 10]
    try:
        task_blocks([0, 0], [1, 1])
        raise AssertionError('Duplicate class accepted')
    except ValueError:
        pass
    stages = [dict(accuracy=.8, seen=[0], per_class_n={'0':10}, per_class_recall={'0':.8}),
              dict(accuracy=.6, seen=[0,1], per_class_n={'0':10,'1':10}, per_class_recall={'0':.5,'1':.7})]
    m = benchmark_metrics(stages, [[0],[1]], {0:500,1:5})
    assert abs(m['task_forgetting']-.3) < 1e-12 and m['class_frequency']['medium']['accuracy'] is None
    print('PASS: reference LT sampling, task coverage, task forgetting and empty frequency groups')


if __name__ == '__main__':
    check()
