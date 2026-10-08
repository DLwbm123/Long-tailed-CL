"""Materialize a shared CIFAR LT subset and freeze aggregate dataset schedules."""
import csv
import json
from pathlib import Path
import pickle
import sys
from collections import Counter

import numpy as np
from PIL import Image

from lt_benchmark import cifar_indices, task_blocks
from run_prototype_single import manifests, save
from run_prototype_coherent import fit_split


def prepare(c):
    root = Path(c['root'])
    target = root/'cifar'
    target.mkdir()
    (target/'images').mkdir()
    (target/'manifests').mkdir()
    for source_split, split in [('train','train'),('test','val')]:
        # Existing canonical CIFAR Python files, never arbitrary downloaded pickles.
        with (Path(c['cifar_source'])/source_split).open('rb') as stream:
            data = pickle.load(stream, encoding='latin1')
        labels = np.asarray(data['fine_labels'])
        pixels = np.asarray(data['data']).reshape(-1,3,32,32).transpose(0,2,3,1)
        if pixels.dtype != np.uint8 or len(labels) != (50000 if split=='train' else 10000):
            raise ValueError('Unexpected canonical CIFAR shape or dtype')
        ids = cifar_indices(labels) if split=='train' else list(range(len(labels)))
        if split=='val' and not np.array_equal(np.bincount(labels), np.full(100,100)):
            raise ValueError('Expected balanced official CIFAR evaluation split')
        (target/'images'/split).mkdir()
        with (target/'manifests'/f'{split}.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=['original_label','split','relative_path','identity_component'])
            writer.writeheader()
            for i in ids:
                path = f'{split}/{i:05d}.png'
                Image.fromarray(pixels[i]).save(target/'images'/path)
                writer.writerow(dict(original_label=int(labels[i]),split=split,relative_path=path,
                                     identity_component=f'cifar-{source_split}-{i:05d}'))
        print(json.dumps(dict(split=split, exported=len(ids))),flush=True)
    public = {}
    for dataset in c['datasets']:
        classes = dataset['classes']
        rows, val = manifests(dict(dataset,order=list(range(classes))))
        counts = Counter(r['label'] for r in rows)
        ordered = sorted(counts,key=lambda k:(-counts[k],k))
        shuffled = (c['cifar_shuffle'] if dataset['name']=='CIFAR100LT'
                    else np.random.RandomState(c['seed']).permutation(classes).tolist())
        fit, meta = fit_split(rows,c['seed'])
        fit_counts = Counter(rows[i]['label'] for i in fit)
        meta_counts = Counter(rows[i]['label'] for i in meta)
        if set(fit_counts) != set(counts) or set(meta_counts) != set(counts):
            raise ValueError('Fit/meta class coverage missing')
        for order in (ordered,shuffled): task_blocks(order,dataset['task_sizes'])
        public[dataset['name']] = dict(train_n=len(rows), evaluation_n=len(val),classes=classes,
            class_counts=dict(sorted(counts.items())),fit_counts=dict(sorted(fit_counts.items())),
            meta_counts=dict(sorted(meta_counts.items())),
            evaluation_counts=dict(sorted(Counter(r['label'] for r in val).items())),
            imbalance_ratio=max(counts.values())/min(counts.values()),
            ordered=ordered,shuffled=shuffled,task_sizes=dataset['task_sizes'],
            evaluation_split=dataset['evaluation_split'],
            train_evaluation_identity_overlap=0,fit_meta_identity_overlap=0)
        print(dataset['name'],len(rows),len(val), 'manifest checks PASS',flush=True)
    save(root/'public/DATA_PROTOCOL.json',public)


if __name__ == '__main__':
    prepare(json.load(sys.stdin))
