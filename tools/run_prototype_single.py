"""Bounded single-label prototype/analytic trajectory. Private config on stdin."""
import copy
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from prototype_analytic import empty, append, ridge, transport
from run_multilabel import ApartFeatures


def save(path, value):
    temp = path.with_suffix('.part')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def manifests(config):
    parts = []
    for split in ('train', 'val'):
        with (Path(config['manifest']) / (split + '.csv')).open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows or any(r['split'] != split for r in rows):
            raise ValueError('Manifest split mismatch')
        for r in rows:
            p = Path(r['relative_path'])
            if p.is_absolute() or '..' in p.parts or not r['identity_component']:
                raise ValueError('Unsafe path or missing identity component')
            r['label'] = int(r['original_label'])
        if len({r['relative_path'] for r in rows}) != len(rows):
            raise ValueError('Duplicate paths')
        parts.append(rows)
    train, val = parts
    for key in ('identity_component', 'relative_path'):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError('Training/validation overlap')
    groups = {}
    for r in train:
        groups.setdefault(r['identity_component'], set()).add(r['label'])
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError('Identity spans classes; explicit protocol needed')
    labels = set(config['order'])
    if len(labels) != len(config['order']) or labels != {r['label'] for r in train}:
        raise ValueError('Order must cover all training classes exactly once')
    if labels != {r['label'] for r in val}:
        raise ValueError('Validation class coverage mismatch')
    return train, val


class Images(Dataset):
    def __init__(self, rows, root, transform, seed):
        self.rows, self.root, self.transform, self.seed, self.epoch = rows, Path(root).resolve(), transform, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path = (self.root / self.rows[i]['relative_path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image escapes root')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.epoch * 1000003 + i)
            with Image.open(path) as image:
                x = self.transform(image.convert('RGB'))
        return x, self.rows[i]['label']


def extract(encoder, loader, budget):
    xs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            budget()
            xs.append(encoder(x.cuda()).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def run(config):
    output = Path(config['output'])
    if output.exists():
        raise ValueError('Fresh output required; no overwrite/retry')
    if config['method'] not in ('stage_global', 'epoch_global', 'epoch_local'):
        raise ValueError('Unknown arm')
    train, val = manifests(config)
    output.mkdir(parents=True)
    save(output / 'INPUT.private.json', config)
    started = time.monotonic(); steps = 0
    def budget():
        if time.monotonic() - started >= config['max_wall_seconds']:
            raise RuntimeError('INCOMPLETE_WALL_BUDGET')
    def status(**kw):
        save(output / 'STATUS.json', dict(steps=steps, elapsed_seconds=time.monotonic()-started, **kw))
    try:
        seed = config['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(4)
        encoder = ApartFeatures(config['legacy_repo'], config['weight'], len(config['order']), 'cuda:0', seed)
        from run_medical_v2 import transform
        def loader(rows, training, task):
            ds = Images(rows, config['images'], transform(training), seed + task * 100003)
            return DataLoader(ds, batch_size=config['batch_size'], shuffle=training, num_workers=0,
                              generator=torch.Generator().manual_seed(seed + task * 2003))
        bank = empty(encoder.dim); seen = []; diagnostics = []
        tasks = [config['order'][i:i+config['increment']] for i in range(0, len(config['order']), config['increment'])]
        for task, classes in enumerate(tasks, 1):
            budget(); seen += classes
            canonical = loader([r for r in train if r['label'] in classes], False, task)
            training = loader(canonical.dataset.rows, True, task)
            teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
            before, y = extract(teacher, canonical, budget)
            W, residual = ridge(append(bank, before, y, classes))
            head = torch.tensor(W, dtype=torch.float32, device='cuda')
            optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=config['lr'], weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=1e-5)
            lookup = torch.tensor([seen.index(c) if c in seen else -1 for c in range(len(config['order']))], device='cuda')
            counts = {c: int((y == c).sum()) for c in classes}
            sample_weights = torch.tensor([len(y)/(len(classes)*counts[c]) if c in classes else 0.
                                           for c in range(len(config['order']))], device='cuda')
            for epoch in range(1, config['epochs'] + 1):
                training.dataset.epoch = epoch; encoder.eval(); losses = []
                status(status='RUNNING', phase='train', task=task, epoch=epoch)
                for x, labels in training:
                    budget()
                    if steps >= config['max_steps']:
                        raise RuntimeError('INCOMPLETE_STEP_BUDGET')
                    x, labels = x.cuda(), labels.cuda()
                    optimizer.zero_grad(set_to_none=True)
                    z = encoder(x)
                    with torch.no_grad():
                        reference = teacher(x)
                    target = torch.nn.functional.one_hot(lookup[labels], len(seen)).float()
                    fit = (.5 * (z @ head - target).square().sum(1) * sample_weights[labels]).mean()
                    fd = (z - reference).square().sum(1).mean()
                    loss = fit + 10. * fd
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
                    optimizer.step(); steps += 1
                    losses.append([float(fit.detach()), float(fd.detach()), float(norm)])
                scheduler.step()
                # All arms perform matched canonical passes and solves. Only head assignment differs.
                after, after_y = extract(encoder, canonical, budget)
                if not np.array_equal(y, after_y):
                    raise ValueError('Canonical sample order changed')
                shifted, audit = transport(bank, before, after, y, classes, config['method'] == 'epoch_local')
                candidate = append(shifted, after, y, classes)
                W_new, residual = ridge(candidate)
                diagnostics.append(dict(task=task, epoch=epoch, steps=steps, ridge_residual=residual,
                    head_relative_change=float(np.linalg.norm(W_new-head.cpu().numpy()) / max(np.linalg.norm(W_new), 1e-12)),
                    mean_fit_loss=float(np.mean(losses, axis=0)[0]), mean_feature_loss=float(np.mean(losses, axis=0)[1]),
                    **audit))
                if config['method'] != 'stage_global' or epoch == config['epochs']:
                    head = torch.tensor(W_new, dtype=torch.float32, device='cuda')
                save(output / 'diagnostics.json', diagnostics)
            bank = candidate
            adapter = {k: v.detach().cpu() for k, v in encoder.named_parameters() if v.requires_grad}
            torch.save(dict(adapter=adapter, head=head.cpu(), seen=seen.copy(), bank=bank, steps=steps), output / f'stage_{task}.pt')
            del teacher, before, after
        # No validation is inspected until the whole trajectory is trained.
        reports = []
        ranked = sorted(config['order'], key=lambda c: (-sum(r['label']==c for r in train), c))
        tail = set(ranked[len(ranked)//2:])
        for task in range(1, len(tasks)+1):
            status(status='RUNNING', phase='evaluate', task=task)
            state = torch.load(output / f'stage_{task}.pt', map_location='cpu', weights_only=False)
            encoder.load_state_dict(state['adapter'], strict=False)
            seen = state['seen']
            z, y = extract(encoder, loader([r for r in val if r['label'] in seen], False, task), budget)
            pred = np.asarray(seen)[np.argmax(z @ state['head'].numpy(), axis=1)]
            recalls = {str(c): float((pred[y==c] == c).mean()) for c in seen}
            reports.append(dict(task=task, seen=seen, validation_n=len(y), accuracy=float((pred==y).mean()),
                balanced_accuracy=float(np.mean(list(recalls.values()))),
                tail_recall=float(np.mean([recalls[str(c)] for c in seen if c in tail])) if tail.intersection(seen) else None,
                per_class_recall=recalls, per_class_n={str(c): int((y==c).sum()) for c in seen}))
        forgetting = []
        for c in config['order']:
            earlier = [r['per_class_recall'][str(c)] for r in reports[:-1] if str(c) in r['per_class_recall']]
            if earlier:
                forgetting.append(max(earlier)-reports[-1]['per_class_recall'][str(c)])
        save(output / 'metrics.json', dict(stages=reports, average_incremental_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reports])),
            final_balanced_accuracy=reports[-1]['balanced_accuracy'], final_tail_recall=reports[-1]['tail_recall'],
            forgetting=float(np.mean(forgetting)), forgetting_classes=len(forgetting),
            test_accessed=False, independent_confirmation=False))
        status(status='COMPLETE', stages=len(tasks), peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
