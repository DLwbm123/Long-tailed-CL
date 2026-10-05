"""One explicitly configured multi-label trajectory. No default NIH task split.

Read JSON configuration from stdin so deployment can use a neutral entry path.
This module has no import-time dataset, model, CUDA, or training side effects.
"""
import copy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_core import METHODS, binary_weights, scores, update
from ml_data import NIH_LABELS, MultiLabelImages, read_manifest, validate_plan
from ml_metrics import evaluate, groups
from ml_statistics import empty, add, translated, translation, ridge


class ApartFeatures(nn.Module):
    """Reuse the existing locked pretrained backbone and its adapter pools."""
    def __init__(self, legacy_repo, weight, labels, device, seed):
        super().__init__()
        root = Path(legacy_repo)
        if not Path(weight).is_file():
            raise ValueError('An existing pretrained checkpoint is required; no automatic download')
        sys.path.insert(0, str(root / 'tools'))
        from run_medical_v2 import Learner
        args = json.loads((root / 'third_party/APART/exps/apart_cifar_shuffle.json').read_text())
        args.update(nb_classes=labels, nb_tasks=1, init_cls=labels, increment=labels,
                    seed=seed, device=[torch.device(device)], medical_v2=True,
                    locked_weight_path=str(weight), concm_stage1=False,
                    concm_stage1_eval_calibration=False, calibration_rule='none',
                    batchwise_prompt=False, shared_prompt_pool=False, shared_prompt_key=False)
        args.pop('longtail', None)
        self.learner = Learner(args)
        self.net = self.learner._network.to(device).eval()
        for name, parameter in self.net.named_parameters():
            parameter.requires_grad_('.pool.pool.' in name or '.pool_few.pool.' in name)
        for pool in (self.net.backbone.pool, self.net.backbone.pool_few):
            pool.batchwise_prompt = False
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError('Adapter parameters not found')
        self.dim = 1536

    def forward(self, x):
        from ct3p_core import features
        return features(self.net, x, self.learner)


def extract(encoder, loader, device):
    features, targets, masks = [], [], []
    encoder.eval()
    with torch.no_grad():
        for x, y, mask in loader:
            features.append(encoder(x.to(device)).cpu().numpy())
            targets.append(y.numpy())
            masks.append(mask.numpy())
    if not features:
        raise ValueError('Empty extraction loader')
    return np.concatenate(features), np.concatenate(targets), np.concatenate(masks)


class Trainer:
    def __init__(self, encoder, classes, method='A', seed=74002, device='cpu'):
        if method not in (*METHODS, 'F_S'):
            raise ValueError('Unknown condition')
        self.encoder, self.device, self.method = encoder.to(device), torch.device(device), method
        self.bank = empty(encoder.dim, classes)
        self.policy = torch.Generator(device=self.device).manual_seed(seed + 71000003)
        self.sigma, self.steps, self.task = .5, 0, 0
        self.head = None

    def fit_task(self, canonical_loader, train_loader, epochs, lr=.0003, before_step=lambda: None):
        if epochs < 1:
            raise ValueError('Positive epoch count required')
        self.task += 1
        teacher = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        before, y, mask = extract(teacher, canonical_loader, self.device)
        weights_cpu = binary_weights(torch.from_numpy(y), torch.from_numpy(mask))
        prior = self.bank
        bootstrap = add(copy.deepcopy(prior), before, y, mask)
        W_start, start_audit = ridge(bootstrap)
        del bootstrap
        weights = weights_cpu.to(self.device)
        head = torch.as_tensor(W_start, device=self.device, dtype=torch.float32)
        old_labels = torch.tensor(prior['n'].sum(1) > 0, device=self.device)
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        records = []
        if self.method != 'F_S' or self.task == 1:
            for epoch in range(epochs):
                self.encoder.eval()  # Preserve frozen-backbone behavior; adapters still have gradients.
                if hasattr(train_loader.dataset, 'epoch'):
                    train_loader.dataset.epoch = epoch + 1
                for batch, (x, by, bm) in enumerate(train_loader):
                    before_step()
                    self.sigma, diagnostics, _ = update(
                        self.encoder, teacher, optimizer, x.to(self.device), by.to(self.device),
                        bm.to(self.device), head, weights, 'A' if self.method == 'F_S' else self.method,
                        self.task, self.sigma, self.policy, old_labels)
                    self.steps += not diagnostics['skipped']
                    records.append(dict(task=self.task, epoch=epoch + 1, batch=batch, **diagnostics))
                scheduler.step()
        after, after_y, after_mask = extract(self.encoder, canonical_loader, self.device)
        if not np.array_equal(y, after_y) or not np.array_equal(mask, after_mask):
            raise ValueError('Canonical extraction order changed')
        shift, transport_audit = translation(before, after, y, mask, weights_cpu.numpy())
        self.bank = add(translated(prior, shift), after, y, mask)
        W_final, final_audit = ridge(self.bank)
        self.head = torch.as_tensor(W_final, device=self.device, dtype=torch.float32)
        return dict(W_start=W_start, W_final=W_final, start_head=start_audit,
                    final_head=final_audit, translation=transport_audit, updates=records)

    def state_dict(self):
        return dict(encoder={k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()},
                    bank=self.bank, head=self.head.detach().cpu(), sigma=self.sigma, steps=self.steps,
                    task=self.task, method=self.method, policy_rng=self.policy.get_state(),
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    def load_state_dict(self, state):
        if state['method'] != self.method:
            raise ValueError('Checkpoint condition mismatch')
        self.encoder.load_state_dict(state['encoder'], strict=True)
        self.bank, self.head = copy.deepcopy(state['bank']), state['head'].to(self.device)
        self.sigma, self.steps, self.task = state['sigma'], state['steps'], state['task']
        self.policy.set_state(state['policy_rng'])
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])


def run(config):
    if config.get('formal_authorized') is not True:
        raise ValueError('Formal training is disabled; provide an approved explicit protocol first')
    root, output = Path(config['data_root']), Path(config['output'])
    if not (root / 'prepared/READY.json').is_file():
        raise ValueError('Dataset download/extraction has not passed its completion check')
    if json.loads((root / 'prepared/READY.json').read_text())['status'] != 'READY':
        raise ValueError('Dataset is not ready')
    if output.exists():
        raise ValueError('Use a new output directory; no overwrite or automatic experiment retry')
    labels = tuple(config.get('labels', NIH_LABELS))
    if labels != NIH_LABELS:
        raise ValueError('NIH entry point retains all fourteen official labels')
    rows = read_manifest(root / 'prepared/manifest.csv', labels)
    plan = json.loads(Path(config['plan']).read_text())
    plan_audit = validate_plan(rows, plan, labels)
    budget_steps, budget_wall = int(config['max_steps']), float(config['max_wall_seconds'])
    if budget_steps <= 0 or budget_wall <= 0:
        raise ValueError('Explicit positive step and wall budgets are required')
    started = time.monotonic()
    seed = int(config['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    encoder = ApartFeatures(config['legacy_repo'], config['weight'], len(labels), config['device'], seed)
    trainer = Trainer(encoder, len(labels), config['method'], seed, config['device'])
    from run_medical_v2 import transform
    output.mkdir(parents=True)
    (output / 'INPUT.json').write_text(json.dumps(config, indent=2))
    def save(name, value):
        p = output / name
        temp = p.with_suffix('.part')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False))
        temp.replace(p)
    def budget():
        if trainer.steps >= budget_steps or time.monotonic() - started >= budget_wall:
            raise RuntimeError('INCOMPLETE_BUDGET')
    def loader(names, visible, training, task):
        dataset = MultiLabelImages(rows, root, names, visible, transform(training),
                                   labels=labels, training=training, seed=seed + task * 100003)
        return DataLoader(dataset, batch_size=int(config.get('batch_size', 48)), shuffle=training,
                          num_workers=0, generator=torch.Generator().manual_seed(seed + task * 2003))
    save('STATUS.json', dict(status='RUNNING', phase='train', plan=plan_audit))
    try:
        for index, task in enumerate(plan['tasks'], 1):
            budget()
            result = trainer.fit_task(loader(task['train_images'], task['visible_labels'], False, index),
                                      loader(task['train_images'], task['visible_labels'], True, index),
                                      int(config['epochs']), float(config.get('lr', .0003)), budget)
            torch.save(trainer.state_dict(), output / f'stage_{index}.pt')
            save(f'stage_{index}_diagnostics.json', {k: v for k, v in result.items() if not k.startswith('W_')})
        # Evaluation starts only after the entire configured trajectory is trained.
        save('STATUS.json', dict(status='RUNNING', phase='evaluate', trained_stages=trainer.task))
        reports = []
        for index, task in enumerate(plan['tasks'], 1):
            if time.monotonic() - started >= budget_wall:
                raise RuntimeError('INCOMPLETE_BUDGET')
            state = torch.load(output / f'stage_{index}.pt', map_location='cpu', weights_only=False)
            trainer.load_state_dict(state)
            seen = [c for c, active in zip(labels, trainer.bank['n'].sum(1) > 0) if active]
            z, y, mask = extract(trainer.encoder, loader(plan['validation_images'], seen, False, index), trainer.device)
            logits = np.c_[z, np.ones(len(z))] @ trainer.head.cpu().numpy()
            report = evaluate(logits, y, mask, labels, config.get('threshold', .5))
            report['groups'] = groups(report, plan.get('metric_groups', {}))
            report.update(task=index, head='W_final', split='validation', method=trainer.method)
            reports.append(report)
        save('metrics.json', reports)
        save('STATUS.json', dict(status='COMPLETE', training_steps=state['steps'], stages=len(reports),
                                 official_test_evaluated=False, independent_confirmation=False))
    except BaseException as exc:
        save('STATUS.json', dict(status='INCOMPLETE', error=str(exc), training_steps=trainer.steps))
        raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
