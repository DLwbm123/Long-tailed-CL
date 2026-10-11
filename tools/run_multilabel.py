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
from ml_statistics import empty, add, translated, translation, ridge, observation_weights
import residual_drift


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


def extract(encoder, loader, device, *, with_local=False, before_batch=lambda: None):
    features, targets, masks = [], [], []
    regions = []
    encoder.eval()
    with torch.no_grad():
        for x, y, mask in loader:
            before_batch()
            captured = []
            hook = encoder.net.backbone.norm.register_forward_hook(
                lambda m, a, out: captured.append(out.detach())) if with_local else None
            try:
                features.append(encoder(x.to(device)).cpu().numpy())
                if with_local:
                    from module_additions import local_parts
                    regions.append(local_parts(captured).cpu().numpy())
            finally:
                if hook is not None:
                    hook.remove()
            targets.append(y.numpy())
            masks.append(mask.numpy())
    if not features:
        raise ValueError('Empty extraction loader')
    result = (np.concatenate(features), np.concatenate(targets), np.concatenate(masks))
    return (*result, np.concatenate(regions)) if with_local else result


class Trainer:
    def __init__(self, encoder, classes, method='A', seed=74002, device='cpu', *, local_transport=False,
                 drift_weight=0., contrast_prior=None):
        if method not in (*METHODS, 'F_S'):
            raise ValueError('Unknown condition')
        self.encoder, self.device, self.method = encoder.to(device), torch.device(device), method
        self.bank = empty(encoder.dim, classes)
        self.policy = torch.Generator(device=self.device).manual_seed(seed + 71000003)
        self.sigma, self.steps, self.task = .5, 0, 0
        self.head = None
        self.drift_weight = residual_drift.strength(drift_weight)
        if contrast_prior is not None and (contrast_prior not in ('difference', 'difference_shared_removed') or
                not local_transport or method != 'A' or self.drift_weight):
            raise ValueError('Contrast prior requires local non-RL A without residual drift')
        self.contrast_prior = contrast_prior
        if self.drift_weight and method != 'A':
            raise ValueError('Residual drift is supported only for non-RL A')
        self.drift = None
        if local_transport and method != 'A':
            raise ValueError('Local binary prior is authorized only for non-RL A')
        self.local = None
        if local_transport:
            from rfmid_local import empty_local
            self.local = empty_local(classes, encoder.dim)

    def fit_task(self, canonical_loader, train_loader, epochs, lr=.0003, before_step=lambda: None,
                 *, preflight=False, before_extract=lambda: None):
        if epochs < 1:
            raise ValueError('Positive epoch count required')
        self.task += 1
        teacher = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        extracted = extract(teacher, canonical_loader, self.device, with_local=self.local is not None,
                            before_batch=before_extract)
        before, y, mask = extracted[:3]
        weights_cpu = binary_weights(torch.from_numpy(y), torch.from_numpy(mask))
        prior = self.bank
        bootstrap = add(copy.deepcopy(prior), before, y, mask)
        activation = dict(extra_columns=0, prior_delta_norm=0.)
        def fit_head(bank, local):
            drift_prior = None
            if self.drift_weight and self.drift is not None:
                drift_prior = residual_drift.add_to_prior(torch.zeros(self.encoder.dim + 1,
                    self.encoder.dim + 1, dtype=torch.float64), torch.from_numpy(self.drift),
                    self.drift_weight, bias=True).numpy()
            if local is None and drift_prior is None:
                return ridge(bank)
            from rfmid_local import penalty
            if local is not None and not np.array_equal(bank['n'], local['n']):
                raise ValueError('Local and global observed binary counts differ')
            if local is not None:
                activation.update(extra_columns=4 * int((local['n'] > 0).sum()), prior_delta_norm=0.)
                if self.contrast_prior is not None:
                    activation.update(extra_columns=4 * int((local['n'] > 0).all(1).sum()),
                        contrast_raw_energy=0., contrast_retained_energy=0.,
                        contrast_shared_rank_max=0, contrast_shared_leakage_max=0.)
            def prior(label):
                R = np.eye(self.encoder.dim + 1) if local is None else (
                    penalty(local, label) if self.contrast_prior is None else None)
                if self.contrast_prior is not None:
                    from rfmid_contrast import columns, penalty as contrast_penalty
                    projected = self.contrast_prior == 'difference_shared_removed'
                    _, diag = columns(local, label, remove_shared=projected)
                    R = contrast_penalty(local, label, remove_shared=projected)
                    activation['contrast_raw_energy'] += diag['raw_energy']
                    activation['contrast_retained_energy'] += diag['retained_energy']
                    activation['contrast_shared_rank_max'] = max(activation['contrast_shared_rank_max'], diag['shared_rank'])
                    activation['contrast_shared_leakage_max'] = max(activation['contrast_shared_leakage_max'], diag['shared_leakage'])
                if local is not None:
                    activation['prior_delta_norm'] += float(np.linalg.norm(R - np.eye(len(R))))
                if drift_prior is not None:
                    R = R + drift_prior
                return R
            return ridge(bank, penalty=prior)
        start_local = None
        if self.local is not None:
            from rfmid_local import add_local, transport_local
            start_local = add_local(self.local, extracted[3], y, mask)
        W_start, start_audit = fit_head(bootstrap, start_local)
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
                    if preflight:
                        from ml_core import objective
                        optimizer.zero_grad(set_to_none=True)
                        x, by, bm = x.to(self.device), by.to(self.device), bm.to(self.device)
                        with torch.no_grad():
                            target = teacher(x)
                        loss, terms, coefficient, _, _diag, actions = objective(
                            self.encoder(x), target, head, by, bm, weights, 'A', self.task,
                            self.sigma, self.policy, old_labels)
                        if not torch.isfinite(loss) or coefficient != 0 or actions is not None:
                            raise ArithmeticError('Invalid non-RL zero-update preflight loss')
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                        if not float(grad) > 0:
                            raise ArithmeticError('Zero preflight adapter gradient')
                        result = dict(preflight=True, loss=float(loss.detach()), bce=float(terms[0].detach()),
                                    fd=float(terms[1].detach()), gradient_norm=float(grad),
                                    optimizer_updates=0, policy_updates=0, activation=activation,
                                    max_solve_residual=max(v for v in start_audit['residuals'] if v is not None))
                        if self.drift_weight:
                            result['residual_drift'] = dict(weight=self.drift_weight,
                                residual_trace=0. if self.drift is None else float(np.trace(self.drift)),
                                first_task_zero_residual=self.task == 1, nonzero_activation_not_proven=self.task == 1)
                        return result
                    self.sigma, diagnostics, _ = update(
                        self.encoder, teacher, optimizer, x.to(self.device), by.to(self.device),
                        bm.to(self.device), head, weights, 'A' if self.method == 'F_S' else self.method,
                        self.task, self.sigma, self.policy, old_labels)
                    self.steps += not diagnostics['skipped']
                    records.append(dict(task=self.task, epoch=epoch + 1, batch=batch, **diagnostics))
                scheduler.step()
        after_extract = extract(self.encoder, canonical_loader, self.device, with_local=self.local is not None,
                                before_batch=before_extract)
        after, after_y, after_mask = after_extract[:3]
        if not np.array_equal(y, after_y) or not np.array_equal(mask, after_mask):
            raise ValueError('Canonical extraction order changed')
        shift, transport_audit = translation(before, after, y, mask, weights_cpu.numpy())
        drift_audit = None
        if self.drift_weight:
            row_w = observation_weights(y, mask, weights_cpu.numpy())
            U, drift_audit = residual_drift.estimate(torch.as_tensor(before, dtype=torch.float64),
                torch.as_tensor(after, dtype=torch.float64), torch.from_numpy(row_w), torch.from_numpy(shift))
            self.drift = U.numpy()
        self.bank = add(translated(prior, shift), after, y, mask)
        local_audit = dict(regional_shift_norm=0.)
        if self.local is not None:
            moved, local_audit = transport_local(self.local, extracted[3], after_extract[3], y, mask, weights_cpu.numpy())
            self.local = add_local(moved, after_extract[3], y, mask)
        W_final, final_audit = fit_head(self.bank, self.local)
        self.head = torch.as_tensor(W_final, device=self.device, dtype=torch.float32)
        result = dict(W_start=W_start, W_final=W_final, start_head=start_audit,
                    final_head=final_audit, translation=transport_audit, updates=records,
                    local_activation=activation, local_transport=local_audit)
        if drift_audit is not None:
            result['residual_drift'] = dict(weight=self.drift_weight, **drift_audit,
                shared_observed_pair_matrix=True, start_uses_previous_boundary=True,
                boundary_replaces_previous_matrix=True, detached=True)
        return result

    def state_dict(self):
        result = dict(encoder={k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()},
                    bank=self.bank, head=self.head.detach().cpu(), sigma=self.sigma, steps=self.steps,
                    task=self.task, method=self.method, policy_rng=self.policy.get_state(),
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                    local=self.local)
        if self.drift_weight:
            result.update(residual_drift_matrix=self.drift, residual_drift_weight=self.drift_weight,
                          residual_drift_version='residual_drift_v1')
        if self.contrast_prior is not None:
            result.update(contrast_prior=self.contrast_prior, contrast_prior_version='binary_contrast_shared_projection_v1')
        return result

    def load_state_dict(self, state):
        if state['method'] != self.method:
            raise ValueError('Checkpoint condition mismatch')
        if (self.local is None) != (state.get('local') is None):
            raise ValueError('Checkpoint local binary prior differs')
        if state.get('residual_drift_weight', 0.) != self.drift_weight:
            raise ValueError('Checkpoint residual drift weight differs')
        if state.get('contrast_prior') != self.contrast_prior or (self.contrast_prior is not None and
                state.get('contrast_prior_version') != 'binary_contrast_shared_projection_v1'):
            raise ValueError('Checkpoint contrast prior version/mode differs')
        if self.drift_weight:
            if state.get('residual_drift_version') != 'residual_drift_v1':
                raise ValueError('Checkpoint residual drift version differs')
            U = np.asarray(state['residual_drift_matrix'], dtype=np.float64)
            residual_drift.validate(torch.from_numpy(U), self.encoder.dim)
            self.drift = U.copy()
        self.encoder.load_state_dict(state['encoder'], strict=True)
        self.bank, self.head = copy.deepcopy(state['bank']), state['head'].to(self.device)
        self.sigma, self.steps, self.task = state['sigma'], state['steps'], state['task']
        self.local = copy.deepcopy(state.get('local'))
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
