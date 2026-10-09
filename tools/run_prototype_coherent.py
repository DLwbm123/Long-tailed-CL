"""Matched coherent prototype trajectories; private configuration on stdin."""
import copy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import prototype_coherent as method
from run_multilabel import ApartFeatures
from run_prototype_single import Images, extract, manifests, save
from lt_benchmark import task_blocks, benchmark_metrics


class IndexedImages(Images):
    def __getitem__(self, i):
        x, label = super().__getitem__(i)
        return x, label, i


def fit_split(rows, seed):
    held = set()
    for label in sorted({r['label'] for r in rows}):
        groups = sorted({r['identity_component'] for r in rows if r['label'] == label})
        if len(groups) < 2:
            raise ValueError('At least two identity groups per current class required')
        random.Random(seed+label*2003).shuffle(groups)
        held.update(groups[:min(len(groups)-1, max(1, round(.2*len(groups))))])
    fit = [i for i, r in enumerate(rows) if r['identity_component'] not in held]
    meta = [i for i, r in enumerate(rows) if r['identity_component'] in held]
    return fit, meta


def common_shift(before, after, labels):
    return torch.stack([(after-before)[labels == c].mean(0)
                        for c in sorted(labels.unique().tolist())]).mean(0)


def cpu_bank(bank):
    return dict(mu=bank['mu'].detach().cpu(), Q=bank['Q'].detach().cpu(), n=bank['n'],
                components=[{k: v.detach().cpu() if torch.is_tensor(v) else v
                             for k, v in c.items()} for c in bank['components']])


def run(config):
    arm = config['method']
    if arm not in ('cf_linear', 'cf_prototype', 'cf_weighted', 'cf_group', 'cf_group_min'):
        raise ValueError('Unknown coherent arm')
    execution_mode = config.get('execution_mode', 'strict')
    if execution_mode not in ('strict', 'diagnostic_proposal') or (execution_mode != 'strict' and arm != 'cf_group'):
        raise ValueError('Diagnostic execution requires cf_group')
    output = Path(config['output'])
    if output.exists():
        raise ValueError('Fresh output required; no overwrite or automatic retry')
    train, val = manifests(config)
    output.mkdir(parents=True)
    save(output/'INPUT.private.json', config)
    started, steps = time.monotonic(), 0
    def budget():
        if time.monotonic()-started >= config['max_wall_seconds']:
            raise RuntimeError('INCOMPLETE_WALL_BUDGET')
    def status(**kw):
        save(output/'STATUS.json', dict(steps=steps, elapsed_seconds=time.monotonic()-started, **kw))
    try:
        seed = config['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(4)
        status(status='RUNNING', phase='initialize')
        encoder = ApartFeatures(config['legacy_repo'], config['weight'], len(config['order']), 'cuda:0', seed)
        from run_medical_v2 import transform
        def loader(rows, training, task):
            ds = (IndexedImages if training else Images)(rows, config['images'], transform(training), seed+task*100003)
            workers = int(config.get('workers', 0))
            options = dict(multiprocessing_context='spawn', prefetch_factor=2,
                           persistent_workers=not training) if workers else {}
            return DataLoader(ds, batch_size=config['batch_size'], shuffle=training,
                num_workers=workers, generator=torch.Generator().manual_seed(seed+task*2003), **options)
        def features(model, data):
            x, y = extract(model, data, budget)
            return torch.as_tensor(x, device='cuda', dtype=torch.float64), torch.as_tensor(y, device='cuda')
        strength = 0. if arm == 'cf_linear' else 1.
        bank = method.empty(encoder.dim, 'cuda'); seen = []; diagnostics = []
        actor = torch.zeros(6, dtype=torch.float64, device='cuda', requires_grad=True)
        generator = torch.Generator(device='cuda').manual_seed(seed+911)
        sizes = config.get('task_sizes')
        if sizes is None:
            sizes = [len(config['order'][i:i+config['increment']])
                     for i in range(0, len(config['order']), config['increment'])]
        tasks = task_blocks(config['order'], sizes)
        for task, classes in enumerate(tasks, 1):
            budget(); seen += classes
            rows = [r for r in train if r['label'] in classes]
            fit_ids, meta_ids = fit_split(rows, config.get('split_seed', seed+task*100003))
            fit_ids = torch.as_tensor(fit_ids, device='cuda'); meta_ids = torch.as_tensor(meta_ids, device='cuda')
            canonical = loader(rows, False, task)
            training = loader([rows[i] for i in fit_ids.tolist()], True, task)
            teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
            status(status='RUNNING', phase='task_start_features', task=task)
            before, raw_y = features(teacher, canonical)
            lookup = torch.as_tensor([seen.index(c) if c in seen else -1 for c in range(len(config['order']))], device='cuda')
            y = lookup[raw_y]; current = before
            seeds = method.seed_components(before[fit_ids], y[fit_ids])
            group, difficulty = method.memberships(before, y, seeds)
            reference_actor = actor.detach().clone()
            actions = before.new_zeros(len(seeds))
            optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=config['lr'], weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=1e-5)
            for epoch in range(1, config['epochs']+1):
                budget(); encoder.eval(); training.dataset.epoch = epoch
                old = method.translate(bank, common_shift(before[fit_ids], current[fit_ids], y[fit_ids]))
                status(status='RUNNING', phase='controller', task=task, epoch=epoch)
                if arm in ('cf_weighted', 'cf_group', 'cf_group_min') and (
                        not config.get('preflight', False) or config.get('preflight_controller', False)):
                    actions, audit = method.controller(old, current[fit_ids], y[fit_ids], group[fit_ids],
                        difficulty[fit_ids], current[meta_ids], y[meta_ids], seeds, actor, reference_actor,
                        'gradient' if arm == 'cf_weighted' else 'group', generator,
                        reward_mode='minimum' if arm == 'cf_group_min' else 'mean',
                        execution_mode=execution_mode)
                else:
                    audit = dict(mode='uniform', optimizer_steps=0, accepted=True,
                                 selected_actions=actions.tolist())
                weights = method.sample_weights(y[fit_ids], group[fit_ids], difficulty[fit_ids], actions)
                candidate = method.append(old, current[fit_ids], y[fit_ids], group[fit_ids], weights)
                W, R = method.head(candidate, strength)
                neutral_weights = method.sample_weights(y[fit_ids], group[fit_ids], difficulty[fit_ids], torch.zeros_like(actions))
                activation = dict(prototype_strength=strength,
                    prototype_metric_relative_change=float((R-torch.eye(len(R),device=R.device,dtype=R.dtype)).norm()/len(R)**.5),
                    selected_action_max_abs=float(actions.abs().max()),
                    weight_l1_change=float((weights-neutral_weights).abs().sum()),
                    policy_optimizer_steps=audit['optimizer_steps'])
                reward_probe = None
                if config.get('preflight', False) and arm == 'cf_group_min':
                    state = method.descriptors(current[fit_ids], y[fit_ids], group[fit_ids], seeds, W)
                    proposed = method.policy_mean(actor, state).tanh()
                    probe_weights = method.sample_weights(y[fit_ids], group[fit_ids], difficulty[fit_ids], proposed)
                    probe_bank = method.append(old, current[fit_ids], y[fit_ids], group[fit_ids], probe_weights)
                    probe_head, _ = method.head(probe_bank)
                    reward = method.reward_change(method.class_ce(current[meta_ids], y[meta_ids], probe_head),
                        method.class_ce(current[meta_ids], y[meta_ids], W).detach(),
                        method.old_square_losses(old, probe_head), method.old_square_losses(old, W).detach(), 'minimum')
                    gradient = torch.autograd.grad(reward, actor)[0]
                    if not torch.isfinite(gradient).all():
                        raise ValueError('Nonfinite native reward gradient')
                    reward_probe = dict(reward=float(reward.detach()), gradient_norm=float(gradient.norm()), optimizer_updates=0)
                    del probe_bank, probe_head, reward, gradient
                inverse, cross = method.proximal_base(old, R, len(seen))
                losses = []
                status(status='RUNNING', phase='train', task=task, epoch=epoch)
                for x, labels, indices in training:
                    budget()
                    if steps >= config['max_steps']:
                        raise RuntimeError('INCOMPLETE_STEP_BUDGET')
                    x, labels, indices = x.cuda(), labels.cuda(), indices.cuda()
                    optimizer.zero_grad(set_to_none=True)
                    z = encoder(x)
                    with torch.no_grad():
                        reference_z = teacher(x)
                        target = F.one_hot(lookup[labels], len(seen)).double()
                        alpha = len(fit_ids)*weights[indices]/(len(seen)*len(x))
                        previous = W
                        W = method.proximal_head(z.detach().double(), target, alpha, inverse, cross, previous)
                    fit = .5*(alpha*(z.double() @ W-target).square().sum(1)).sum()
                    old_loss = .5*method.old_square_losses(old, W).sum()/len(seen)
                    ridge = .0005*(W*(R @ W)).sum()
                    proximal = .005*(W-previous).square().sum()
                    fd = (z-reference_z).square().sum(1).mean()
                    loss = fit+old_loss+ridge+proximal+10.*fd
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite coherent training objective')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
                    if config.get('preflight', False):
                        save(output/'PREFLIGHT.json', dict(status='PASS', encoder_dim=encoder.dim,
                            feature_norm_min=float(z.norm(dim=1).min()), feature_norm_max=float(z.norm(dim=1).max()),
                            gradient_norm=float(norm), loss=float(loss.detach()), optimizer_updates=0,
                            finite_head=bool(torch.isfinite(W).all()), batch_n=len(x), reward_probe=reward_probe,
                            activation=activation, controller=audit))
                        status(status='COMPLETE', preflight=True, optimizer_updates=0)
                        return
                    optimizer.step(); steps += 1
                    losses.append([float(fit.detach()), float(fd.detach()), float(norm)])
                scheduler.step()
                current, after_y = features(encoder, canonical)
                if not torch.equal(raw_y, after_y):
                    raise ValueError('Canonical order changed')
                diagnostics.append(dict(task=task, epoch=epoch, steps=steps, arm=arm,
                    fit_n=len(fit_ids), meta_n=len(meta_ids), controller=audit, activation=activation,
                    prototype_count=len(candidate['components']),
                    shift_norm=float(common_shift(before[fit_ids], current[fit_ids], y[fit_ids]).norm()),
                    mean_fit_loss=float(np.mean(losses, axis=0)[0]),
                    mean_feature_loss=float(np.mean(losses, axis=0)[1])))
                save(output/'diagnostics.json', diagnostics)
            # Store only class moments/prototypes. All current training images contribute to final refit.
            old = method.translate(bank, common_shift(before, current, y))
            weights = method.sample_weights(y, group, difficulty, actions)
            bank = method.append(old, current, y, group, weights)
            W, _ = method.head(bank, strength)
            adapter = {k: p.detach().cpu() for k, p in encoder.named_parameters() if p.requires_grad}
            torch.save(dict(adapter=adapter, head=W.float().cpu(), bank=cpu_bank(bank), seen=seen.copy(),
                actor=actor.detach().cpu(), steps=steps), output/f'stage_{task}.pt')
            del teacher, inverse, cross, R, candidate, old
        # Validation is opened only after all policy and encoder updates are complete.
        reports = []
        ranked = sorted(config['order'], key=lambda c: (-sum(r['label'] == c for r in train), c))
        tail = set(ranked[len(ranked)//2:])
        for task in range(1, len(tasks)+1):
            status(status='RUNNING', phase='evaluate', task=task)
            state = torch.load(output/f'stage_{task}.pt', map_location='cpu', weights_only=False)
            encoder.load_state_dict(state['adapter'], strict=False); seen = state['seen']
            x, labels = extract(encoder, loader([r for r in val if r['label'] in seen], False, task), budget)
            pred = np.asarray(seen)[np.argmax(x @ state['head'].numpy(), axis=1)]
            recalls = {str(c): float((pred[labels == c] == c).mean()) for c in seen}
            reports.append(dict(task=task, seen=seen, validation_n=len(labels), accuracy=float((pred == labels).mean()),
                balanced_accuracy=float(np.mean(list(recalls.values()))),
                tail_recall=float(np.mean([recalls[str(c)] for c in seen if c in tail])) if tail.intersection(seen) else None,
                per_class_recall=recalls, per_class_n={str(c): int((labels == c).sum()) for c in seen}))
        forgetting = []
        for c in config['order']:
            previous = [r['per_class_recall'][str(c)] for r in reports[:-1] if str(c) in r['per_class_recall']]
            if previous:
                forgetting.append(max(previous)-reports[-1]['per_class_recall'][str(c)])
        save(output/'metrics.json', dict(stages=reports,
            **benchmark_metrics(reports, tasks, {c:sum(r['label']==c for r in train) for c in config['order']}),
            average_incremental_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reports])),
            final_balanced_accuracy=reports[-1]['balanced_accuracy'], final_tail_recall=reports[-1]['tail_recall'],
            forgetting=float(np.mean(forgetting)), forgetting_classes=len(forgetting),
            test_accessed=config.get('evaluation_split') == 'official_test',
            evaluation_split=config.get('evaluation_split', 'development_validation'), independent_confirmation=False))
        status(status='COMPLETE', stages=len(tasks), peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
