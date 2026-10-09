"""Matched fine/coarse edge-control trajectories; private config on stdin."""
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

import core1_competition as competition
import edge_competition as edge
import prototype_coherent as method
from lt_benchmark import benchmark_metrics, task_blocks
from run_multilabel import ApartFeatures
from run_prototype_coherent import IndexedImages, common_shift, cpu_bank, fit_split
from run_prototype_single import Images, extract, manifests, save


def run(config):
    output = Path(config['output'])
    if output.exists():
        raise ValueError('Fresh output required; preserve all previous attempts')
    controller = edge.Controller(config['method'], config['seed'])
    train, val = manifests(config)
    output.mkdir(parents=True)
    save(output/'INPUT.private.json', config)
    started, steps = time.monotonic(), 0
    max_residual = 0.

    def budget():
        if (time.monotonic()-started >= config['max_wall_seconds'] or
                time.time() >= config['original_deadline']):
            raise RuntimeError('INCOMPLETE_ORIGINAL_DEADLINE')

    def status(**values):
        save(output/'STATUS.json', dict(steps=steps, policy_updates=controller.updates,
            elapsed_seconds=time.monotonic()-started, max_solve_residual=max_residual, **values))

    try:
        seed = config['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(4)
        status(status='RUNNING', phase='initialize')
        encoder = ApartFeatures(config['legacy_repo'], config['weight'], len(config['order']), 'cuda:0', seed)
        from run_medical_v2 import transform

        def loader(rows, training, task):
            dataset = (IndexedImages if training else Images)(rows, config['images'], transform(training), seed+task*100003)
            workers = config['workers']
            options = dict(multiprocessing_context='spawn', prefetch_factor=2,
                           persistent_workers=not training) if workers else {}
            return DataLoader(dataset, batch_size=config['batch_size'], shuffle=training,
                num_workers=workers, generator=torch.Generator().manual_seed(seed+task*2003), **options)

        def features(model, data):
            x, y = extract(model, data, budget)
            return torch.as_tensor(x, dtype=torch.float64, device='cuda'), torch.as_tensor(y, device='cuda')

        tasks = task_blocks(config['order'], config['task_sizes'])
        ranked = sorted(config['order'], key=lambda c: (-sum(r['label'] == c for r in train), c))
        tail = set(ranked[len(ranked)//2:])
        bank = method.empty(encoder.dim, 'cuda')
        seen, diagnostics = [], []
        for task, classes in enumerate(tasks, 1):
            budget(); old_count = len(seen); seen += classes
            rows = [r for r in train if r['label'] in classes]
            fit_ids, meta_ids = fit_split(rows, config['split_seed'])
            fit_ids = torch.as_tensor(fit_ids, device='cuda')
            meta_ids = torch.as_tensor(meta_ids, device='cuda')
            canonical = loader(rows, False, task)
            training = loader([rows[i] for i in fit_ids.tolist()], True, task)
            teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
            status(status='RUNNING', phase='task_start_features', task=task)
            before, raw_y = features(teacher, canonical)
            lookup = torch.as_tensor([seen.index(c) if c in seen else -1 for c in range(len(config['order']))], device='cuda')
            y, current = lookup[raw_y], before
            seeds = method.seed_components(before[fit_ids], y[fit_ids])
            group, difficulty = method.memberships(before, y, seeds)
            zeros = before.new_zeros(len(seeds))
            weights = method.sample_weights(y[fit_ids], group[fit_ids], difficulty[fit_ids], zeros)
            tail_indices = [i for i, c in enumerate(seen) if c in tail]
            controller.begin_task()
            optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=config['lr'], weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=1e-5)
            for epoch in range(1, config['epochs']+1):
                budget(); encoder.eval(); training.dataset.epoch = epoch
                old = method.translate(bank, common_shift(before[fit_ids], current[fit_ids], y[fit_ids]))
                candidate = method.append(old, current[fit_ids], y[fit_ids], group[fit_ids], weights)
                native_head, metric = method.head(candidate, 1.)
                base_pairs = competition.competition(candidate, native_head)
                status(status='RUNNING', phase='controller', task=task, epoch=epoch)
                pairs, head, selected, audit = controller.choose(candidate, old, base_pairs, metric,
                    current[meta_ids], y[meta_ids], tail_indices, budget)
                max_residual = max(max_residual, audit['max_solve_residual'])
                inverse, cross = method.proximal_base(old, metric, len(seen))
                losses, solves = [], []
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
                        previous = head
                        native_head = method.proximal_head(z.detach().double(), target, alpha, inverse, cross, previous)
                        q, mu, mass = competition.moments(old, z.detach().double(), lookup[labels], alpha, len(seen))
                        head, solve = competition.solve(q, mu, mass, pairs, previous=previous,
                            proximal=.01, inverse=inverse, x=z.detach().double(), alpha=alpha,
                            native=native_head, regularizer=metric)
                        max_residual = max(max_residual, solve['relative_residual'])
                    fit = .5*(alpha*(z.double() @ head-target).square().sum(1)).sum()
                    pair = .5*competition.pair_loss(old, z.double(), lookup[labels], alpha, head, pairs)
                    old_loss = .5*method.old_square_losses(old, head).sum()/len(seen)
                    ridge = .0005*(head*(metric @ head)).sum()
                    proximal = .005*(head-previous).square().sum()
                    fd = (z-reference_z).square().sum(1).mean()
                    loss = fit+pair+old_loss+ridge+proximal+10.*fd
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite edge training objective')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
                    if config.get('preflight', False):
                        if not torch.isfinite(norm) or norm <= 0:
                            raise ValueError('Missing adapter gradient')
                        save(output/'PREFLIGHT.json', dict(status='PASS', encoder_dim=encoder.dim,
                            adapter_updates=0, policy_updates=controller.updates, task=task, epoch=epoch,
                            gradient_norm=float(norm), loss=float(loss.detach()), controller=audit,
                            batch_solve=solve, batch_n=len(x), peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                        status(status='COMPLETE', preflight=True, adapter_updates=0)
                        return
                    optimizer.step(); steps += 1
                    losses.append([float(fit.detach()), float(pair.detach()), float(fd.detach())])
                    solves.append(solve['relative_residual'])
                scheduler.step()
                current, after_y = features(encoder, canonical)
                if not torch.equal(raw_y, after_y):
                    raise ValueError('Canonical order changed')
                diagnostics.append(dict(task=task, epoch=epoch, steps=steps, arm=config['method'],
                    fit_n=len(fit_ids), meta_n=len(meta_ids), controller=audit,
                    max_batch_solve_residual=max(solves), prototype_count=len(candidate['components']),
                    mean_fit_loss=float(np.mean(losses, axis=0)[0]),
                    mean_pair_loss=float(np.mean(losses, axis=0)[1]),
                    mean_feature_loss=float(np.mean(losses, axis=0)[2])))
                save(output/'diagnostics.json', diagnostics)
            # Match the existing protocol: meta joins the task-end refit, never independent validation.
            old = method.translate(bank, common_shift(before, current, y))
            all_weights = method.sample_weights(y, group, difficulty, zeros)
            bank = method.append(old, current, y, group, all_weights)
            native_head, metric = method.head(bank, 1.)
            base_pairs = competition.competition(bank, native_head)
            fixed_head, _ = competition.bank_head(bank, base_pairs, regularizer=metric)
            state = edge.descriptors(bank, fixed_head, old_count)
            if config['method'] == 'coarse_rl':
                state = state[:, :, :3]
            pairs = edge.allocation(base_pairs, state, selected.to(base_pairs))
            head, solve = competition.bank_head(bank, pairs, regularizer=metric)
            max_residual = max(max_residual, solve['relative_residual'])
            diagnostics[-1]['boundary_competition'] = dict(selected_coefficients=selected.tolist(),
                base_pairs=base_pairs.tolist(), selected_pairs=pairs.tolist(), solve=solve,
                refit_uses_all_current_training=True, new_policy_updates=0)
            save(output/'diagnostics.json', diagnostics)
            adapter = {k: p.detach().cpu() for k, p in encoder.named_parameters() if p.requires_grad}
            torch.save(dict(adapter=adapter, head=head.float().cpu(), bank=cpu_bank(bank), seen=seen.copy(),
                actor=controller.parameter.detach(), selected_coefficients=selected,
                steps=steps, boundary_pair_weights=pairs.cpu()), output/f'stage_{task}.pt')
            del teacher, inverse, cross, candidate, old
        if steps != config['expected_steps']:
            raise ValueError('Training updates differ from frozen matched baseline')
        status(status='RUNNING', phase='training_complete')
        reports = []
        for task in range(1, len(tasks)+1):
            budget(); status(status='RUNNING', phase='evaluate', task=task)
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
            prior = [r['per_class_recall'][str(c)] for r in reports[:-1] if str(c) in r['per_class_recall']]
            if prior:
                forgetting.append(max(prior)-reports[-1]['per_class_recall'][str(c)])
        save(output/'metrics.json', dict(stages=reports,
            **benchmark_metrics(reports, tasks, {c: sum(r['label'] == c for r in train) for c in config['order']}),
            average_incremental_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reports])),
            final_balanced_accuracy=reports[-1]['balanced_accuracy'], final_tail_recall=reports[-1]['tail_recall'],
            forgetting=float(np.mean(forgetting)), forgetting_classes=len(forgetting),
            test_accessed=config['evaluation_split'] == 'official_test', evaluation_split=config['evaluation_split'],
            independent_confirmation=False))
        status(status='COMPLETE', stages=len(tasks), peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
