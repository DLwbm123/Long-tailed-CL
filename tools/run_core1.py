"""CORE1 competition trajectories on the unchanged NEXT1 training path."""
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
import core1_competition as competition
import next1_support as support
from run_multilabel import ApartFeatures
from run_prototype_single import Images, extract, manifests, save


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
    if arm not in ('R', 'PC', 'PC_uniform', 'PC_mean', 'PC_history', 'PC_history_uniform', 'PC_history_mean'):
        raise ValueError('Unknown coherent arm')
    output = Path(config['output'])
    if output.exists():
        raise ValueError('Fresh output required; no overwrite or automatic retry')
    train, val = manifests(config)
    output.mkdir(parents=True)
    save(output/'INPUT.private.json', config)
    started, steps = time.monotonic(), 0
    diagnostic_seconds = 0.
    initial_steps = 0
    def budget():
        if time.monotonic()-started >= config['max_wall_seconds']:
            raise RuntimeError('INCOMPLETE_WALL_BUDGET')
    def status(**kw):
        save(output/'STATUS.json', dict(steps=steps, actual_updates=steps-initial_steps, diagnostic_gpu_seconds=diagnostic_seconds, elapsed_seconds=time.monotonic()-started, **kw))
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
        strength = 0.
        mean_only = arm in ('PC_mean', 'PC_history_mean')
        historical_only = arm.startswith('PC_history')
        beta = 0. if arm == 'R' else .5
        bank = method.empty(encoder.dim, 'cuda'); seen = []; diagnostics = []
        actor = torch.zeros(6, dtype=torch.float64, device='cuda', requires_grad=True)
        generator = torch.Generator(device='cuda').manual_seed(seed+911)
        tasks = [config['order'][i:i+config['increment']] for i in range(0, len(config['order']), config['increment'])]
        start_task = 1; W = None
        if config.get('prefix'):
            prefix = Path(config['prefix'])
            state = torch.load(prefix, map_location='cpu', weights_only=False)
            if state['seen'] != config['order'][:2] or state['steps'] != 276:
                raise ValueError('Wrong shared first-task boundary')
            encoder.load_state_dict(state['model'], strict=True)
            bank = dict(mu=state['bank']['mu'].cuda(), Q=state['bank']['Q'].cuda(), n=state['bank']['n'],
                components=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in c.items()} for c in state['bank']['components']])
            W=state['head'].double().cuda(); seen=state['seen']; steps=initial_steps=state['steps']
            diagnostics=state['diagnostics'].copy();support.restore_rng(state['rng']);start_task=2
            (output/'stage_1.pt').symlink_to(prefix)
            save(output/'PREFIX.json', dict(prefix_sha256=state['prefix_encoder_sha256'], logical_updates=276, actual_updates=0, independent_repeat=False))
        if config.get('preflight') and start_task != 2:
            raise ValueError('CORE1 preflight requires a real first-task prefix')
        frozen_digest = support.state_digest(encoder.state_dict()) if arm=='F' and start_task==2 else None
        for task, classes in enumerate(tasks, 1):
            if task < start_task: continue
            old_head = None if W is None else W.detach().clone()
            trainable = arm!='F' or task==1
            budget(); seen += classes
            rows = [r for r in train if r['label'] in classes]
            fit_ids, meta_ids = support.fixed_split(rows, config)
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
            for epoch in (range(1, config['epochs']+1) if trainable else []):
                budget(); encoder.eval(); training.dataset.epoch = epoch
                old = method.translate(bank, common_shift(before[fit_ids], current[fit_ids], y[fit_ids]))
                status(status='RUNNING', phase='controller', task=task, epoch=epoch)
                audit = dict(mode='uniform', optimizer_steps=0, accepted=True,
                             selected_actions=actions.tolist())
                weights = method.sample_weights(y[fit_ids], group[fit_ids], difficulty[fit_ids], actions)
                candidate = method.append(old, current[fit_ids], y[fit_ids], group[fit_ids], weights)
                W, R = method.head(candidate, strength)
                block_beta = beta if task > 1 else 0.
                pair_weights = competition.competition(candidate, W, 'uniform' if arm.endswith('_uniform') else 'prototype',
                    historical_count=len(bank['n']) if historical_only else None)
                if block_beta:
                    W, entry_solve = competition.bank_head(candidate, pair_weights, block_beta, mean_only)
                else:
                    entry_solve = dict(iterations=0, relative_residual=0.)
                pair_audit = dict(beta=block_beta, mode=arm, class_order=seen.copy(),
                    weights=pair_weights.tolist(), entry_solve=entry_solve, solves=[], mean_pair_loss=0.,
                    scope='historical_only' if historical_only else 'all_classes', historical_count=len(bank['n']))
                pair_losses = []
                reward_probe = None
                inverse, cross = method.proximal_base(old, R, len(seen))
                losses = []; sample_counts = np.zeros(len(fit_ids), dtype=np.int64); batch_sizes = []; gradient_checks = []
                block_lr = optimizer.param_groups[0]['lr']
                status(status='RUNNING', phase='train', task=task, epoch=epoch)
                for batch_index, (x, labels, indices) in enumerate(support.batches(training, 'R', task, epoch)):
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
                        if block_beta:
                            q, mu, mass = competition.moments(old, z.detach().double(), lookup[labels], alpha, len(seen))
                            W, solve_audit = competition.solve(q, mu, mass, pair_weights, block_beta, mean_only,
                                previous, .01, inverse, z.detach().double(), alpha, W)
                            pair_audit['solves'].append(solve_audit)
                            del q, mu, mass
                    fit = .5*(alpha*(z.double() @ W-target).square().sum(1)).sum()
                    pair = (block_beta*competition.pair_loss(old, z.double(), lookup[labels], alpha, W,
                        pair_weights, mean_only)) if block_beta else fit.new_zeros(())
                    pair_losses.append(float(pair.detach()))
                    old_loss = .5*method.old_square_losses(old, W).sum()/len(seen)
                    ridge = .0005*(W*(R @ W)).sum()
                    proximal = .005*(W-previous).square().sum()
                    fd = support.fd_loss((z-reference_z).square().sum(1), labels,
                        {c:sum(r['label']==c for r in training.dataset.rows) for c in classes},
                        arm in ('CB','UCB') and task>1)
                    sample_counts += np.bincount(indices.cpu().numpy(), minlength=len(fit_ids));batch_sizes.append(len(x))
                    if task>1 and batch_index in (0,support.block_steps('R',task,len(fit_ids))-1):
                        check, cost = support.gradient_diagnostic(fit+pair, fd, [p for p in encoder.parameters() if p.requires_grad])
                        diagnostic_seconds += cost
                        gradient_checks.append(dict(batch_index=batch_index, class_counts={str(c):int((labels==c).sum()) for c in classes}, **check))
                    loss = fit+pair+old_loss+ridge+proximal+10.*fd
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite coherent training objective')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
                    if config.get('preflight', False):
                        save(output/'PREFLIGHT.json', dict(status='PASS', encoder_dim=encoder.dim,
                            feature_norm_min=float(z.norm(dim=1).min()), feature_norm_max=float(z.norm(dim=1).max()),
                            gradient_norm=float(norm), loss=float(loss.detach()), optimizer_updates=0,
                            finite_head=bool(torch.isfinite(W).all()), batch_n=len(x), reward_probe=reward_probe,
                            competition=pair_audit, pair_loss=float(pair.detach()),
                            native_fd=float(support.fd_loss((z-reference_z).square().sum(1), labels, {}, False)),
                            balanced_fd=float(support.fd_loss((z-reference_z).square().sum(1), labels, {c:sum(r['label']==c for r in training.dataset.rows) for c in classes}, True))))
                        status(status='COMPLETE', preflight=True, optimizer_updates=0)
                        return
                    optimizer.step(); steps += 1
                    losses.append([float(fit.detach()), float(fd.detach()), float(norm)])
                scheduler.step()
                current, after_y = features(encoder, canonical)
                if not torch.equal(raw_y, after_y):
                    raise ValueError('Canonical order changed')
                moment_check, cost = support.moment_diagnostic(before,current,y,fit_ids,meta_ids,old_head)
                diagnostic_seconds += cost
                exposure = {}
                for c in classes:
                    ids=[i for i,r in enumerate(training.dataset.rows) if r['label']==c]
                    visited=[i for i in ids if sample_counts[i]>0]
                    exposure[str(c)]=dict(sampled_images=int(sample_counts[ids].sum()), unique_images=len(visited),
                        unique_identity_groups=len({training.dataset.rows[i]['identity_component'] for i in visited}),
                        repeat_histogram={str(int(n)):int((sample_counts[ids]==n).sum()) for n in np.unique(sample_counts[ids])})
                pair_audit['mean_pair_loss'] = float(np.mean(pair_losses))
                diagnostics.append(dict(task=task, epoch=epoch, steps=steps, arm=arm, competition=pair_audit,
                    fit_n=len(fit_ids), meta_n=len(meta_ids), controller=audit, learning_rate=block_lr,
                    actual_block_updates=len(batch_sizes), batch_sizes=batch_sizes, exposure=exposure, gradient_checks=gradient_checks,
                    moment_check=moment_check, class_losses=support.class_losses(current,y,W,fit_ids,meta_ids),
                    prototype_count=len(candidate['components']),
                    shift_norm=float(common_shift(before[fit_ids], current[fit_ids], y[fit_ids]).norm()),
                    mean_fit_loss=float(np.mean(losses, axis=0)[0]),
                    mean_feature_loss=float(np.mean(losses, axis=0)[1])))
                save(output/'diagnostics.json', diagnostics)
            if not trainable:
                moment_check, cost = support.moment_diagnostic(before,current,y,fit_ids,meta_ids,old_head)
                diagnostic_seconds += cost
                for epoch in (1,2):
                    diagnostics.append(dict(task=task,epoch=epoch,arm=arm,steps=steps,
                        actual_block_updates=0,fit_n=len(fit_ids),meta_n=len(meta_ids),
                        gradient_checks=[],gradient_diagnostic_reason='NA: encoder frozen',
                        exposure={str(c):dict(sampled_images=0,unique_images=0,unique_identity_groups=0,repeat_histogram={}) for c in classes},
                        batch_sizes=[],moment_check=moment_check,shift_norm=0.))
                save(output/'diagnostics.json', diagnostics)
            # Store only class moments/prototypes. All current training images contribute to final refit.
            old = method.translate(bank, common_shift(before, current, y))
            weights = method.sample_weights(y, group, difficulty, actions)
            bank = method.append(old, current, y, group, weights)
            W, _ = method.head(bank, strength)
            boundary_competition = None
            if beta and task > 1:
                a = competition.competition(bank, W, 'uniform' if arm.endswith('_uniform') else 'prototype',
                    historical_count=len(seen)-len(classes) if historical_only else None)
                W, solve_audit = competition.bank_head(bank, a, beta, mean_only)
                boundary_competition = dict(weights=a.tolist(), solve=solve_audit, beta=beta, class_order=seen.copy())
            adapter = {k: p.detach().cpu() for k, p in encoder.named_parameters() if p.requires_grad}
            model = {k:v.detach().cpu() for k,v in encoder.state_dict().items()}
            encoder_digest=support.state_digest(model)
            if frozen_digest is not None and encoder_digest!=frozen_digest:
                raise ValueError('Frozen encoder parameters or buffers changed')
            torch.save(dict(adapter=adapter, model=model, head=W.float().cpu(), bank=cpu_bank(bank), seen=seen.copy(),
                actor=actor.detach().cpu(), steps=steps, configuration=config, rng=support.rng_state(), diagnostics=diagnostics,
                prefix_encoder_sha256=encoder_digest), output/f'stage_{task}.pt')
            save(output/f'boundary_{task}.json',dict(encoder_sha256=encoder_digest, frozen_verified=arm=='F' and task>1,
                class_losses=support.class_losses(current,y,W,fit_ids,meta_ids),logical_updates=steps,competition=boundary_competition))
            del teacher
            if config.get('stop_after_task') == task:
                save(output/'PREFIX_FILE_SHA.json',dict(sha256=support.digest_file(output/f'stage_{task}.pt')))
                status(status='TRAINED', stages=task, peak_gpu_bytes=torch.cuda.max_memory_allocated());return
            del before,current,raw_y,y,seeds,group,difficulty,training,canonical,fit_ids,meta_ids
            if trainable:
                del x,labels,indices,z,reference_z,target,fit,fd,loss,previous
        status(status='TRAINED', stages=len(tasks), peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
