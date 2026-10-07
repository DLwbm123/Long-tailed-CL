"""PCRL1 trajectories: bounded counterfactual branches on the CORE1 training path."""
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
import pcrl_control as control
from collections import deque
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
    if arm not in ('R', 'PC', 'RL', 'GREEDY', 'RL_uniform', 'RL_mean'):
        raise ValueError('Unknown coherent arm')
    output = Path(config['output'])
    if output.exists():
        raise ValueError('Fresh output required; no overwrite or automatic retry')
    train, val = manifests(config)
    output.mkdir(parents=True)
    save(output/'INPUT.private.json', config)
    started, steps = time.monotonic(), 0
    diagnostic_seconds = 0.
    rollout_updates = 0
    policy = control.Policy(config['seed'])
    selected_action = 0
    adaptive = arm in ('RL', 'GREEDY', 'RL_uniform', 'RL_mean')
    initial_steps = 0
    def budget():
        if time.monotonic()-started >= config['max_wall_seconds']:
            raise RuntimeError('INCOMPLETE_WALL_BUDGET')
    def status(**kw):
        save(output/'STATUS.json', dict(steps=steps, actual_updates=steps-initial_steps+rollout_updates, retained_updates=steps-initial_steps, rollout_updates=rollout_updates, policy_updates=policy.updates, diagnostic_gpu_seconds=diagnostic_seconds, elapsed_seconds=time.monotonic()-started, **kw))
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
        mean_only = arm == 'RL_mean'
        historical_only = False
        beta = 0. if arm == 'R' else .5
        bank = method.empty(encoder.dim, 'cuda'); seen = []; diagnostics = []; all_controller_audits = []
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
            selected_action = 0
            if adaptive and task > 1:
                def subset(ids, offset):
                    chosen=[]
                    for c in sorted(y[ids].unique().tolist()):
                        positions=ids[y[ids]==c].tolist()
                        random.Random(seed+task*100003+c*2003+offset).shuffle(positions)
                        chosen.extend(positions[:64])
                    return torch.as_tensor(chosen, device='cuda')
                probe_fit = subset(fit_ids, 11); probe_meta = subset(meta_ids, 29)
                probe_ids = torch.cat([probe_fit, probe_meta])
                probe_loader = loader([rows[i] for i in probe_ids.tolist()], False, task)
                probe_y = y[probe_ids]
                tail_labels = set(sorted(config['order'], key=lambda c:(-sum(r['label']==c for r in train),c))[4:])
                tail_indices = [i for i,c in enumerate(seen) if c in tail_labels]
                @torch.no_grad()
                def probe(head):
                    observed, labels_probe = features(encoder, probe_loader)
                    if not torch.equal(lookup[labels_probe], probe_y):
                        raise ValueError('Probe order mismatch')
                    delta = common_shift(before[probe_fit], observed[:len(probe_fit)], y[probe_fit])
                    translated = method.translate(bank, delta)
                    value = control.risks(translated, head, observed[len(probe_fit):], y[probe_meta])
                    return value, float(delta.norm())
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
                def update(batch, head, pair_a, apply_update, diagnose=False):
                    nonlocal diagnostic_seconds, rollout_updates
                    budget()
                    x, labels, indices = [v.cuda() for v in batch]
                    optimizer.zero_grad(set_to_none=True)
                    z = encoder(x)
                    with torch.no_grad():
                        reference_z = teacher(x)
                        target = F.one_hot(lookup[labels], len(seen)).double()
                        alpha = len(fit_ids)*weights[indices]/(len(seen)*len(x))
                        previous = head
                        head = method.proximal_head(z.detach().double(), target, alpha, inverse, cross, previous)
                        solve_audit = dict(iterations=0, relative_residual=0.)
                        if block_beta:
                            q, mu, mass = competition.moments(old, z.detach().double(), lookup[labels], alpha, len(seen))
                            head, solve_audit = competition.solve(q, mu, mass, pair_a, block_beta, mean_only,
                                previous, .01, inverse, z.detach().double(), alpha, head)
                    fit = .5*(alpha*(z.double() @ head-target).square().sum(1)).sum()
                    pair = block_beta*competition.pair_loss(old, z.double(), lookup[labels], alpha, head,
                        pair_a, mean_only) if block_beta else fit.new_zeros(())
                    old_loss = .5*method.old_square_losses(old, head).sum()/len(seen)
                    ridge = .0005*(head*(R @ head)).sum()
                    proximal = .005*(head-previous).square().sum()
                    fd = support.fd_loss((z-reference_z).square().sum(1), labels, {}, False)
                    check = None
                    if diagnose:
                        check, cost = support.gradient_diagnostic(fit+pair, fd,
                            [p for p in encoder.parameters() if p.requires_grad])
                        diagnostic_seconds += cost
                    loss = fit+pair+old_loss+ridge+proximal+10.*fd
                    if not torch.isfinite(loss): raise ValueError('Nonfinite coherent training objective')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
                    if apply_update:
                        if apply_update == 'rollout' and rollout_updates >= 150:
                            raise RuntimeError('ROLLOUT_UPDATE_LIMIT')
                        optimizer.step()
                        if apply_update == 'rollout': rollout_updates += 1
                    return head, dict(fit=float(fit.detach()), fd=float(fd.detach()), norm=float(norm),
                        pair=float(pair.detach()), loss=float(loss.detach()), solve=solve_audit, gradient_check=check)

                base_pairs = pair_weights.clone()
                controller_audits = []
                iterator = iter(support.batches(training, 'R', task, epoch)); pending = deque()
                block_n = support.block_steps('R', task, len(fit_ids))
                for batch_index in range(block_n):
                    if steps >= config['max_steps']: raise RuntimeError('INCOMPLETE_STEP_BUDGET')
                    if not pending: pending.append(next(iterator))
                    decision = adaptive and task > 1 and batch_index % 16 == 0
                    if config.get('preflight', False):
                        if not decision: raise ValueError('Preflight requires an adaptive Task 2 context')
                        snap = control.Snapshot(encoder, optimizer)
                        reference_w, native_check = update(pending[0], W.clone(), base_pairs, False)
                        base_risk, shift = probe(reference_w)
                        snap.restore(optimizer); snap.verify(optimizer)
                        changed = control.allocation(base_pairs, len(bank['n']), 1)
                        changed_w, changed_check = update(pending[0], W.clone(), changed, False)
                        changed_risk, _ = probe(changed_w)
                        if (reference_w-changed_w).norm() <= 1e-12 or changed_check['norm'] <= 0:
                            raise ValueError('Preflight action or adapter gradient is inactive')
                        snap.restore(optimizer); snap.verify(optimizer)
                        replay_w, replay_check = update(pending[0], W.clone(), base_pairs, False)
                        if not torch.equal(reference_w, replay_w) or native_check['loss'] != replay_check['loss']:
                            raise ValueError('Real zero-update branch replay differs')
                        snap.restore(optimizer); snap.verify(optimizer)
                        save(output/'PREFLIGHT.json', dict(status='PASS', optimizer_updates=0,
                            native=native_check, changed=changed_check, risk_reference=base_risk.tolist(),
                            risk_changed=changed_risk.tolist(), reward=control.reward(changed_risk,base_risk,len(bank['n']),tail_indices),
                            head_difference=float((reference_w-changed_w).norm()), action_mass=changed.sum().item(),
                            restore_verified=True, identical_replay=True, branch_horizon_updates=0,
                            probe_fit_n=len(probe_fit), probe_meta_n=len(probe_meta), peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                        status(status='COMPLETE', preflight=True, optimizer_updates=0)
                        return
                    if decision:
                        horizon = min(2, block_n-batch_index)
                        while len(pending)<horizon: pending.append(next(iterator))
                        cached = list(pending)[:horizon]
                        snap = control.Snapshot(encoder, optimizer)
                        old_count = len(bank['n'])
                        entry_risk, shift = probe(W)
                        feat = control.state(candidate, W, base_pairs, entry_risk, old_count,
                            tail_indices, batch_index/max(1,block_n-1), shift)
                        valid = torch.ones(7, dtype=torch.bool)
                        mass = (control.masks(len(seen),old_count,base_pairs.device)*base_pairs).sum((1,2))
                        for g in range(3):
                            if mass[g]<=0: valid[1+2*g:3+2*g]=False
                        chosen, prior = policy.propose(feat, valid, greedy=arm=='GREEDY')
                        branch_results=[]; branch_solve_max=0.; branch_started=time.monotonic()
                        for action in [0]+chosen.tolist():
                            snap.restore(optimizer); snap.verify(optimizer)
                            branch_w = W.clone()
                            allocation = control.allocation(base_pairs,old_count,action)
                            for batch in cached:
                                branch_w, details = update(batch,branch_w,allocation,'rollout')
                                branch_solve_max=max(branch_solve_max,details['solve']['relative_residual'])
                            value, _ = probe(branch_w)
                            branch_results.append(value)
                            status(status='RUNNING',phase='controller_rollouts',task=task,epoch=epoch,batch_index=batch_index)
                        snap.restore(optimizer); snap.verify(optimizer)
                        reference = branch_results[0]
                        values=[control.reward(v,reference,old_count,tail_indices) for v in branch_results[1:]]
                        if arm=='GREEDY':
                            best=max(range(5),key=lambda i:([0.]+values)[i])
                            selected_action=([0]+chosen.tolist())[best]
                            policy_steps=0
                        else:
                            policy_steps=policy.update(feat,valid,chosen,values,prior)
                            selected_action=int(policy.probabilities(feat,valid).argmax())
                        pair_weights=control.allocation(base_pairs,old_count,selected_action)
                        controller_audits.append(dict(batch_index=batch_index, horizon=horizon,
                            proposed_actions=chosen.tolist(), rewards=values, risks=[v.tolist() for v in branch_results],
                            selected_action=selected_action, before_probabilities=prior.tolist(),
                            after_probabilities=policy.probabilities(feat,valid).detach().tolist(),
                            policy_updates=policy_steps, state=feat.tolist(), restored=True,
                            rollout_updates=5*horizon, seconds=time.monotonic()-branch_started,
                            max_residual=branch_solve_max,
                            base_group_mass=mass.tolist(), selected_group_mass=(control.masks(len(seen),old_count,base_pairs.device)*pair_weights).sum((1,2)).tolist(),
                            probe_fit_n=len(probe_fit),probe_meta_n=len(probe_meta)))
                        save(output/'CONTROLLER.json',dict(decisions=all_controller_audits+controller_audits,
                            rollout_updates=rollout_updates,policy_updates=policy.updates))
                    batch = pending.popleft()
                    W, detail = update(batch,W,pair_weights,'retained',task>1 and batch_index in (0,block_n-1))
                    steps += 1
                    indices=batch[2]
                    sample_counts += np.bincount(indices.numpy(), minlength=len(fit_ids));batch_sizes.append(len(indices))
                    pair_losses.append(detail['pair']);pair_audit['solves'].append(detail['solve'])
                    if detail['gradient_check'] is not None:
                        gradient_checks.append(dict(batch_index=batch_index,**detail['gradient_check']))
                    losses.append([detail['fit'],detail['fd'],detail['norm']])
                all_controller_audits.extend(controller_audits)
                audit = dict(mode=arm,decisions=controller_audits,optimizer_steps=sum(v['policy_updates'] for v in controller_audits),
                    selected_action=selected_action,rollout_updates=sum(v['rollout_updates'] for v in controller_audits))
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
                if adaptive: a = control.allocation(a,len(seen)-len(classes),selected_action)
                W, solve_audit = competition.bank_head(bank, a, beta, mean_only)
                boundary_competition = dict(weights=a.tolist(), solve=solve_audit, beta=beta, class_order=seen.copy())
            adapter = {k: p.detach().cpu() for k, p in encoder.named_parameters() if p.requires_grad}
            model = {k:v.detach().cpu() for k,v in encoder.state_dict().items()}
            encoder_digest=support.state_digest(model)
            if frozen_digest is not None and encoder_digest!=frozen_digest:
                raise ValueError('Frozen encoder parameters or buffers changed')
            torch.save(dict(adapter=adapter, model=model, head=W.float().cpu(), bank=cpu_bank(bank), seen=seen.copy(),
                actor=policy.theta.detach().cpu(), controller_action=selected_action, rollout_updates=rollout_updates, steps=steps, configuration=config, rng=support.rng_state(), diagnostics=diagnostics,
                prefix_encoder_sha256=encoder_digest), output/f'stage_{task}.pt')
            save(output/f'boundary_{task}.json',dict(encoder_sha256=encoder_digest, frozen_verified=arm=='F' and task>1,
                class_losses=support.class_losses(current,y,W,fit_ids,meta_ids),logical_updates=steps,competition=boundary_competition))
            del teacher
            if config.get('stop_after_task') == task:
                save(output/'PREFIX_FILE_SHA.json',dict(sha256=support.digest_file(output/f'stage_{task}.pt')))
                status(status='TRAINED', stages=task, peak_gpu_bytes=torch.cuda.max_memory_allocated());return
            del before,current,raw_y,y,seeds,group,difficulty,training,canonical,fit_ids,meta_ids
            if adaptive and task>1: del probe_loader,probe_ids,probe_fit,probe_meta,probe_y
        status(status='TRAINED', stages=len(tasks), peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
