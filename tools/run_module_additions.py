"""Matched single-module additions; current-data training, sealed development evaluation."""
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
import module_additions as modules
import prototype_coherent as method
from lt_benchmark import benchmark_metrics, task_blocks
from run_multilabel import ApartFeatures
from run_prototype_coherent import IndexedImages, common_shift, cpu_bank, fit_split
from run_prototype_single import Images, extract, manifests, save


def run(config):
    output = Path(config['output'])
    if output.exists():
        raise ValueError('Fresh output required; preserve all previous attempts')
    arm = config['module']
    if arm not in modules.ARMS: raise ValueError('Unknown module')
    if arm=='local_readout' and config.get('local_readout_weight')!=.25:
        raise ValueError('Frozen local readout amplitude differs')
    controller = edge.Controller('static_pc', config['seed'])
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
        updates = modules.Updates(encoder, arm)
        from run_medical_v2 import transform

        def loader(rows, training, task):
            dataset = (IndexedImages if training else Images)(rows, config['images'], transform(training), seed+task*100003)
            workers = config['workers']
            options = dict(multiprocessing_context='spawn', prefetch_factor=2,
                           persistent_workers=not training) if workers else {}
            return DataLoader(dataset, batch_size=config['batch_size'], shuffle=training,
                num_workers=workers, generator=torch.Generator().manual_seed(seed+task*2003), **options)

        def features(model, data):
            return modules.extract(model, data, budget, arm.startswith('local'))

        tasks = task_blocks(config['order'], config['task_sizes'])
        ranked = sorted(config['order'], key=lambda c: (-sum(r['label'] == c for r in train), c))
        tail = set(ranked[len(ranked)//2:])
        bank = method.empty(encoder.dim, 'cuda')
        seen, diagnostics = [], []
        local_bank = None
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
            before, raw_y, before_local = features(teacher, canonical)
            current_local = before_local
            updates.begin()
            lookup = torch.as_tensor([seen.index(c) if c in seen else -1 for c in range(len(config['order']))], device='cuda')
            y, current = lookup[raw_y], before
            seeds = method.seed_components(before[fit_ids], y[fit_ids])
            group, difficulty = method.memberships(before, y, seeds)
            zeros = before.new_zeros(len(seeds))
            weights = method.sample_weights(y[fit_ids], group[fit_ids], difficulty[fit_ids], zeros)
            tail_indices = [i for i, c in enumerate(seen) if c in tail]
            controller.begin_task()
            optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=config['lr']*updates.multiplier, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=1e-5*updates.multiplier)
            for epoch in range(1, config['epochs']+1):
                budget(); encoder.eval(); training.dataset.epoch = epoch
                shift = common_shift(before[fit_ids], current[fit_ids], y[fit_ids])
                old = method.translate(bank, shift)
                candidate = method.append(old, current[fit_ids], y[fit_ids], group[fit_ids], weights)
                local_shift = common_shift(before_local[fit_ids],current_local[fit_ids],y[fit_ids]) if arm in ('local_transport','local_readout') else shift
                local_candidate = modules.local_memory(local_bank, local_shift, current_local[fit_ids] if current_local is not None else None, y[fit_ids])
                native_head, metric, activation = modules.head(candidate, arm, config['name'], seen, local_candidate)
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
                        readout_check={}
                        if arm=='local_readout':
                            score=modules.local_scores(current_local[fit_ids[:len(x)]],local_candidate)
                            readout_check=dict(local_readout_shape_valid=score.shape==(len(x),len(seen)),
                                local_readout_max_abs=float(score.abs().max()),local_readout_weight=config['local_readout_weight'])
                            if not readout_check['local_readout_shape_valid'] or readout_check['local_readout_max_abs']>1+1e-6:
                                raise ValueError('Native local readout check failed')
                        save(output/'PREFLIGHT.json', dict(status='PASS', encoder_dim=encoder.dim, **readout_check,
                            adapter_updates=0, policy_updates=controller.updates, task=task, epoch=epoch,
                            gradient_norm=float(norm), loss=float(loss.detach()), controller=audit, activation=activation,
                            batch_solve=solve, batch_n=len(x), peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                        status(status='COMPLETE', preflight=True, adapter_updates=0)
                        return
                    optimizer.step(); steps += 1
                    losses.append([float(fit.detach()), float(pair.detach()), float(fd.detach())])
                    solves.append(solve['relative_residual'])
                scheduler.step()
                current, after_y, current_local = features(encoder, canonical)
                if not torch.equal(raw_y, after_y):
                    raise ValueError('Canonical order changed')
                diagnostics.append(dict(task=task, epoch=epoch, steps=steps, arm=config['method'],
                    fit_n=len(fit_ids), meta_n=len(meta_ids), controller=audit, activation=activation,
                    history_rank=updates.rank, learning_rate_multiplier=updates.multiplier,
                    local_transport_norm=float(local_shift.norm()) if arm in ('local_transport','local_readout') else 0.,
                    max_batch_solve_residual=max(solves), prototype_count=len(candidate['components']),
                    mean_fit_loss=float(np.mean(losses, axis=0)[0]),
                    mean_pair_loss=float(np.mean(losses, axis=0)[1]),
                    mean_feature_loss=float(np.mean(losses, axis=0)[2])))
                save(output/'diagnostics.json', diagnostics)
            if arm == 'fusion' and task > 1:
                status(status='RUNNING', phase='current_task_curvature', task=task)
                for images, labels in loader([rows[i] for i in fit_ids.tolist()], False, task):
                    budget()
                    z=encoder(images.cuda())
                    curvature_loss=F.cross_entropy(z @ head.float(),lookup[labels.cuda()])
                    updates.collect(curvature_loss)
                    del z,curvature_loss
            module_audit=updates.finish(task)
            if module_audit['fused']:
                current,after_y,current_local=features(encoder,canonical)
                if not torch.equal(raw_y,after_y):raise ValueError('Post-fusion canonical order changed')
            diagnostics[-1]['update_module']=module_audit
            # Match the existing protocol: meta joins the task-end refit, never independent validation.
            shift = common_shift(before, current, y)
            old = method.translate(bank, shift)
            local_shift=common_shift(before_local,current_local,y) if arm in ('local_transport','local_readout') else shift
            local_bank = modules.local_memory(local_bank, local_shift, current_local, y)
            if arm in ('local_transport','local_readout'):
                diagnostics[-1]['boundary_local_transport_norm']=float(local_shift.norm())
            all_weights = method.sample_weights(y, group, difficulty, zeros)
            bank = method.append(old, current, y, group, all_weights)
            native_head, metric, activation = modules.head(bank, arm, config['name'], seen, local_bank)
            base_pairs = competition.competition(bank, native_head)
            fixed_head, _ = competition.bank_head(bank, base_pairs, regularizer=metric)
            state = edge.descriptors(bank, fixed_head, old_count)
            
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
                steps=steps, boundary_pair_weights=pairs.cpu(),
                model={k:v.detach().cpu() for k,v in encoder.state_dict().items()},
                module=arm, module_memory=None if local_bank is None else local_bank.cpu()), output/f'stage_{task}.pt')
            del teacher, inverse, cross, candidate, old
        if steps != config['expected_steps']:
            raise ValueError('Training updates differ from frozen matched baseline')
        status(status='TRAINED', stages=len(tasks), peak_gpu_bytes=torch.cuda.max_memory_allocated())
        return
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


def evaluate(config):
    output=Path(config['output']);gate=json.loads(Path(config['evaluation_gate']).read_text())
    if gate['phase'] != 'EVALUATE':raise ValueError('Development evaluation sealed')
    started=time.monotonic()
    def budget():
        if time.time()>=config['original_deadline']:raise TimeoutError('Campaign deadline')
    seed=config['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
    encoder=ApartFeatures(config['legacy_repo'],config['weight'],len(config['order']),'cuda:0',seed)
    from run_medical_v2 import transform
    train,val=manifests(config);tasks=task_blocks(config['order'],config['task_sizes'])
    ranked=sorted(config['order'],key=lambda c:(-sum(r['label']==c for r in train),c));tail=set(ranked[len(ranked)//2:])
    reports=[]
    for task in range(1,len(tasks)+1):
        budget();state=torch.load(output/f'stage_{task}.pt',map_location='cpu',weights_only=False)
        encoder.load_state_dict(state['model'],strict=True);seen=state['seen'];encoder.eval()
        rows=[r for r in val if r['label'] in seen]
        loader=DataLoader(Images(rows,config['images'],transform(False),seed+task*100003),
            batch_size=config['batch_size'],num_workers=0,shuffle=False)
        if config['module']=='local_readout':
            x,labels,parts=modules.extract(encoder,loader,budget,True)
            global_score=x @ state['head'].to(x)
            local_score=modules.local_scores(parts,state['module_memory'].to(parts))
            scores=global_score+config['local_readout_weight']*local_score
            global_pred=np.asarray(seen)[global_score.argmax(1).cpu().numpy()]
            global_labels=labels.cpu().numpy()
            local_audit=dict(global_per_class_recall={str(c):float((global_pred[global_labels==c]==c).mean()) for c in seen},
                local_score_rms=float(local_score.square().mean().sqrt()),
                readout_weight=config['local_readout_weight'],
                readout_changed_predictions_n=int((scores.argmax(1)!=global_score.argmax(1)).sum()))
            pred=np.asarray(seen)[scores.argmax(1).cpu().numpy()];labels=labels.cpu().numpy()
        else:
            x,labels=extract(encoder,loader,budget);pred=np.asarray(seen)[(x@state['head'].numpy()).argmax(1)]
            local_audit={}
        recalls={str(c):float((pred[labels==c]==c).mean()) for c in seen}
        reports.append(dict(task=task,seen=seen,validation_n=len(labels),accuracy=float((pred==labels).mean()),**local_audit,
            balanced_accuracy=float(np.mean(list(recalls.values()))),per_class_recall=recalls,
            per_class_n={str(c):int((labels==c).sum()) for c in seen},
            tail_recall=float(np.mean([recalls[str(c)] for c in seen if c in tail])) if tail.intersection(seen) else None))
    old=config['order'][:-config['task_sizes'][-1]];new=config['order'][-config['task_sizes'][-1]:]
    forgetting=[max(r['per_class_recall'][str(c)] for r in reports[:-1] if c in r['seen'])-reports[-1]['per_class_recall'][str(c)] for c in old]
    metrics=dict(stages=reports,**benchmark_metrics(reports,tasks,{c:sum(r['label']==c for r in train) for c in config['order']}),
        average_incremental_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reports])),
        final_balanced_accuracy=reports[-1]['balanced_accuracy'],final_tail_recall=reports[-1]['tail_recall'],
        forgetting=float(np.mean(forgetting)),forgetting_classes=len(old),
        final_old_recall=float(np.mean([reports[-1]['per_class_recall'][str(c)] for c in old])),
        final_new_recall=float(np.mean([reports[-1]['per_class_recall'][str(c)] for c in new])),
        test_accessed=False,evaluation_split='development_validation',independent_confirmation=False)
    if config['module']=='base':
        historical=json.loads(Path(config['historical_metrics']).read_text())
        if any(a['per_class_recall']!=b['per_class_recall'] for a,b in zip(reports,historical['stages'])):
            raise ValueError('Matched static-PC historical baseline differs')
        metrics['historical_baseline_recalls_equal']=True
    save(output/'metrics.json',metrics)
    status=json.loads((output/'STATUS.json').read_text());status.update(status='COMPLETE',evaluation_seconds=time.monotonic()-started)
    save(output/'STATUS.json',status)


if __name__ == '__main__':
    config=json.load(sys.stdin)
    if config.get('operation')=='evaluate':evaluate(config)
    else:run(config)
