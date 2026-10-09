"""Train-side terminal decisions, locked before development evaluation."""
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import edge_competition as edge
import prototype_coherent as native
from decision_probability import probabilities, choose
from diagnose_holdout_feedback import rebuild, to_device
from diagnose_edge_learning import paired
from run_feedback_audit import metrics
from run_action_trace import actions, head_for
from run_prototype_coherent import common_shift, fit_split
from run_prototype_single import Images, extract, save
from run_multilabel import ApartFeatures


KEYS = ['edge_%02d'%i for i in range(17)]+['norm_%02d'%i for i in range(1,17)]


@torch.no_grad()
def run(config):
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=False)
    started, cpu = time.monotonic(), time.process_time(); completed = []; evaluated = []
    torch.set_num_threads(4); maximum_residual = 0.; image_rows = 0
    save(out/'INPUT.private.json', config)
    def budget():
        if time.monotonic()-started >= config['max_wall_seconds'] or time.time() >= config['original_deadline']:
            raise TimeoutError('Original deadline exhausted')
    def status(state, **extra):
        save(out/'STATUS.json', dict(status=state, completed_states=completed, evaluated_states=evaluated,
            elapsed_seconds=time.monotonic()-started, cpu_process_seconds=time.process_time()-cpu,
            adapter_updates=0, policy_updates=0, image_rows=image_rows, test_accessed=False, **extra))
    def load(path):
        budget(); return torch.load(path, map_location='cpu', weights_only=False)
    def manifest(original, split):
        with (Path(original['manifest'])/(split+'.csv')).open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(r['split'] == split for r in rows)
        for row in rows:
            p = Path(row['relative_path'])
            assert not p.is_absolute() and '..' not in p.parts and row['identity_component']
            row['label'] = int(row['original_label'])
        return rows
    def encoder(original):
        # Frozen routing keys are initialized, not included in adapter checkpoints.
        random.seed(original['seed']); np.random.seed(original['seed'])
        torch.manual_seed(original['seed']); torch.cuda.manual_seed_all(original['seed'])
        model = ApartFeatures(original['legacy_repo'], original['weight'], len(original['order']), 'cuda:0', original['seed'])
        from run_medical_v2 import transform
        model.eval().requires_grad_(False)
        return model, transform(False)
    def features(model, transform, original, rows, task):
        nonlocal image_rows
        loader = DataLoader(Images(rows, original['images'], transform, original['seed']+task*100003),
            batch_size=64, num_workers=4, multiprocessing_context='spawn',
            generator=torch.Generator().manual_seed(original['seed']+task*2003))
        z, labels = extract(model, loader, budget); image_rows += len(rows)
        assert np.array_equal(labels, np.asarray([r['label'] for r in rows]))
        return torch.as_tensor(z, device='cuda', dtype=torch.float64), labels
    def heads(bank, old_count):
        nonlocal maximum_residual
        values = []; kls = []
        for a in actions():
            budget(); w, _, _, base, pairs, residual = head_for(bank, a, old_count)
            values.append(w.float().double()); kls.append(float(edge.action_kl(pairs, base)))
            maximum_residual = max(maximum_residual, residual)
        norm = values[0].norm(dim=0).clamp_min(1e-12)
        values.extend((values[0]/norm.pow(i/16)).float().double() for i in range(1,17))
        return torch.stack(values), kls
    def scores(z, w):
        return np.asarray(z) @ w.float().cpu().numpy()
    try:
        status('RUNNING', phase='prepare'); records = {}
        for spec in config['states']:
            budget(); key = spec['name']; task = spec['task']; source = Path(spec['source'])
            original = json.loads((source/'INPUT.private.json').read_text())
            assert original['method'] == 'static_pc' and original['evaluation_split'] == 'development_validation'
            train = manifest(original, 'train'); seen = original['order'][:sum(original['task_sizes'][:task])]
            old_count = sum(original['task_sizes'][:task-1]); current = seen[old_count:]
            rows = [r for r in train if r['label'] in current]; fi, mi = fit_split(rows, original['split_seed'])
            assert not {rows[i]['identity_component'] for i in fi} & {rows[i]['identity_component'] for i in mi}
            assert not {rows[i]['identity_component'] for i in mi} & {r['identity_component'] for r in train if r['label'] in seen[:old_count]}
            ranked = sorted(original['order'], key=lambda c: (-sum(r['label']==c for r in train), c))
            tail = [i for i,c in enumerate(seen) if c in ranked[len(ranked)//2:]]
            previous = load(source/f'stage_{task-1}.pt'); old = to_device(previous['bank'], 'cuda')
            if spec.get('cached'):
                cached = Path(spec['cached']); initial = load(cached/'INITIAL.private.pt'); zero = load(cached/'candidate_00/TRACE.private.pt')
                assert initial['fit_indices'] == fi and initial['meta_indices'] == mi
                before = initial['features'].cuda(); after = zero['post_features'].cuda(); raw_y = zero['labels']
                adapter = zero['adapter']; saved_all = zero['all_refit']; cohort = 'previous_one_epoch_task2'
                del initial, zero
            else:
                model, transform = encoder(original); model.load_state_dict(previous['adapter'], strict=False)
                status('RUNNING', phase='training_features_before', active_state=key)
                before, raw_y = features(model, transform, original, rows, task)
                state = load(source/f'stage_{task}.pt'); model.load_state_dict(state['adapter'], strict=False)
                status('RUNNING', phase='training_features_after', active_state=key)
                after, after_y = features(model, transform, original, rows, task)
                assert np.array_equal(raw_y, after_y); adapter = state['adapter']; saved_all = state['head']
                cohort = 'existing_full_two_epoch_static_trajectory'; del model, state
            assert np.array_equal(raw_y, np.asarray([r['label'] for r in rows]))
            y = torch.tensor([seen.index(int(c)) for c in raw_y], device='cuda')
            fit = torch.tensor(fi, device='cuda'); meta = torch.tensor(mi, device='cuda')
            bank, _ = rebuild(old, before, after, y, fit, False); fit_heads, kls = heads(bank, old_count)
            shifted = native.translate(old, common_shift(before[fit], after[fit], y[fit]))
            meta_scores = [scores(after[meta].cpu(), w) for w in fit_heads]
            meta_metrics = [metrics(s, raw_y[mi], seen, current) for s in meta_scores]
            meta_acc = torch.tensor([[m['per_class'][str(c)]['recall'] if c in current else 0. for c in seen] for m in meta_metrics], device='cuda', dtype=torch.float64)
            status('RUNNING', phase='gaussian_integration', active_state=key)
            estimate, covariance_audit = probabilities(bank, fit_heads, config['integration_seed']+spec['seed_offset'], budget)
            decision, audit = choose(estimate, meta_acc, old_count, tail, list(range(17)))
            norm_decision, norm_audit = choose(estimate, meta_acc, old_count, tail, [0]+list(range(17,33)))
            risks = [edge.risks(shifted, w, after[meta], y[meta]) for w in fit_heads[:17]]
            rewards = [float(edge.reward(v, risks[0], old_count, tail))-.01*kl for v,kl in zip(risks,kls)]
            reward_decision = min(range(17), key=lambda i: (-rewards[i], i))
            if rewards[reward_decision] <= 1e-8: reward_decision = 0
            selected = dict(zero='edge_00', original_reward=KEYS[reward_decision], decision=KEYS[decision], norm_decision=KEYS[norm_decision])
            record = dict(dataset=spec['dataset'], task=task, cohort=cohort, seen=seen, current=current, tail=[seen[i] for i in tail],
                fit_images=len(fi), meta_images=len(mi), current_support={str(c):dict(fit_images=sum(rows[i]['label']==c for i in fi),
                    fit_identities=len({rows[i]['identity_component'] for i in fi if rows[i]['label']==c}),
                    meta_images=sum(rows[i]['label']==c for i in mi), meta_identities=len({rows[i]['identity_component'] for i in mi if rows[i]['label']==c})) for c in current},
                selected=selected, original_rewards=dict(zip(KEYS[:17],rewards)), covariance_audit=covariance_audit,
                gaussian_batches=estimate.cpu().tolist(), gaussian_mean=estimate.mean(0).cpu().tolist(),
                meta={k:m for k,m in zip(KEYS,meta_metrics)}, decision_audit={KEYS[i]:v for i,v in audit.items()},
                norm_decision_audit={KEYS[i]:v for i,v in norm_audit.items()},
                current_meta_excluded_from_fit=True, current_meta_prediction_is_train_side_diagnostic=True,
                integration_error_is_not_statistical_confidence=True, independent_confirmation=False)
            target = out/key; target.mkdir()
            save(target/'SELECTION.json', record)
            # The optional all-refit changes only the head, never the locked selection.
            all_bank, _ = rebuild(old, before, after, y, fit, True); all_heads, _ = heads(all_bank, old_count)
            error = float((all_heads[0].float().cpu()-saved_all.float()).abs().max())
            assert torch.allclose(all_heads[0].float().cpu(), saved_all.float(), atol=1e-6, rtol=1e-5), f'Zero-head reconstruction error: {error}'
            record['zero_all_head_reconstruction_error'] = error
            torch.save(dict(fit_heads=fit_heads.cpu(), all_heads=all_heads.cpu(), adapter=adapter,
                            current_features=after.cpu(), labels=raw_y, fit_indices=fi, meta_indices=mi), target/'HEADS.private.pt')
            records[key] = record; completed.append(key)
            save(out/'SELECTIONS.json', records)
            del previous, old, bank, all_bank, before, after, shifted, fit_heads, all_heads, estimate, adapter
        save(out/'SELECT_COMPLETE.json', dict(states=completed, selected_at=time.time(), development_features_accessed=False,
             adapter_updates=0, policy_updates=0))
        status('RUNNING', phase='selection_barrier')
        while not all((Path(p)/'SELECT_COMPLETE.json').exists() for p in config['barrier_jobs']):
            budget()
            for p in config['barrier_jobs']:
                check = Path(p)/'STATUS.json'
                if check.exists() and json.loads(check.read_text())['status'] == 'INCOMPLETE':
                    raise RuntimeError('Another frozen selection failed; development remains closed')
            time.sleep(1)
        for spec in config['states']:
            key = spec['name']; task = spec['task']; record = records[key]
            status('RUNNING', phase='development_evaluation', active_state=key)
            selection = load(out/key/'HEADS.private.pt')
            if spec.get('cached'):
                dev = load(Path(spec['cached'])/'candidate_00/DEVELOPMENT.private.pt'); z, labels = dev['features'], dev['labels']
            else:
                original = json.loads((Path(spec['source'])/'INPUT.private.json').read_text())
                validation = [r for r in manifest(original, 'val') if r['label'] in record['seen']]
                assert not {r['identity_component'] for r in manifest(original,'train')} & {r['identity_component'] for r in validation}
                model, transform = encoder(original); model.load_state_dict(selection['adapter'], strict=False)
                z, labels = features(model, transform, original, validation, task); z = z.float().cpu().numpy(); del model
                torch.save(dict(features=z,labels=labels),out/key/'DEVELOPMENT.private.pt')
            record['development'] = {}
            for partition in ('fit','all'):
                values = [scores(z, w) for w in selection[partition+'_heads']]
                record['development'][partition] = {k:dict(metrics=metrics(s,labels,record['seen'],record['current']),
                    paired_to_zero=paired(values[0],s,labels,record['seen'])) for k,s in zip(KEYS,values)}
            record['development_after_all_selections'] = True; evaluated.append(key)
            save(out/'RESULTS.json', records); del selection
        status('COMPLETE', maximum_solve_residual=maximum_residual)
    except BaseException as exc:
        status('INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
