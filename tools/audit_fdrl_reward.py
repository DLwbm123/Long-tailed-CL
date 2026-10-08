"""Paired real-adapter calibration at the end of the already-arrived T1 task."""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import fdrl_control as ctl
import next1_support as support
import prototype_coherent as method
from pcrl_control import Snapshot
from run_multilabel import ApartFeatures
from run_pcrl import IndexedImages
from run_prototype_single import Images, extract, manifests, save


def empirical(x, y, head):
    scores = x @ head
    loss = .5 * (scores-F.one_hot(y, head.shape[1])).square().sum(1)
    prediction = scores.argmax(1)
    own = scores.gather(1, y[:, None]).flatten()
    other = scores.clone(); other.scatter_(1, y[:, None], -torch.inf)
    margin = own-other.max(1).values
    return dict(loss=[float(loss[y == c].mean()) for c in sorted(y.unique().tolist())],
        recall=[float((prediction[y == c] == c).double().mean()) for c in sorted(y.unique().tolist())],
        margin=[float(margin[y == c].mean()) for c in sorted(y.unique().tolist())])


def run(c):
    output = Path(c['output']); output.mkdir(parents=True, exist_ok=False)
    save(output/'INPUT.private.json', c)
    started = time.monotonic(); updates = 0; rows_out = []
    def budget():
        if time.monotonic()-started >= c['max_wall_seconds']:
            raise RuntimeError('CALIBRATION_WALL_BUDGET')
    def status(**kw):
        save(output/'STATUS.json', dict(actual_updates=updates,
            elapsed_seconds=time.monotonic()-started, **kw))
    try:
        seed = int(c['audit_seed']); random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.set_num_threads(2)
        train, _ = manifests(c)
        prefix = torch.load(c['prefix'], map_location='cpu', weights_only=False)
        if prefix['steps'] != 276 or prefix['seen'] != c['order'][:2] or not prefix.get('fit_only'):
            raise ValueError('A fit-only T1 prefix with separate meta moments is required')
        pair = c['order'][:2][::(-1 if c['reverse'] else 1)]
        seen = pair; rows = [r for r in train if r['label'] in pair]
        fit, meta = support.fixed_split(rows, c)
        if {rows[i]['identity_component'] for i in fit} & {rows[i]['identity_component'] for i in meta}:
            raise ValueError('Fit/meta identity overlap')
        # Pseudo-old images remain current-task data and are read only by the diagnostic oracle.
        rng = random.Random(seed); selected = []
        for group, cap in ((fit, 128), (meta, 64)):
            for label in pair:
                candidates = [i for i in group if rows[i]['label'] == label]
                rng.shuffle(candidates); selected.extend(candidates[:cap])
        selected = sorted(set(selected)); rows = [rows[i] for i in selected]
        fit, meta = support.fixed_split(rows, c)
        if any(sum(rows[i]['label'] == label for i in meta) < 8 for label in pair):
            raise ValueError('Insufficient pseudo-class meta support')
        encoder = ApartFeatures(c['legacy_repo'], c['weight'], 8, 'cuda:0', seed)
        encoder.load_state_dict(prefix['model'], strict=True); encoder.eval()
        from run_medical_v2 import transform
        def loader(items, training, salt):
            ds = (IndexedImages if training else Images)(items, c['images'], transform(training), seed+salt)
            return DataLoader(ds, batch_size=64, shuffle=training, num_workers=c.get('workers', 2),
                generator=torch.Generator().manual_seed(seed+salt),
                **(dict(multiprocessing_context='spawn', prefetch_factor=2) if c.get('workers', 2) else {}))
        canonical = loader(rows, False, 100)
        @torch.no_grad()
        def features():
            budget(); x, labels = extract(encoder, canonical, budget)
            return torch.as_tensor(x, dtype=torch.float64, device='cuda'), torch.tensor([seen.index(int(v)) for v in labels], device='cuda')
        status(status='RUNNING', phase='calibration_features')
        before, y = features(); fi = torch.tensor(fit, device='cuda'); mi = torch.tensor(meta, device='cuda')
        # The prefix encoder has seen both current T1 classes; these are calibration pseudo-tasks.
        bank = method.empty(encoder.dim,'cuda'); meta_bank = method.empty(encoder.dim,'cuda')
        params = [p for p in encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=c['lr'], weight_decay=.01)
        teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
        lookup = torch.tensor([seen.index(i) if i in seen else -1 for i in range(8)], device='cuda')
        def append(bank, x, labels):
            seeds = method.seed_components(x, labels); group, difficulty = method.memberships(x, labels, seeds)
            weights = method.sample_weights(labels, group, difficulty, x.new_zeros(len(seeds)))
            return method.append(bank, x, labels, group, weights)
        joint = append(bank, before[fi], y[fi]); head, metric = method.head(joint, 0.)
        inverse, cross = method.proximal_base(bank, metric, 2)
        def step(batch, head, action, inverse, cross, class_count):
            nonlocal updates
            budget(); x, labels, _ = [v.cuda() for v in batch]
            optimizer.zero_grad(set_to_none=True); z = encoder(x)
            with torch.no_grad():
                target = F.one_hot(lookup[labels], 2).double(); reference = teacher(x)
                weights = z.new_full((len(x),), class_count/(2*len(x)), dtype=torch.float64)
                head = method.proximal_head(z.detach().double(), target, weights, inverse, cross, head)
            loss = .5*(weights*(z.double()@head-target).square().sum(1)).sum()+ctl.ACTIONS[action]*(z-reference).square().sum(1).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite calibration loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
            optimizer.step(); updates += 1
            if updates > int(c['start_depth'])+48: raise RuntimeError('CALIBRATION_UPDATE_CAP')
            return head
        current_loader = loader([rows[i] for i in fit], True, 200)
        batches = list(current_loader)
        for i in range(int(c['start_depth'])):
            head = step(batches[i % len(batches)], head, 1, inverse, cross, 2)
        anchor, _ = features()
        shift = torch.stack([(anchor-before)[fi][y[fi] == k].mean(0) for k in (0, 1)]).mean(0)
        bank = method.translate(bank, shift); meta_bank = method.translate(meta_bank, shift)
        old_fit = fi[y[fi] == 0]; new_fit = fi[y[fi] == 1]; old_meta = mi[y[mi] == 0]
        bank = append(bank, anchor[old_fit], y[old_fit])
        meta_bank = ctl.meta_append(meta_bank, anchor[old_meta], y[old_meta])
        teacher.load_state_dict(encoder.state_dict(), strict=True)
        optimizer = torch.optim.AdamW(params, lr=c['lr'], weight_decay=.01)
        joint = append(bank, anchor[new_fit], y[new_fit]); head, metric = method.head(joint, 0.)
        inverse, cross = method.proximal_base(bank, metric, 2)
        training = loader([rows[i] for i in new_fit.tolist()], True, 300)
        batches = list(training)
        baseline = empirical(anchor[mi], y[mi], head)
        base_proxy = ctl.risks(meta_bank, head, anchor[mi[y[mi] == 1]], y[mi[y[mi] == 1]])
        snap = Snapshot(encoder, optimizer); initial_head = head.clone()
        tail_raw = set(sorted(c['order'], key=lambda k:(-sum(r['label'] == k for r in train), k))[4:])
        tail = [i for i, label in enumerate(pair) if label in tail_raw]
        for action in range(3):
            snap.restore(optimizer); head = initial_head.clone()
            for i in range(16):
                head = step(batches[i % len(batches)], head, action, inverse, cross, 1)
            after, _ = features()
            displacement = (after[new_fit]-anchor[new_fit]).mean(0)
            predicted_old = method.translate(bank, displacement)
            refit, _ = method.head(append(predicted_old, after[new_fit], y[new_fit]), 0.)
            proxy = ctl.risks(method.translate(meta_bank, displacement), refit,
                after[mi[y[mi] == 1]], y[mi[y[mi] == 1]])
            oracle = empirical(after[mi], y[mi], refit)
            true_before = torch.tensor(baseline['loss'], device='cuda', dtype=torch.float64)
            true_after = torch.tensor(oracle['loss'], device='cuda', dtype=torch.float64)
            rows_out.append(dict(action=action, fd_weight=ctl.ACTIONS[action], proxy_before=base_proxy.tolist(),
                proxy_after=proxy.tolist(), oracle_before=baseline, oracle_after=oracle,
                proxy_reward=ctl.reward(base_proxy, proxy, 1, tail),
                oracle_reward=ctl.reward(true_before, true_after, 1, tail)))
            status(status='RUNNING', phase='calibration_branches', completed_actions=action+1)
        snap.restore(optimizer); snap.verify(optimizer)
        result = dict(status='COMPLETE', audit_seed=seed, reverse=c['reverse'], start_depth=c['start_depth'],
            pseudo_order=pair, arms=rows_out, actual_updates=updates, expected_updates=c['start_depth']+48,
            meta_counts=[int((y[mi] == k).sum()) for k in (0, 1)], fit_meta_identity_disjoint=True,
            calibration_task=1,shared_prefix_already_saw_both_pseudo_classes=True,
            real_old_images_accessed=False, future_images_accessed=False, validation_images_accessed=False,
            test_accessed=False, model_retained=False, restore_verified=True, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic()-started)
        if updates != result['expected_updates']: raise ValueError('Calibration accounting mismatch')
        save(output/'CALIBRATION.json', result); status(status='COMPLETE', peak_gpu_bytes=result['peak_gpu_bytes'])
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    import sys
    run(json.load(sys.stdin))
