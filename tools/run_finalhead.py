"""Fit-only boundary selection on a fixed encoder trajectory; no optimizer steps."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import core1_competition as competition
import pcrl_control as control
import prototype_coherent as method
from next1_support import fixed_split
from run_multilabel import ApartFeatures
from run_pcrl import common_shift
from run_prototype_single import Images, extract, manifests, save


def fit_bank(old, before, current, labels):
    """Inputs contain only fit rows, including prototype membership and translation."""
    seeds = method.seed_components(before, labels)
    group, difficulty = method.memberships(before, labels, seeds)
    weights = method.sample_weights(labels, group, difficulty, before.new_zeros(len(seeds)))
    translated = method.translate(old, common_shift(before, current, labels))
    return method.append(translated, current, labels, group, weights), translated


def head(bank, old_count, action):
    reference, _ = method.head(bank, 0.)
    pairs = competition.competition(bank, reference)
    return competition.bank_head(bank, control.allocation(pairs, old_count, action), .5)


def rewards(values, old, tail):
    """Vectorized existing PCRL reward, now referenced to FIXED1 instead of action 0."""
    gain = (values[..., 1:2, :] - values) / np.maximum(values[..., 1:2, :], .1)
    negative = np.minimum(gain, 0.)
    return (gain.mean(-1) + negative[..., :old].mean(-1)
            + negative[..., old:].mean(-1)
            + (negative[..., tail].mean(-1) if tail else 0.))


def select(old_risks, losses, labels, identities, tail, seed):
    """Paired identity-block bootstrap; only current-meta sampling uncertainty."""
    old_risks, losses = np.asarray(old_risks), np.asarray(losses)
    labels, identities = np.asarray(labels), np.asarray(identities)
    old = old_risks.shape[1]
    classes = sorted(np.unique(labels).tolist())
    if (old_risks.shape[0] != 7 or losses.shape != (7, len(labels))
            or len(identities) != len(labels) or classes != list(range(old, old+len(classes)))
            or not np.isfinite(old_risks).all() or not np.isfinite(losses).all()):
        raise ValueError('Invalid boundary risk inputs')
    point = np.concatenate([old_risks, np.stack([losses[:, labels == c].mean(1) for c in classes], 1)], 1)
    boot = np.broadcast_to(point, (512, *point.shape)).copy()
    rng = np.random.default_rng(seed)
    group_counts = []
    for c in classes:
        rows = np.flatnonzero(labels == c)
        groups = np.unique(identities[rows]); group_counts.append(len(groups))
        if any(len(np.unique(labels[identities == g])) != 1 for g in groups):
            raise ValueError('Meta identity spans classes')
        sums = np.stack([losses[:, rows[identities[rows] == g]].sum(1) for g in groups])
        counts = np.array([np.sum(identities[rows] == g) for g in groups])
        draws = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=512)
        boot[:, :, c] = (draws @ sums) / (draws @ counts)[:, None]
    point_reward, boot_reward = rewards(point, old, tail), rewards(boot, old, tail)
    alternatives = [a for a in range(7) if a != 1]
    # Simultaneous one-sided basic-bootstrap proxy bound across six alternatives.
    correction = max(0., float(np.quantile((point_reward-boot_reward)[:, alternatives].max(1), .95, method='higher')))
    lower = point_reward-correction; lower[1] = 0.
    eligible = [a for a in alternatives if lower[a] > 1e-8] if min(group_counts) >= 2 else []
    selected = max(eligible, key=lambda a: (lower[a], -a)) if eligible else 1
    return selected, dict(risks=point.tolist(), rewards=point_reward.tolist(),
        lower_proxy_bounds=lower.tolist(), simultaneous_correction=correction,
        bootstrap_replicates=512, bootstrap_seed=seed, meta_identity_counts=group_counts,
        eligible_actions=eligible, selected_action=selected, reference_action=1,
        old_risk_uncertainty_modeled=False, bound_is_recall_guarantee=False)


def device_bank(bank):
    return dict(mu=bank['mu'].cuda(), Q=bank['Q'].cuda(), n=bank['n'],
        components=[{k: v.cuda() if torch.is_tensor(v) else v for k, v in c.items()}
                    for c in bank['components']])


@torch.no_grad()
def run(config):
    output = Path(config['output']); source = Path(config['checkpoint_directory'])
    if output.exists() or config['method'] != 'FINALHEAD':
        raise ValueError('Fresh FINALHEAD output required; no retry')
    receipt = json.loads((source/'STATUS.json').read_text())
    if receipt.get('status') != 'TRAINED' or receipt.get('steps') != 476:
        raise ValueError('Fixed encoder trajectory is not complete')
    output.mkdir(parents=True); save(output/'INPUT.private.json', config)
    started = time.monotonic(); decisions = []
    def budget():
        if time.monotonic()-started >= config['max_wall_seconds']:
            raise TimeoutError('FINALHEAD residence cap')
    def status(**fields):
        save(output/'STATUS.json', dict(steps=476, reused_encoder_updates=476,
            actual_updates=0, retained_updates=0, rollout_updates=0, policy_updates=0,
            diagnostic_gpu_seconds=0., elapsed_seconds=time.monotonic()-started, **fields))
    try:
        torch.set_num_threads(4); seed = config['seed']
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        status(status='RUNNING', phase='initialize')
        train, _ = manifests(config)
        tail = set(sorted(config['order'], key=lambda c: (-sum(r['label'] == c for r in train), c))[4:])
        encoder = ApartFeatures(config['legacy_repo'], config['weight'], len(config['order']), 'cuda:0', seed)
        encoder.requires_grad_(False).eval()
        from run_medical_v2 import transform
        def features(state, rows, task):
            encoder.load_state_dict(state['model'], strict=True); encoder.eval()
            workers = int(config.get('workers', 0))
            options = dict(multiprocessing_context='spawn', persistent_workers=True) if workers else {}
            loader = DataLoader(Images(rows, config['images'], transform(False), seed+task*100003),
                batch_size=config['batch_size'], num_workers=workers, shuffle=False,
                generator=torch.Generator().manual_seed(seed+task*2003), **options)
            x, y = extract(encoder, loader, budget)
            if y.tolist() != [r['label'] for r in rows]:
                raise ValueError('Canonical feature order changed')
            return torch.as_tensor(x, dtype=torch.float64, device='cuda')
        (output/'stage_1.pt').symlink_to(source/'stage_1.pt')
        previous = torch.load(source/'stage_1.pt', map_location='cpu', weights_only=False)
        for task in range(2, 5):
            budget(); status(status='RUNNING', phase='fit_features', task=task)
            current = torch.load(source/f'stage_{task}.pt', map_location='cpu', weights_only=False)
            expected_seen = config['order'][:task*2]
            if (current['seen'] != expected_seen or previous['seen'] != expected_seen[:-2]
                    or current['configuration']['method'] != 'FIXED1'
                    or current['configuration']['seed'] != seed or current['controller_action'] != 1):
                raise ValueError('Incorrect fixed encoder source')
            rows = [r for r in train if r['label'] in expected_seen[-2:]]
            fit, meta = fixed_split(rows, config)
            fit_rows, meta_rows = [rows[i] for i in fit], [rows[i] for i in meta]
            if {r['identity_component'] for r in fit_rows} & {r['identity_component'] for r in meta_rows}:
                raise ValueError('Fit/meta identity overlap')
            labels = torch.tensor([expected_seen.index(r['label']) for r in fit_rows], device='cuda')
            before = features(previous, fit_rows, task)
            after = features(current, fit_rows, task)
            candidate, translated = fit_bank(device_bank(previous['bank']), before, after, labels)
            heads, solves = zip(*(head(candidate, len(previous['seen']), a) for a in range(7)))
            old_risks = torch.stack([control.old_risk(translated, w) for w in heads]).cpu().numpy()
            status(status='RUNNING', phase='meta_scoring', task=task)
            xmeta = features(current, meta_rows, task)
            ymeta = torch.tensor([expected_seen.index(r['label']) for r in meta_rows], device='cuda')
            losses = torch.stack([F.cross_entropy(xmeta @ w, ymeta, reduction='none') for w in heads]).cpu().numpy()
            selected, audit = select(old_risks, losses, ymeta.cpu().numpy(),
                [r['identity_component'] for r in meta_rows],
                [i for i, c in enumerate(expected_seen) if c in tail], seed+task*100003+511)
            audit.update(task=task, seen=expected_seen, fit_n=len(fit), meta_n=len(meta),
                candidate_fit_counts=candidate['n'], candidate_solves=solves,
                candidate_meta_excluded=True, encoder_optimizer_updates=0)
            # Full-training statistics are used for refit only after this decision.
            bank = device_bank(current['bank'])
            fixed, fixed_audit = head(bank, len(previous['seen']), 1)
            error = float((fixed.float().cpu()-current['head']).abs().max())
            if not torch.allclose(fixed.float().cpu(), current['head'], atol=2e-6, rtol=2e-6):
                raise ValueError('FIXED1 full-refit head reconstruction mismatch')
            chosen, solve = (fixed, fixed_audit) if selected == 1 else head(bank, len(previous['seen']), selected)
            audit.update(fixed1_head_max_abs_error=error, final_solve=solve,
                head_change_norm=float((chosen-fixed).norm()), selected_at=time.time())
            decisions.append(audit)
            save(output/'CONTROLLER.json', dict(decisions=decisions, adapter_updates=0, policy_updates=0))
            saved = dict(current, head=chosen.float().cpu(), controller_action=selected,
                configuration=config, readout_selection=audit)
            torch.save(saved, output/f'stage_{task}.pt')
            previous = current
            del saved, before, after, candidate, translated, heads, xmeta, bank, fixed, chosen
        status(status='TRAINED', phase='SELECTION_SEALED', stages=4,
            selected_actions=[v['selected_action'] for v in decisions], peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
