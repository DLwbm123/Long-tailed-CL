"""Small CPU fixtures for train-only tasks, masked binary heads and local priors."""
import json
import time

import numpy as np
import torch

from check_multilabel import statistics_checks
from ml_core import binary_weights
from ml_statistics import add, empty, ridge
from prepare_rfmid_ml import BENCHMARK_LABELS
from rfmid_local import add_local, empty_local, penalty, transport_local
from rfmid_protocol import build_plan


class TinyRegions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 6
        self.net = torch.nn.Module()
        self.net.regular = torch.nn.Linear(3, 3)
        self.net.few = torch.nn.Linear(3, 3)
        self.net.backbone = torch.nn.Module()
        self.net.backbone.norm = torch.nn.LayerNorm(3).requires_grad_(False)

    def forward(self, x):
        regular = self.net.backbone.norm(self.net.regular(x))
        few = self.net.backbone.norm(self.net.few(x))
        return torch.cat((few[:, 0], regular[:, 0]), dim=-1)


def trainer_fixture(native_trainer=None):
    import copy
    from torch.utils.data import DataLoader, TensorDataset
    from run_multilabel import Trainer
    torch.manual_seed(734)
    encoder = TinyRegions()
    x = torch.randn(10, 5, 3)
    y = torch.tensor([[i % 2, (i // 2) % 2, i % 3 == 0] for i in range(10)], dtype=torch.float32)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[:5, 2] = False
    trainers = [Trainer(copy.deepcopy(encoder), 3), Trainer(copy.deepcopy(encoder), 3, local_transport=True)]
    if native_trainer is not None:
        trainers.append(native_trainer(copy.deepcopy(encoder), 3))
    for lo, hi in ((0, 5), (5, 10)):
        data = TensorDataset(x[lo:hi], y[lo:hi], mask[lo:hi])
        loader = DataLoader(data, batch_size=3, shuffle=False)
        for trainer in trainers:
            record = trainer.fit_task(loader, loader, 2)
            assert trainer.steps == (4 if lo == 0 else 8)
            assert all(r['clip_calls'] == r['optimizer_step_calls'] == 1 and r['reward_coefficient'] == 0 for r in record['updates'])
        if native_trainer is not None:
            for k, v in trainers[0].encoder.state_dict().items():
                assert torch.equal(v, trainers[2].encoder.state_dict()[k])
            assert torch.equal(trainers[0].head, trainers[2].head)
            for k in trainers[0].bank:
                assert np.array_equal(trainers[0].bank[k], trainers[2].bank[k])
    assert np.array_equal(trainers[1].bank['n'], trainers[1].local['n'])
    state = trainers[1].state_dict()
    restored = Trainer(copy.deepcopy(encoder), 3, local_transport=True)
    restored.load_state_dict(state)
    assert torch.equal(restored.head, trainers[1].head)
    assert np.array_equal(restored.local['means'], trainers[1].local['means'])
    for regional in (False, True):
        t = Trainer(copy.deepcopy(encoder), 3, local_transport=regional)
        prior = copy.deepcopy(t.encoder.state_dict())
        r = t.fit_task(loader, loader, 2, preflight=True)
        assert t.steps == r['optimizer_updates'] == r['policy_updates'] == 0 and r['gradient_norm'] > 0
        assert all(torch.equal(v, t.encoder.state_dict()[k]) for k, v in prior.items())
    return dict(two_task_base_and_candidate_finite=True, masked_label_counts_exact=True,
                original_optimizer_and_clip_retained=True, checkpoint_local_roundtrip=True,
                zero_optimizer_preflight_leaves_encoder_exact=True,
                prechange_native_A_trajectory_exact=native_trainer is not None)


def permission_fixture():
    import copy
    import tempfile
    from pathlib import Path
    from run_rfmid import validate_config, evaluation_barrier
    config = dict(rfmid_multilabel_authorized=True, name='RFMiD', evaluation_split='official_validation',
                  seed=74002, epochs=2, batch_size=64, workers=4, lr=.0003, expected_steps=66,
                  policy_updates=0, threshold=.5, test_accessed=False, module='base', operation='train',
                  labels=list(BENCHMARK_LABELS))
    validate_config(config)
    for key, bad in [('rfmid_multilabel_authorized', False), ('name', 'NIHCXRLT'),
                     ('evaluation_split', 'official_test'), ('test_accessed', True),
                     ('module', 'local_readout'), ('expected_steps', 64), ('threshold', .6),
                     ('labels', list(BENCHMARK_LABELS[:-1])), ('operation', 'test')]:
        changed = copy.deepcopy(config); changed[key] = bad
        try: validate_config(changed)
        except ValueError: pass
        else: raise AssertionError('RFMiD scope or test permission escaped')
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory) / 'PROGRAM_STATE.json'
        p.write_text(json.dumps(dict(phase='TRAIN', failures=[], training_sealed_at=None)))
        try: evaluation_barrier(dict(evaluation_gate=str(p)), dict(tasks=[{}] * 5))
        except ValueError: pass
        else: raise AssertionError('RFMiD evaluation opened before both trained arms')
    return dict(test_and_cross_dataset_access_rejected=True, method_and_budget_fixed=True,
                premature_validation_evaluation_rejected=True)


def report_fixture():
    import copy
    import tempfile
    from pathlib import Path
    from ml_metrics import evaluate, groups
    from summarize_rfmid import report
    rng = np.random.default_rng(77)
    y = ((np.arange(640)[:, None] + np.arange(29)[None, :]) % 37 == 0).astype(float)
    scores = rng.normal(size=y.shape)
    runs = {}
    for arm in ('base', 'local_transport'):
        stages, previous = [], []
        for t, count in enumerate((9, 14, 19, 24, 29), 1):
            visible = list(BENCHMARK_LABELS[:count])
            mask = np.tile(np.arange(29) < count, (640, 1))
            result = evaluate(scores + (.1 * y if arm == 'local_transport' else 0), y, mask, BENCHMARK_LABELS)
            metric_groups = dict(head=list(BENCHMARK_LABELS[:10]), middle=list(BENCHMARK_LABELS[10:20]),
                                 tail=list(BENCHMARK_LABELS[20:]), old=previous, new=visible[len(previous):])
            result.update(task=t, visible_labels=visible,
                          groups=groups(result, {k: [c for c in v if c in visible] for k, v in metric_groups.items()}))
            stages.append(result); previous = visible
        runs['RFMiD_' + arm] = dict(metrics=dict(dataset='RFMiD', test_accessed=False,
                                                evaluation_split='official_validation', stages=stages))
    audit = dict(round_gpu_process_residence_seconds=0., parallel_wall_seconds=0., cumulative_gpu_process_residence_seconds=0.)
    with tempfile.TemporaryDirectory() as folder:
        result = report(dict(runs=runs), Path(folder) / 'fixture', audit)
        assert set(result['rows']) == {'base', 'local_transport'} and len(result['delta_pp']) == 7
        assert result['delta_pp']['final_mAP'] > 0
        malformed = copy.deepcopy(runs)
        malformed['RFMiD_base']['metrics']['test_accessed'] = True
        try: report(dict(runs=malformed), Path(folder) / 'bad', audit)
        except ValueError: pass
        else: raise AssertionError('Test report accepted as RFMiD validation')
    return dict(full_29_label_report_and_five_stage_denominators_checked=True,
                all_six_mAP_contrasts_present=True, report_test_permission_enforced=True,
                synthetic_fixture_only=True)


def check():
    begun = time.monotonic()
    rows = []
    for i, label in enumerate(BENCHMARK_LABELS):
        for _ in range(3 + i):
            rows.append(dict(ID=str(len(rows) + 1), **{k: str(int(k == label)) for k in BENCHMARK_LABELS}))
    plan, audit = build_plan(rows)
    ids = [s for t in plan['tasks'] for s in t['train_ids']]
    assert len(ids) == len(set(ids)) == len(rows)
    assert [len(t['visible_labels']) for t in plan['tasks']] == [9, 14, 19, 24, 29]
    assert all(v == 0 for v in audit['hidden_future_positive_occurrences'].values())
    assert all(t['meta_images'] == 0 for t in audit['tasks'])
    assert build_plan(rows[::-1])[1] == audit
    co = rows.copy()
    co[0] = dict(co[0], Normal='0', ST='1', DR='1')
    _, co_audit = build_plan(co)
    assert sum(co_audit['hidden_future_positive_occurrences'].values()) > 0
    try: build_plan(rows + [rows[0]])
    except ValueError: pass
    else: raise AssertionError('Repeated training identity accepted')
    native_audit = statistics_checks()
    rng = np.random.default_rng(741)
    x, parts = rng.normal(size=(12, 5)), rng.normal(size=(12, 4, 5))
    y = np.array([[i % 2, 0, np.nan] for i in range(12)])
    mask = np.tile([True, True, False], (12, 1))
    state = add(empty(5, 3), x, y, mask)
    W, _ = ridge(state)
    identity_W, _ = ridge(state, penalty=lambda _: np.eye(6))
    assert np.array_equal(W, identity_W)
    local = add_local(empty_local(3, 5), parts, y, mask)
    split = add_local(add_local(empty_local(3, 5), parts[:4], y[:4], mask[:4]), parts[4:], y[4:], mask[4:])
    assert np.array_equal(local['n'], state['n']) and not local['n'][2].any()
    np.testing.assert_allclose(local['means'], split['means'], atol=1e-14, rtol=1e-14)
    weights = binary_weights(torch.from_numpy(y), torch.from_numpy(mask)).numpy()
    shift = rng.normal(size=(4, 5))
    moved, _ = transport_local(local, parts, parts + shift, y, mask, weights)
    direct = add_local(empty_local(3, 5), parts + shift, y, mask)
    np.testing.assert_allclose(moved['means'], direct['means'], atol=1e-14, rtol=1e-14)
    for label in range(3):
        prior = penalty(local, label)
        eig = np.linalg.eigvalsh(prior)
        assert eig.min() > 0 and eig.max() <= 1 + 1e-12 and prior[-1, -1] == 1
        centers = local['means'][label, local['n'][label] > 0].reshape(-1, 5)
        P = (centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12) / 2).T
        np.testing.assert_allclose(prior[:-1, :-1], np.linalg.inv(np.eye(5) + P @ P.T), atol=1e-14)
    local_W, solve = ridge(state, penalty=lambda label: penalty(local, label))
    assert np.isfinite(local_W).all() and not local_W[:, 2].any()
    assert not np.array_equal(local_W[:, :2], W[:, :2])
    for bad in (np.eye(5), np.full((6, 6), np.nan), np.triu(np.ones((6, 6)))):
        try: ridge(state, penalty=lambda _: bad)
        except ValueError: pass
        else: raise AssertionError('Invalid binary prior accepted')
    return dict(status='PASS', samples_assigned_once=True, hidden_future_masks_preserved=True,
                multihot_hidden_positive_loss_disclosed=True, native_statistics=native_audit,
                native_identity_head_exact=True, regional_binary_counts_match_global=True,
                hidden_targets_not_counted_negative=True, absent_sign_and_inactive_label_checked=True,
                regional_transport_matches_direct_fixture=True, inverse_prior_matches_direct=True,
                bias_regularization_unchanged=True, candidate_head_finite=True,
                max_fixture_solve_residual=max(v for v in solve['residuals'] if v is not None),
                malformed_prior_rejected=True, gpu_work=False, test_accessed=False,
                cpu_seconds=time.monotonic() - begun)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, allow_nan=False))
