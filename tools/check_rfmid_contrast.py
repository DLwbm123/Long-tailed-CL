"""Synthetic CPU algebra and adapter integration; never reads RFMiD images."""
import copy
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from check_rfmid_protocol import TinyRegions, permission_fixture
from ml_statistics import add, empty, ridge
import rfmid_local
from rfmid_contrast import columns, penalty
from run_multilabel import Trainer


def check():
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    rng = np.random.default_rng(74002)
    state = rfmid_local.empty_local(3, 9)
    state['n'][:] = [[13, 7], [12, 8], [14, 6]]
    state['means'][:] = rng.normal(size=state['means'].shape)
    original = copy.deepcopy(state)
    max_residual, max_inverse, max_leakage = 0., 0., 0.
    for j in range(3):
        P, audit = columns(state, j)
        R = penalty(state, j)
        direct = np.linalg.inv(np.eye(9) + P @ P.T)
        max_inverse = max(max_inverse, float(np.max(np.abs(R[:-1, :-1] - direct))))
        np.testing.assert_allclose(R[:-1, :-1], direct, atol=1e-12)
        assert np.linalg.eigvalsh(R).min() >= .5 - 1e-12
        assert R[-1, -1] == 1 and not R[-1, :-1].any()
        max_leakage = max(max_leakage, audit['shared_leakage'])
        shared = ((state['means'][j, 1] + state['means'][j, 0]) / 2).T
        np.testing.assert_allclose(R[:-1, :-1] @ shared, shared, atol=1e-12)
        swapped = dict(n=state['n'][:, ::-1], means=state['means'][:, ::-1])
        np.testing.assert_allclose(R, penalty(swapped, j), atol=1e-12)
        rotation, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        rotated = dict(n=state['n'], means=state['means'] @ rotation)
        np.testing.assert_allclose(penalty(rotated, j)[:-1, :-1], rotation.T @ R[:-1, :-1] @ rotation, atol=1e-12)
    for kind in ('equal', 'missing', 'zero', 'tiny'):
        fixture = copy.deepcopy(state)
        if kind == 'equal': fixture['means'][:, 1] = fixture['means'][:, 0]
        elif kind == 'missing': fixture['n'][:, 1] = 0
        elif kind == 'zero': fixture['means'][:] = 0
        else: fixture['means'][:, 1] = fixture['means'][:, 0] + 1e-14
        np.testing.assert_allclose(penalty(fixture, 0), np.eye(10), atol=1e-12)
    for kind in ('negative_count', 'fractional_count', 'nan', 'shape'):
        bad = copy.deepcopy(state)
        if kind == 'negative_count': bad['n'][0, 0] = -1
        elif kind == 'fractional_count': bad['n'] = bad['n'].astype(float) + .5
        elif kind == 'nan': bad['means'][0, 0, 0, 0] = np.nan
        else: bad['means'] = bad['means'][:, :, :3]
        try: penalty(bad, 0)
        except ValueError: pass
        else: raise AssertionError('Invalid prior accepted: ' + kind)
    z, y = rng.normal(size=(30, 9)), rng.integers(0, 2, (30, 3))
    mask = rng.random((30, 3)) > .15
    bank = add(empty(9, 3), z, y, mask)
    bank_copy = copy.deepcopy(bank)
    native, _ = ridge(bank)
    for projected in (False, True):
        head, audit = ridge(bank, penalty=lambda j: penalty(state, j, remove_shared=projected))
        max_residual = max(max_residual, max(audit['residuals']))
        assert np.isfinite(head).all()
    for key in bank: np.testing.assert_array_equal(bank[key], bank_copy[key])
    np.testing.assert_array_equal(native, ridge(bank)[0])
    for key in state: np.testing.assert_array_equal(state[key], original[key])
    parts = rng.normal(size=(30, 4, 9))
    local = rfmid_local.add_local(rfmid_local.empty_local(3, 9), parts, y, mask)
    regional_shift = rng.normal(size=(4, 9)) * .01
    moved, _ = rfmid_local.transport_local(local, parts, parts + regional_shift,
                                           y, mask, np.ones((3, 2)))
    np.testing.assert_array_equal(moved['n'], local['n'])
    np.testing.assert_allclose(moved['means'][:, 1] - moved['means'][:, 0],
                               local['means'][:, 1] - local['means'][:, 0], atol=1e-12)

    # Only this CPU process substitutes the prior callable; production files stay unchanged.
    torch.manual_seed(74002)
    initial = TinyRegions()
    x = torch.randn(12, 5, 3)
    y = torch.tensor([[i % 2, (i // 2) % 2, i % 3 == 0] for i in range(12)], dtype=torch.float32)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[:6, 2] = False
    y = torch.where(mask, y, 0.)
    loaders = [DataLoader(TensorDataset(x[a:b], y[a:b], mask[a:b]), batch_size=3)
               for a, b in ((0, 6), (6, 12))]
    old_penalty = rfmid_local.penalty
    synthetic_steps = 0
    try:
        for projected in (False, True):
            rfmid_local.penalty = lambda s, j: penalty(s, j, remove_shared=projected)
            pre = Trainer(copy.deepcopy(initial), 3, local_transport=True)
            before = copy.deepcopy(pre.encoder.state_dict())
            audit = pre.fit_task(loaders[0], loaders[0], 2, preflight=True)
            assert pre.steps == audit['optimizer_updates'] == audit['policy_updates'] == 0
            assert audit['gradient_norm'] > 0
            assert all(torch.equal(v, pre.encoder.state_dict()[k]) for k, v in before.items())
            trainer = Trainer(copy.deepcopy(initial), 3, local_transport=True)
            for loader in loaders:
                trainer.fit_task(loader, loader, 2)
                np.testing.assert_array_equal(trainer.local['n'], trainer.bank['n'])
                assert torch.isfinite(trainer.head).all()
            assert trainer.steps == 8
            synthetic_steps += trainer.steps
            restored = Trainer(copy.deepcopy(initial), 3, local_transport=True)
            restored.load_state_dict(trainer.state_dict())
            assert torch.equal(restored.head, trainer.head)
            np.testing.assert_array_equal(restored.local['means'], trainer.local['means'])
    finally:
        rfmid_local.penalty = old_penalty
    permission_fixture()
    return dict(status='PASS', inverse_max_error=max_inverse, max_solve_residual=max_residual,
                shared_leakage_max=max_leakage, eigenvalue_lower_bound=.5,
                equal_missing_zero_tiny_prior_identity=True, sign_swap_invariant=True,
                orthogonal_equivariance=True, shared_span_penalty_identity=True,
                global_and_local_aggregates_unmodified=True, native_identity_head_exact=True,
                original_common_transport_counts_and_raw_difference_preserved=True,
                invalid_inputs_rejected=True, cpu_adapter_two_task_and_checkpoint_pass=True,
                synthetic_zero_update_backwards=2, synthetic_optimizer_updates=synthetic_steps,
                real_gpu_preflights=0, gpu_processes=0, formal_adapter_updates=0, test_accessed=False,
                production_worker_enabled=False, cpu_process_seconds=time.process_time()-cpu,
                wall_seconds=time.monotonic()-wall)


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, allow_nan=False, indent=2))
