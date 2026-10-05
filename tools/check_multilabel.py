"""CPU-only mathematical and end-to-end synthetic checks; no NIH training."""
import copy
import csv
import itertools
import json
import os
from pathlib import Path
import tempfile
import types
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from ml_core import METHODS, auxiliary, bernoulli_reward, binary_weights, objective, project, update
from ml_data import NIH_LABELS, MultiLabelImages, read_manifest, validate_plan
from ml_metrics import evaluate, ranking_metrics
from ml_statistics import empty, add, translated, ridge
from run_multilabel import Trainer, run


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected rejection')


def math_checks():
    g = torch.Generator().manual_seed(23)
    z = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
    head = torch.cat((torch.eye(3, dtype=z.dtype), torch.zeros(1, 3, dtype=z.dtype)))
    y = torch.tensor([[1., 0., float('nan')], [0., 1., -1.], [1., 1., -1.], [0., 0., -1.]], dtype=z.dtype)
    mask = torch.tensor([[1, 1, 0]] * 4, dtype=torch.bool)
    weights = binary_weights(y, mask)
    first = []
    for method in METHODS:
        state = g.get_state().clone()
        total, _, _, sigma, _, actions = objective(z, target, head, y, mask, weights, method, 1, .5, g, [0, 0, 0])
        assert torch.equal(state, g.get_state()) and sigma == .5 and actions is None
        first.append(total)
    for value in first:
        torch.testing.assert_close(value, first[0], rtol=0, atol=0)
    grad = torch.autograd.grad(first[0], z)[0]
    assert not grad[:, 2].count_nonzero()
    total = objective(z, target, head, y, mask, weights, 'EP01', 2, .5, g, [1, 0, 0])[0]
    assert torch.autograd.grad(total, target, allow_unused=True)[0] is None
    lp, _, retention = bernoulli_reward(z, z.detach(), torch.zeros_like(z), torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(retention, torch.ones_like(retention))
    global_rng = torch.get_rng_state().clone()
    policy_rng = g.get_state().clone()
    auxiliary(lp, torch.ones_like(lp), mask, torch.ones_like(z), .5, 'sample', g)
    assert torch.equal(global_rng, torch.get_rng_state()) and not torch.equal(policy_rng, g.get_state())

    # Exhaustive Bernoulli group expectation, including nonuniform weight.
    logit = torch.tensor([[.4]], dtype=torch.float64, requires_grad=True)
    logp, reward, _ = bernoulli_reward(logit, torch.tensor([[-.3]], dtype=torch.float64),
                                      torch.ones_like(logit), torch.ones(1, dtype=torch.bool))
    m, w = torch.ones_like(logit, dtype=torch.bool), torch.full_like(logit, 2.3)
    exact = auxiliary(logp, reward, m, w, .5, 'exact')[0]
    expected = torch.autograd.grad(exact, logit, retain_graph=True)[0]
    accumulated = torch.zeros_like(logit)
    for group in itertools.product((0, 1), repeat=8):
        actions = torch.tensor([group])
        probability = logp.detach().exp()[0, 0, actions].prod()
        sampled = auxiliary(logp, reward, m, w, .5, 'sample', actions=actions)[0]
        accumulated += probability * torch.autograd.grad(sampled, logit, retain_graph=True)[0]
    torch.testing.assert_close(accumulated, expected, atol=1e-12, rtol=1e-12)
    loss = auxiliary(logp, torch.ones_like(reward), m, w, .5, 'sample', actions=torch.tensor([[0, 1] * 4]))[0]
    assert not torch.autograd.grad(loss, logit)[0].count_nonzero()
    hp, info = project((torch.tensor([-2., 3.]), torch.tensor([4.])), (torch.tensor([1., 0.]), None))
    torch.testing.assert_close(hp[0], torch.tensor([0., 3.]))
    assert info['triggered'] and info['dot_after'] >= -info['numerical_tolerance']
    return dict(masked_gradients_zero=True, teacher_detached=True, task1_identical=True,
                policy_rng_isolated=True, constant_reward_zero=True,
                equal_teacher_retention_constant=True, enumerated_groups=256,
                raw_LOO_expectation_error=float((accumulated - expected).abs().max()))


def statistics_checks():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 4))
    y = rng.integers(0, 2, size=(12, 3)).astype(float)
    mask = rng.random(y.shape) > .2
    mask[:, 2] = False
    y[~mask] = np.nan
    state = add(empty(4, 3), x, y, mask)
    W, audit = ridge(state)
    for c in range(2):
        valid = mask[:, c]
        X, Y = np.c_[x[valid], np.ones(valid.sum())], y[valid, c]
        weight = np.array([1 / (2 * (Y == v).sum()) for v in Y])
        direct = np.linalg.solve(X.T @ (weight[:, None] * X) + .001 * np.eye(5),
                                 X.T @ (weight * (2 * Y - 1)))
        np.testing.assert_allclose(W[:, c], direct, atol=1e-11)
    assert not W[:, 2].any() and audit['residuals'][2] is None
    b = rng.normal(size=4)
    carried = translated(state, b)
    direct = add(empty(4, 3), x + b, y, mask)
    for key in state:
        np.testing.assert_allclose(carried[key], direct[key], atol=1e-11)
    # Accumulation across tasks does not turn unknown labels into negatives.
    combined = add(add(empty(4, 3), x[:5], y[:5], mask[:5]), x[5:], y[5:], mask[5:])
    for key in state:
        np.testing.assert_allclose(combined[key], state[key], atol=1e-11)
    return dict(masked_binary_ridge_matches_direct=True, translation_exact=True,
                split_accumulation_exact=True, unobserved_label_inactive=True)


class TinyEncoder(nn.Module):
    dim = 4
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, self.dim)
    def forward(self, x):
        return torch.nn.functional.normalize(self.layer(x), dim=1)


def engine_checks():
    torch.manual_seed(44)
    base = TinyEncoder()
    x = torch.randn(12, 3)
    y = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]] * 3)
    m = torch.ones_like(y, dtype=torch.bool)
    ds = TensorDataset(x, y, m)
    def loaders():
        return (DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4, shuffle=True,
                                                       generator=torch.Generator().manual_seed(14)))
    task1 = []
    for method in (*METHODS, 'F_S'):
        trainer = Trainer(copy.deepcopy(base), 2, method)
        trainer.fit_task(*loaders(), epochs=1)
        task1.append(copy.deepcopy(trainer.encoder.state_dict()))
        checkpoint = copy.deepcopy(trainer.state_dict())
        clip_original = torch.nn.utils.clip_grad_norm_
        step_original, step_calls = torch.optim.AdamW.step, []
        def counted_step(optimizer, *args, **kwargs):
            step_calls.append(1)
            return step_original(optimizer, *args, **kwargs)
        with patch('torch.nn.utils.clip_grad_norm_', wraps=clip_original) as clip, \
                patch.object(torch.optim.AdamW, 'step', counted_step):
            result = trainer.fit_task(*loaders(), epochs=1)
            assert clip.call_count == (0 if method == 'F_S' else 3)
            assert len(step_calls) == clip.call_count
        assert all(v['optimizer_step_calls'] == v['clip_calls'] == 1 for v in result['updates'])
        if method == 'F_S':
            for k, v in trainer.encoder.state_dict().items():
                torch.testing.assert_close(v, checkpoint['encoder'][k], rtol=0, atol=0)
        expected_state = copy.deepcopy(trainer.state_dict())
        trainer.load_state_dict(checkpoint)
        repeated = trainer.fit_task(*loaders(), epochs=1)
        for k, v in trainer.encoder.state_dict().items():
            torch.testing.assert_close(v, expected_state['encoder'][k], rtol=0, atol=0)
        assert trainer.sigma == expected_state['sigma']
        assert torch.equal(trainer.policy.get_state(), expected_state['policy_rng'])
        np.testing.assert_array_equal(repeated['W_final'], result['W_final'])
        assert trainer.bank['n'].sum() == 48
    for state in task1[1:]:
        for k in state:
            torch.testing.assert_close(state[k], task1[0][k], rtol=0, atol=0)
    teacher = copy.deepcopy(base).requires_grad_(False)
    opt = torch.optim.AdamW(base.parameters(), lr=.01)
    w = torch.ones(2, 2)
    head = torch.randn(5, 2)
    policy = torch.Generator().manual_seed(1)
    before = copy.deepcopy(base.state_dict())
    state = policy.get_state().clone()
    _, diag, actions = update(base, teacher, opt, x, y, torch.zeros_like(m), head, w, 'P01', 2, .5, policy, [1, 1])
    assert diag['skipped'] and actions is None and torch.equal(state, policy.get_state())
    for k, v in base.state_dict().items():
        assert torch.equal(v, before[k])
    rejects(lambda: run({'formal_authorized': False}))
    return dict(conditions=list((*METHODS, 'F_S')), two_tasks_each=True,
                task1_state_equal=True, stage_boundary_restore_exact=True,
                frozen_reference_unchanged=True, no_label_batch_skips_update=True,
                one_clip_and_optimizer_step_per_update=True, formal_gate_closed=True)


class TinyImageEncoder(TinyEncoder):
    def forward(self, x):
        return super().forward(x.mean(dim=(2, 3)))


def entrypoint_check():
    """Exercise actual orchestration using synthetic images and a tiny encoder."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); (root / 'prepared').mkdir()
        (root / 'prepared/READY.json').write_text(json.dumps({'status': 'READY', 'synthetic': True}))
        names = [f'{i}.png' for i in range(14)]
        with (root / 'prepared/manifest.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'patient_id', 'official_split', *NIH_LABELS])
            for i, name in enumerate(names):
                writer.writerow([name, f'patient{i}', 'test' if i >= 12 else 'train_val',
                                 *[(i + c) % 2 for c in range(14)]])
                Image.new('RGB', (8, 8), (i * 15, 127, 42)).save(root / name)
        plan = {'tasks': [{'train_images': names[:4], 'visible_labels': list(NIH_LABELS[:2])},
                          {'train_images': names[4:8], 'visible_labels': list(NIH_LABELS)}],
                'validation_images': names[8:12]}
        (root / 'plan.json').write_text(json.dumps(plan))
        config = dict(formal_authorized=True, data_root=str(root), output=str(root / 'out'),
                      plan=str(root / 'plan.json'), max_steps=10, max_wall_seconds=60,
                      seed=74002, legacy_repo='SYNTHETIC', weight='SYNTHETIC', device='cpu',
                      method='P01', epochs=1, batch_size=2)
        opened = []
        original_open = Image.open
        def image_open(path, *args, **kwargs):
            filename = Path(path).name
            assert filename not in names[12:], 'Official test image was opened'
            if filename in names[8:12]:
                assert (root / 'out/stage_2.pt').exists(), 'Validation preceded training completion'
            opened.append(filename)
            return original_open(path, *args, **kwargs)
        transforms = types.SimpleNamespace(transform=lambda train: lambda image:
            torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32) / 255)
        with patch('run_multilabel.ApartFeatures', side_effect=lambda *args: TinyImageEncoder()), \
                patch.dict('sys.modules', {'run_medical_v2': transforms}), \
                patch.object(Image, 'open', image_open):
            run(config)
        status = json.loads((root / 'out/STATUS.json').read_text())
        reports = json.loads((root / 'out/metrics.json').read_text())
        assert status['status'] == 'COMPLETE' and status['training_steps'] == 4
        assert len(reports) == 2 and len(reports[-1]['per_label']) == 14
        assert set(opened) == set(names[:12])
    return dict(status='PASS', image_stream_to_reports=True, fourteen_labels=True,
                validation_after_all_stages=True, official_test_never_opened=True)


def data_and_metrics_checks():
    labels = ('a', 'b')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (root / 'manifest.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'patient_id', 'official_split', *labels])
            for i, (patient, split, a, b) in enumerate([('p1', 'train_val', 1, 0), ('p2', 'train_val', 0, 1),
                                                       ('p3', 'train_val', 1, -1), ('p4', 'test', 0, 1)]):
                w.writerow([f'{i}.png', patient, split, a, b])
                Image.new('L', (8, 8), 64).save(root / f'{i}.png')
        rows = read_manifest(root / 'manifest.csv', labels)
        plan = dict(tasks=[dict(train_images=['0.png'], visible_labels=['a']),
                           dict(train_images=['1.png'], visible_labels=['a', 'b'])], validation_images=['2.png'])
        assert validate_plan(rows, plan, labels)['patient_overlap'] == 0
        broken = copy.deepcopy(plan); broken['tasks'][0]['train_images'] = ['3.png']
        rejects(lambda: validate_plan(rows, broken, labels))
        broken = copy.deepcopy(plan); broken['validation_images'] = ['0.png']
        rejects(lambda: validate_plan(rows, broken, labels))
        ds = MultiLabelImages(rows, root, ['1.png', '2.png'], ['a'], lambda image: torch.tensor(np.array(image)), labels=labels, training=True)
        assert not ds.observed[:, 1].any() and not ds.targets[:, 1].any()
        assert ds[0][0].shape == (8, 8, 3)
        rejects(lambda: MultiLabelImages(rows, root, ['3.png'], labels, lambda x: x, labels=labels, training=True))
        rows['1.png']['patient_id'] = 'p1'
        rejects(lambda: validate_plan(rows, plan, labels))
    y = np.array([[1., 0., 1.], [0., 0., 0.], [1., 0., 1.], [0., 0., 0.]])
    z = np.array([[3., -1., 0.], [-1., -1., 0.], [2., -1., 0.], [-2., -1., 0.]])
    mask = np.ones_like(y, dtype=bool); mask[:, 2] = False
    report = evaluate(z, y, mask, ['perfect', 'no_positive', 'unknown'])
    assert report['per_label'][0]['AP'] == report['per_label'][0]['AUROC'] == 1
    assert report['per_label'][1]['AP'] is None and report['per_label'][2]['observed'] == 0
    assert report['aggregate']['mAP_valid_labels'] == 1
    ap, auc = ranking_metrics(np.array([1, 0, 1, 0]), np.zeros(4))
    assert ap == auc == .5
    # AUROC reference from explicit positive/negative pair comparisons, with ties.
    rng = np.random.default_rng(9)
    for _ in range(30):
        true = rng.integers(0, 2, 30); score = rng.integers(-2, 3, 30)
        ap, auc = ranking_metrics(true, score)
        p, n = score[true == 1], score[true == 0]
        direct = ((p[:, None] > n).astype(float) + .5 * (p[:, None] == n)).mean()
        np.testing.assert_allclose(auc, direct)
        thresholds = sorted(set(score), reverse=True)
        prev, ref = 0., 0.
        for threshold in thresholds:
            selected = score >= threshold
            recall = true[selected].sum() / true.sum()
            ref += (recall - prev) * true[selected].mean(); prev = recall
        np.testing.assert_allclose(ap, ref)
    return dict(test_and_patient_leakage_rejected=True, visibility_masks=True,
                perfect_and_undefined_metrics=True, tied_AP_AUROC=True,
                ranking_metrics_match_independent_references=True)


def main():
    torch.set_num_threads(2)
    result = dict(status='PASS', scope='CPU synthetic only; no NIH images or formal training',
                  math=math_checks(), statistics=statistics_checks(),
                  engine=engine_checks(), data_metrics=data_and_metrics_checks(),
                  entrypoint=entrypoint_check())
    assert not torch.cuda.is_initialized()
    result['cuda_initialized'] = False
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
