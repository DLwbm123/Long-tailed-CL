"""Train-only compatibility fixtures for the approved recurrent RFMiD plan."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from check_rfmid_protocol import TinyRegions, check, permission_fixture, report_fixture
from ml_data import MultiLabelImages
from prepare_rfmid_ml import BENCHMARK_LABELS
from rfmid_protocol import read_mapped, recurrent_plan
from run_multilabel import Trainer
from run_rfmid import validate_config, evaluation_barrier


def run(config):
    started = time.monotonic()
    torch.set_num_threads(4)
    root = Path(config['root'])
    rows = read_mapped(Path(config['data_root']) / 'labels29/train.csv', 1920)
    proposal = json.loads(Path(config['proposal']).read_text())
    variant = config['worker_config']['partition_variant']
    plan, support = recurrent_plan(rows, proposal, variant)
    design = config['design']['scheme']
    for task, stage in zip(support['tasks'], design['stages']):
        for label in task['visible_labels']:
            assert task['current_positive'][label] == stage['positive'][label]
            assert task['current_negative'][label] == stage['negative'][label]
    assert support['old_positive_occurrences'] == sum(t['old_positive_occurrences'] for t in design['stages'])
    assert support['observed_negative_occurrences'] == 34010
    assert sum(t['images_with_old_positive'] for t in support['tasks']) == sum(t['images_with_old_positive'] for t in design['stages'])
    for mutate in ('duplicate', 'group', 'budget', 'future_positive'):
        bad = copy.deepcopy(plan)
        if mutate == 'duplicate': bad['tasks'][0]['train_ids'][0] = bad['tasks'][0]['train_ids'][1]
        elif mutate == 'group': bad['tasks'][0]['introduced_labels'].reverse()
        elif mutate == 'budget': bad['tasks'][0]['expected_steps'] = 13
        else:
            by_id = {r['ID']: r for r in rows}
            j = next(j for j,i in enumerate(bad['tasks'][4]['train_ids']) if any(int(by_id[i][k]) for k in bad['tasks'][4]['introduced_labels']))
            bad['tasks'][0]['train_ids'][0], bad['tasks'][4]['train_ids'][j] = bad['tasks'][4]['train_ids'][j], bad['tasks'][0]['train_ids'][0]
        try: recurrent_plan(rows, bad, variant)
        except ValueError: pass
        else: raise AssertionError('Changed frozen plan was accepted: ' + mutate)
    spec = importlib.util.spec_from_file_location('original_native_rfmid', config['native_trainer'])
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    torch.manual_seed(74002)
    initial = TinyRegions()
    trainers = [Trainer(copy.deepcopy(initial), 29), Trainer(copy.deepcopy(initial), 29, local_transport=True),
                native.Trainer(copy.deepcopy(initial), 29)]
    by_id = {r['ID']: r for r in rows}
    maximum = 0.
    for task in plan['tasks']:
        generator = torch.Generator().manual_seed(74002 + task['task'])
        x = torch.randn(384, 5, 3, generator=generator)
        raw = torch.tensor([[int(by_id[i][k]) for k in BENCHMARK_LABELS] for i in task['train_ids']], dtype=torch.float32)
        mask = torch.tensor([k in task['visible_labels'] for k in BENCHMARK_LABELS]).expand_as(raw)
        y = torch.where(mask, raw, 0.)
        loader = DataLoader(TensorDataset(x, y, mask), batch_size=64, shuffle=False)
        if task['task'] == 1:
            for local in (False, True):
                t = Trainer(copy.deepcopy(initial), 29, local_transport=local)
                before = copy.deepcopy(t.encoder.state_dict())
                pre = t.fit_task(loader, loader, 2, preflight=True)
                assert pre['optimizer_updates'] == t.steps == pre['policy_updates'] == 0 and pre['gradient_norm'] > 0
                assert all(torch.equal(v, t.encoder.state_dict()[k]) for k, v in before.items())
        records = [t.fit_task(loader, loader, 2) for t in trainers]
        assert all(t.steps == 12 * task['task'] for t in trainers)
        for key, value in trainers[0].encoder.state_dict().items():
            assert torch.equal(value, trainers[2].encoder.state_dict()[key])
        assert torch.equal(trainers[0].head, trainers[2].head)
        assert all(np.array_equal(trainers[0].bank[k], trainers[2].bank[k]) for k in trainers[0].bank)
        expected = support['tasks'][task['task'] - 1]['cumulative_observed_counts']
        for trainer in trainers:
            for index, label in enumerate(BENCHMARK_LABELS):
                assert trainer.bank['n'][index].tolist() == expected.get(label, [0, 0])
        assert np.array_equal(trainers[1].bank['n'], trainers[1].local['n'])
        assert records[1]['local_activation']['extra_columns'] == 8 * len(task['visible_labels'])
        assert records[1]['local_activation']['prior_delta_norm'] > 0
        for record in records:
            assert len(record['updates']) == 12 and all(not r['skipped'] and r['reward_coefficient'] == 0 for r in record['updates'])
            maximum = max(maximum, *(v for head in ('start_head', 'final_head') for v in record[head]['residuals'] if v is not None))
        assert maximum <= 1e-8
    restored = Trainer(copy.deepcopy(initial), 29, local_transport=True)
    restored.load_state_dict(trainers[1].state_dict())
    assert torch.equal(restored.head, trainers[1].head) and np.array_equal(restored.local['means'], trainers[1].local['means'])
    first = plan['tasks'][0]['train_ids'][0]
    mapped = {first: dict(image='images/train/' + first + '.png', official_split='train_val',
                         target=[int(by_id[first][k]) for k in BENCHMARK_LABELS], observed=[True] * 29)}
    fixture = MultiLabelImages(mapped, config['data_root'], [first], plan['tasks'][0]['visible_labels'],
                              lambda image: torch.tensor(list(image.resize((2, 2)).getdata())), labels=BENCHMARK_LABELS, training=True)
    image, target, observed = fixture[0]
    assert image.shape == (4, 3) and torch.equal(target[~observed], torch.zeros_like(target[~observed]))
    c = dict(config['worker_config'], operation='train')
    validate_config(c)
    mutations = [('expected_steps', 66), ('rfmid_recurrent_authorized', False), ('evaluation_split', 'official_test')]
    if variant == 'negative_coverage_v3':
        mutations += [('rfmid_negative_coverage_authorized', False)]
        # Earlier science sources remain valid and cannot silently adopt v3.
        old_proposal = json.loads(Path(config['previous_plan']).read_text())
        _, old_support = recurrent_plan(rows, old_proposal)
        assert old_support == json.loads(Path(config['previous_support']).read_text())
        assert support['normal_per_task'] == [81, 80, 80, 80, 80]
        try: recurrent_plan(rows, proposal)
        except ValueError: pass
        else: raise AssertionError('Coverage plan accepted as v2')
    for key, value in mutations:
        bad = dict(c, **{key: value})
        try: validate_config(bad)
        except ValueError: pass
        else: raise AssertionError('Recurrent permission or budget escaped')
    import tempfile
    with tempfile.TemporaryDirectory(dir=root) as folder:
        p = Path(folder); (p / 'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='EVALUATE', failures=[], training_sealed_at=1)))
        for arm in ('base', 'local_transport'):
            out = p / 'runs' / ('RFMiD_' + arm); out.mkdir(parents=True)
            (out / 'STATUS.json').write_text(json.dumps(dict(status='TRAINED', steps=60, policy_updates=0, trained_stages=5)))
        gate = dict(c, evaluation_gate=str(p / 'PROGRAM_STATE.json'))
        evaluation_barrier(gate, plan)
        (p / 'runs/RFMiD_base/STATUS.json').write_text(json.dumps(dict(status='TRAINED', steps=66, policy_updates=0, trained_stages=5)))
        try: evaluation_barrier(gate, plan)
        except ValueError: pass
        else: raise AssertionError('Old budget accepted as new paired training')
    old_checks = dict(protocol=check(), permission=permission_fixture(), report=report_fixture())
    for name, value in (('TASK_PLAN.private.json', plan), ('TASK_SUPPORT_AUDIT.json', support)):
        p = root / name; assert not p.exists(); p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return dict(status='PASS', cpu_pipeline_seconds=time.monotonic() - started, gpu_work=False, test_accessed=False,
                validation_images_or_targets_read=False, real_train_image_loader_checked=True,
                five_task_native_base_trajectory_exact=True, binary_counts_match_frozen_support=True,
                candidate_positive_negative_prior_compatible=True, checkpoint_roundtrip=True,
                synthetic_cpu_fixture_optimizer_steps_per_arm=60, formal_adapter_updates=0,
                zero_optimizer_fixture_encoder_unchanged=True, new_and_legacy_permission_barriers_checked=True,
                malformed_frozen_plan_rejected=True, max_fixture_solve_residual=maximum, regressions=old_checks,
                support=support)


if __name__ == '__main__':
    print(json.dumps(run(json.load(sys.stdin)), indent=2, allow_nan=False))
