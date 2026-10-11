"""Five synthetic tasks through the actual versioned Trainer, without GPU/images."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from check_rfmid_protocol import TinyRegions, permission_fixture
from prepare_rfmid_ml import BENCHMARK_LABELS
from run_multilabel import Trainer
from run_rfmid import CONTRAST_MODULES, validate_config, evaluation_barrier
from run_rfmid_contrast_cycle import validate as validate_cycle


def check(native_path, config):
    started = time.monotonic()
    torch.set_num_threads(1)
    validate_cycle(config)
    spec = importlib.util.spec_from_file_location('frozen_v3_trainer', native_path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    torch.manual_seed(74002)
    encoder = TinyRegions()
    trainers = [Trainer(copy.deepcopy(encoder), 29), old.Trainer(copy.deepcopy(encoder), 29),
                Trainer(copy.deepcopy(encoder), 29, local_transport=True),
                old.Trainer(copy.deepcopy(encoder), 29, local_transport=True)]
    new = [Trainer(copy.deepcopy(encoder), 29, local_transport=True, contrast_prior=mode)
           for mode in CONTRAST_MODULES.values()]
    peak = 0.
    count = 0
    for task, seen in enumerate((9, 14, 19, 24, 29), 1):
        gen = torch.Generator().manual_seed(74002 + task)
        x = torch.randn(384, 5, 3, generator=gen)
        raw = torch.tensor([[(i + j + task) % (2 + j % 4) == 0 for j in range(29)]
                            for i in range(384)], dtype=torch.float32)
        mask = (torch.arange(29) < seen).expand_as(raw)
        y = torch.where(mask, raw, 0.)
        loader = DataLoader(TensorDataset(x, y, mask), batch_size=64, shuffle=False)
        if task == 1:
            for mode in CONTRAST_MODULES.values():
                trainer = Trainer(copy.deepcopy(encoder), 29, local_transport=True, contrast_prior=mode)
                before = copy.deepcopy(trainer.encoder.state_dict())
                record = trainer.fit_task(loader, loader, 2, preflight=True)
                assert record['optimizer_updates'] == trainer.steps == record['policy_updates'] == 0
                assert record['gradient_norm'] > 0 and record['activation']['extra_columns'] == 4 * seen
                assert all(torch.equal(v, trainer.encoder.state_dict()[k]) for k, v in before.items())
        for trainer in trainers + new:
            record = trainer.fit_task(loader, loader, 2)
            assert len(record['updates']) == 12 and trainer.steps == task * 12
            expected = np.array([[int((mask[:, j] & (y[:, j] == sign)).sum()) for sign in (0, 1)] for j in range(29)])
            count += trainer is new[0]
            if task == 1: trainer._fixture_counts = expected
            else: trainer._fixture_counts += expected
            np.testing.assert_array_equal(trainer.bank['n'], trainer._fixture_counts)
            if trainer.local is not None: np.testing.assert_array_equal(trainer.local['n'], trainer.bank['n'])
            peak = max(peak, max(v for head in ('start_head', 'final_head') for v in record[head]['residuals'] if v is not None))
            if trainer in new:
                a = record['local_activation']
                assert a['extra_columns'] == 4 * seen and 0 <= a['contrast_retained_energy'] <= a['contrast_raw_energy'] + 1e-12
                if trainer.contrast_prior == 'difference_shared_removed': assert a['contrast_shared_leakage_max'] < 1e-10
                restored = Trainer(copy.deepcopy(encoder), 29, local_transport=True, contrast_prior=trainer.contrast_prior)
                restored.load_state_dict(trainer.state_dict())
                assert torch.equal(restored.head, trainer.head)
                np.testing.assert_array_equal(restored.local['means'], trainer.local['means'])
        for current, frozen in ((trainers[0], trainers[1]), (trainers[2], trainers[3])):
            for key, value in current.encoder.state_dict().items(): assert torch.equal(value, frozen.encoder.state_dict()[key])
            assert torch.equal(current.head, frozen.head)
            for key in current.bank: np.testing.assert_array_equal(current.bank[key], frozen.bank[key])
            if current.local is not None:
                for key in current.local: np.testing.assert_array_equal(current.local[key], frozen.local[key])
    assert count == 5
    for i, trainer in enumerate(new):
        state = trainer.state_dict()
        for bad in ('missing', 'other', 'version'):
            wrong = copy.deepcopy(state)
            if bad == 'missing': wrong.pop('contrast_prior')
            elif bad == 'other': wrong['contrast_prior'] = new[1-i].contrast_prior
            else: wrong['contrast_prior_version'] = 'wrong'
            try: trainer.load_state_dict(wrong)
            except ValueError: pass
            else: raise AssertionError('Mismatched checkpoint accepted')
    permission_fixture()
    for job in config['jobs']:
        good = dict(job, operation='train'); validate_config(good)
        for key, value in [('rfmid_contrast_authorized',False),('contrast_prior_version','wrong'),
                           ('contrast_prior','wrong'),('evaluation_split','official_test'),
                           ('test_accessed',True),('expected_steps',66)]:
            bad = dict(good); bad[key] = value
            try: validate_config(bad)
            except ValueError: pass
            else: raise AssertionError('Permission or version escaped: '+key)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); (root/'runs').mkdir()
        gate = root/'PROGRAM_STATE.json'
        for job in config['jobs']:
            out = root/'runs'/('RFMiD_'+job['module']); out.mkdir()
            (out/'STATUS.json').write_text(json.dumps(dict(status='TRAINED',steps=60,policy_updates=0,trained_stages=5)))
        good = dict(config['jobs'][0], operation='evaluate',evaluation_gate=str(gate))
        for phase in ('TRAIN','EVALUATE'):
            gate.write_text(json.dumps(dict(phase=phase,failures=[],training_sealed_at=123.)))
            if phase == 'EVALUATE': evaluation_barrier(good,dict(tasks=[{}]*5))
            else:
                try: evaluation_barrier(good,dict(tasks=[{}]*5))
                except ValueError: pass
                else: raise AssertionError('Evaluation opened before both trained')
    return dict(status='PASS',five_task_native_and_legacy_local_exact=True,all_task_positive_negative_counts_exact=True,
                contrast_columns_4K=True,shared_projection_leakage_below_1e_10=True,checkpoint_version_and_mode_rejection=True,
                new_worker_authorization_and_test_rejection=True,new_paired_training_barrier=True,
                synthetic_zero_update_preflights=2,synthetic_optimizer_updates=360,formal_adapter_updates=0,
                gpu_processes=0,test_accessed=False,max_solve_residual=peak,cpu_pipeline_seconds=time.monotonic()-started)


if __name__ == '__main__':
    import sys
    c = json.load(sys.stdin)
    print(json.dumps(check(c['native_trainer'],c['cycle_config']),allow_nan=False,indent=2))
