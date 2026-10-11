"""Two-operation RFMiD worker; official validation requires both trained arms."""
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_data import MultiLabelImages
from ml_metrics import evaluate, groups
from module_additions import local_parts  # Pin frozen pooling before ApartFeatures adds its legacy import path.
from prepare_rfmid_ml import BENCHMARK_LABELS
from rfmid_protocol import build_plan, read_mapped
from run_multilabel import ApartFeatures, Trainer, extract


def validate_config(config):
    if config.get('rfmid_multilabel_authorized') is not True or config.get('name') != 'RFMiD':
        raise ValueError('Explicit RFMiD multi-label authorization required')
    fixed = dict(evaluation_split='official_validation', seed=74002, epochs=2, batch_size=64,
                 workers=4, lr=.0003, expected_steps=66, policy_updates=0, threshold=.5)
    if any(config.get(k) != v for k, v in fixed.items()) or config.get('test_accessed') is not False:
        raise ValueError('RFMiD frozen settings or test permission differ')
    if config.get('module') not in ('base', 'local_transport') or config.get('operation') not in ('train', 'evaluate'):
        raise ValueError('Unapproved RFMiD module or operation')
    if config.get('labels') != list(BENCHMARK_LABELS) or 'checkpoint_root' in config:
        raise ValueError('RFMiD label view differs or external checkpoint requested')
    if config.get('preflight') and config['operation'] != 'train':
        raise ValueError('Preflight cannot open evaluation')


def evaluation_barrier(config, plan):
    gate = json.loads(Path(config['evaluation_gate']).read_text())
    if gate.get('phase') != 'EVALUATE' or gate.get('failures') or not gate.get('training_sealed_at'):
        raise ValueError('RFMiD paired training barrier is not sealed')
    root = Path(config['evaluation_gate']).parent
    for arm in ('base', 'local_transport'):
        status = json.loads((root / 'runs' / ('RFMiD_' + arm) / 'STATUS.json').read_text())
        if status.get('status') not in ('TRAINED', 'RUNNING', 'COMPLETE') or status.get('steps') != 66 or status.get('policy_updates') != 0:
            raise ValueError('Both RFMiD arms must have the exact completed training budget')
        if status.get('trained_stages') != len(plan['tasks']):
            raise ValueError('Incomplete RFMiD trained checkpoints')


def run(config):
    validate_config(config)
    root, output = Path(config['data_root']), Path(config['output'])
    data_receipt = json.loads((root / 'READY.json').read_text())
    if data_receipt.get('test_accessed') is not False or data_receipt.get('label_mapping_frozen') is not True:
        raise ValueError('RFMiD data receipt reports test access')
    rows = read_mapped(root / 'labels29/train.csv', 1920)
    plan, support = build_plan(rows)
    if plan != json.loads((root / 'TASK_PLAN.private.json').read_text()) or support['expected_steps_per_arm'] != 66:
        raise ValueError('RFMiD train-only plan differs from frozen preparation')
    if config['operation'] == 'train':
        if output.exists():
            raise ValueError('No RFMiD overwrite or automatic retry')
        output.mkdir(parents=True)
    elif not (output / 'STATUS.json').is_file():
        raise ValueError('RFMiD trained output is missing')
    def save(name, value):
        path = output / name
        tmp = path.with_suffix('.part')
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
        tmp.replace(path)
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = 'cuda:0'
    encoder = ApartFeatures(config['legacy_repo'], config['weight'], 29, device, seed)
    trainer = Trainer(encoder, 29, 'A', seed, device, local_transport=config['module'] == 'local_transport')
    status = dict(status='RUNNING', phase='PREFLIGHT' if config.get('preflight') else 'TRAIN',
                  steps=0, policy_updates=0, trained_stages=0, test_accessed=False)
    if config['operation'] == 'evaluate':
        status = json.loads((output / 'STATUS.json').read_text())
    def wall():
        if time.time() >= config['deadline']:
            raise RuntimeError('RFMiD CAMPAIGN_DEADLINE')
    def step():
        wall()
        if trainer.steps >= 66:
            raise RuntimeError('RFMiD adapter step budget exceeded')
        status.update(steps=trainer.steps, phase='PREFLIGHT_BACKWARD' if config.get('preflight') else 'TRAIN')
        save('STATUS.json', status)
    from run_medical_v2 import transform
    train_rows = {r['ID']: dict(image='images/train/' + r['ID'] + '.png', official_split='train_val',
                               target=[int(r[k]) for k in BENCHMARK_LABELS], observed=[True] * 29) for r in rows}
    def loader(mapped, ids, visible, training, task):
        data = MultiLabelImages(mapped, root, ids, visible, transform(training), labels=BENCHMARK_LABELS,
                                training=training, seed=seed + task * 100003)
        return DataLoader(data, batch_size=64, shuffle=training, num_workers=0 if config.get('preflight') else 4,
                          generator=torch.Generator().manual_seed(seed + task * 2003))
    try:
        wall()
        if config['operation'] == 'train':
            save('STATUS.json', status)
            diagnostics = []
            for task in plan['tasks']:
                wall(); status.update(task=task['task'], phase='TASK_START_FEATURES'); save('STATUS.json', status)
                result = trainer.fit_task(loader(train_rows, task['train_ids'], task['visible_labels'], False, task['task']),
                                          loader(train_rows, task['train_ids'], task['visible_labels'], True, task['task']),
                                          2, .0003, step, preflight=bool(config.get('preflight')), before_extract=wall)
                if config.get('preflight'):
                    if not result['preflight'] or trainer.steps != 0:
                        raise ArithmeticError('RFMiD zero-update preflight failed')
                    save('PREFLIGHT.json', dict(status='PASS', **result))
                    status.update(status='COMPLETE', phase='PREFLIGHT_PASS', steps=0)
                    save('STATUS.json', status); return
                records = result['updates']
                if len(records) != task['expected_steps'] or any(r['skipped'] or r['reward_coefficient'] != 0 for r in records):
                    raise ArithmeticError('RFMiD task update count or non-RL objective differs')
                expected = support['tasks'][task['task'] - 1]['cumulative_observed_counts']
                for j, label in enumerate(BENCHMARK_LABELS):
                    if label in expected and trainer.bank['n'][j].tolist() != expected[label]:
                        raise ArithmeticError('RFMiD observed positive/negative denominators differ')
                for epoch in (1, 2):
                    selected = [r for r in records if r['epoch'] == epoch]
                    diagnostics.append(dict(task=task['task'], epoch=epoch, steps=len(selected),
                        mean_loss=float(np.mean([r['loss'] for r in selected])),
                        mean_bce=float(np.mean([r['bce'] for r in selected])),
                        mean_fd=float(np.mean([r['fd'] for r in selected])),
                        max_pre_clip_norm=max(r['pre_clip_norm'] for r in selected),
                        optimizer_steps=len(selected), policy_updates=0))
                diagnostics.append(dict(task=task['task'], boundary=True, **result['local_activation'],
                    **result['local_transport'], global_shift_norm=result['translation']['shift_norm'],
                    max_solve_residual=max(v for head in ('start_head', 'final_head') for v in result[head]['residuals'] if v is not None)))
                torch.save(trainer.state_dict(), output / ('stage_' + str(task['task']) + '.pt'))
                status.update(steps=trainer.steps, trained_stages=task['task']); save('STATUS.json', status)
            if trainer.steps != 66:
                raise ArithmeticError('RFMiD final adapter budget differs')
            save('diagnostics.json', diagnostics)
            status.update(status='TRAINED', phase='WAIT_PAIRED_TRAINING_BARRIER'); save('STATUS.json', status); return
        evaluation_barrier(config, plan)
        val = read_mapped(root / 'labels29/val.csv', 640)
        val_rows = {r['ID']: dict(image='images/val/' + r['ID'] + '.png', official_split='train_val',
                                target=[int(r[k]) for k in BENCHMARK_LABELS], observed=[True] * 29) for r in val}
        status.update(status='RUNNING', phase='EVALUATE'); save('STATUS.json', status)
        reports = []
        for task in plan['tasks']:
            wall()
            state = torch.load(output / ('stage_' + str(task['task']) + '.pt'), map_location='cpu', weights_only=False)
            trainer.load_state_dict(state)
            expected_steps = sum(t['expected_steps'] for t in plan['tasks'][:task['task']])
            if trainer.steps != expected_steps or trainer.task != task['task']:
                raise ValueError('RFMiD source stage budget differs')
            z, y, mask = extract(trainer.encoder, loader(val_rows, [r['ID'] for r in val], task['visible_labels'], False, task['task']),
                                 device, before_batch=wall)
            logits = np.c_[z, np.ones(len(z))] @ trainer.head.cpu().numpy()
            report = evaluate(logits, y, mask, BENCHMARK_LABELS, .5)
            visible = set(task['visible_labels'])
            group_labels = {k: [c for c in v if c in visible] for k, v in plan['metric_groups'].items()}
            group_labels.update(old=[c for c in task['visible_labels'] if c not in task['introduced_labels']], new=task['introduced_labels'])
            report.update(task=task['task'], visible_labels=task['visible_labels'], groups=groups(report, group_labels))
            reports.append(report)
        save('metrics.json', dict(dataset='RFMiD', module=config['module'], stages=reports,
                                 evaluation_split='official_validation', test_accessed=False, threshold=.5))
        status.update(status='COMPLETE', phase='AWAIT_PUBLIC_DELIVERY', steps=66, trained_stages=5)
        save('STATUS.json', status)
    except BaseException as exc:
        status.update(status='INCOMPLETE', error=str(exc), steps=trainer.steps)
        save('STATUS.json', status)
        raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
