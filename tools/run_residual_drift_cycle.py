"""Frozen four new trajectories; wait for the protected queue before GPU dispatch."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_module_cycle import run, save


def validate(config):
    if config['gpus'] != [0, 1] or len(config['jobs']) != 4 or len(config['preflights']) != 4:
        raise ValueError('Frozen four-trajectory GPU01 scope differs')
    expected = {(ds, arm) for ds in ('ISIC', 'HK') for arm in ('base', 'local_transport')}
    if {(j['name'], j['module']) for j in config['jobs']} != expected:
        raise ValueError('Contribution cells differ')
    for j, p in zip(config['jobs'], config['preflights']):
        if (j['residual_drift_weight'] != .1 or j['residual_drift_authorized'] is not True or
                j['residual_drift_version'] != 'residual_drift_v1' or
                j['evaluation_split'] != 'development_validation' or
                j['expected_steps'] != {'ISIC': 468, 'HK': 176}[j['name']] or
                j['seed'] != 74002 or j['split_seed'] != 74002 or j['epochs'] != 2 or
                j['batch_size'] != 64 or j['workers'] != 4 or j['lr'] != .0003 or
                j['method'] != 'static_pc' or j.get('checkpoint_root') or
                p['name'] != j['name'] or p['module'] != j['module'] or
                p['residual_drift_weight'] != .1 or p['expected_steps'] != 0 or
                p.get('preflight') is not True or p['workers'] != 0):
            raise ValueError('Frozen drift parameters/budgets differ')
        sizes = {'ISIC': [4, 2, 2], 'HK': [13, 2, 2, 2, 2, 2]}
        order = {'ISIC': [4, 7, 5, 1, 2, 6, 0, 3],
                 'HK': [4, 5, 6, 22, 20, 1, 2, 15, 14, 12, 10, 11, 19, 16, 7, 0, 17, 9, 18, 13, 21, 8, 3]}
        if j['task_sizes'] != sizes[j['name']] or j['order'] != order[j['name']]:
            raise ValueError('Frozen class order/task sizes differ')
        for key in ('seed', 'split_seed', 'epochs', 'batch_size', 'lr', 'method', 'order',
                    'task_sizes', 'residual_drift_authorized', 'residual_drift_version',
                    'evaluation_split', 'manifest', 'images', 'weight', 'legacy_repo'):
            if p[key] != j[key]:
                raise ValueError('Preflight scientific config differs')
    if len({j['output'] for j in config['jobs'] + config['preflights']}) != 8:
        raise ValueError('Fresh contribution outputs required')


def main(config):
    validate(config)
    root = Path(config['root']); started = time.time()
    state = dict(status='RUNNING', phase='WAITING_PROTECTED_RFMID', started=started,
                 active={}, completed=[], failures=[], costs=[])
    save(root / 'PROGRAM_STATE.json', state)
    def deadline():
        if time.time() >= config['deadline']:
            raise TimeoutError('Original campaign deadline; no automatic retry')
    while True:
        deadline()
        previous = json.loads(Path(config['wait_for_program']).read_text())
        if previous['status'] in ('COMPLETE', 'INCOMPLETE') and not previous.get('active'):
            break
        time.sleep(15)
    mount = subprocess.check_output(['findmnt', '-T', str(root), '-n', '-o', 'FSTYPE'], text=True).strip()
    if not mount.startswith('nfs') or shutil.disk_usage(root).free < 30 * 1024**3:
        raise ValueError('NAS admission failed')
    probe = root / ('.gpu_probe_' + str(time.time_ns()))
    probe.write_text('probe'); assert probe.read_text() == 'probe'; probe.unlink()
    while True:
        deadline()
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                       '--format=csv,noheader,nounits'], text=True).splitlines()
        available = {int(v.split(',')[0]): int(v.split(',')[1]) for v in rows}
        if all(available[g] >= 32768 for g in config['gpus']):
            break
        state['phase'] = 'WAITING_GPU_01_MEMORY'; save(root / 'PROGRAM_STATE.json', state)
        time.sleep(30)
    save(root / 'GPU_LAUNCH_CHECK.json', dict(gpus=[0, 1], free_mib={g: available[g] for g in (0, 1)},
                                            mount=mount, checked_at=time.time()))
    run(config)
    state = json.loads((root / 'PROGRAM_STATE.json').read_text())
    state['resource_wait_seconds'] = state['started'] - started
    state['whole_round_wall_seconds'] = time.time() - started
    save(root / 'PROGRAM_STATE.json', state)


if __name__ == '__main__':
    config = json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    try:
        main(config)
    except BaseException as exc:
        path = Path(config['root']) / 'PROGRAM_STATE.json'
        state = json.loads(path.read_text()) if path.exists() else dict(active={}, costs=[], failures=[])
        state.update(status='INCOMPLETE', coordinator_error=str(exc), ended=time.time())
        save(path, state)
        raise
