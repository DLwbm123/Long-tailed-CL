"""Two frozen contrast arms, admitted only after the protected drift processes end."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_module_cycle import run, save
from run_rfmid import CONTRAST_MODULES, validate_config


def validate(config):
    if (config['gpus'] != [0, 1] or len(config['jobs']) != 2 or len(config['preflights']) != 2 or
            {j['module'] for j in config['jobs']} != set(CONTRAST_MODULES) or
            config.get('strict_preflight_receipts') is not True or config['max_wall_seconds'] != 86400):
        raise ValueError('Frozen contrast contribution scope differs')
    for job, pre in zip(config['jobs'], config['preflights']):
        validate_config(dict(job, operation='train'))
        validate_config(dict(pre, operation='train'))
        if (not pre.get('preflight') or job.get('preflight') or
                job['job_name'] != 'RFMiD_' + job['module'] or
                pre['job_name'] != job['job_name'] + '_preflight'):
            raise ValueError('Independent preflight required')
        for key in job:
            if key not in ('output', 'job_name') and pre.get(key) != job[key]:
                raise ValueError('Preflight/final science config differs: ' + key)
    if len({j['output'] for j in config['jobs'] + config['preflights']}) != 4:
        raise ValueError('Fresh independent outputs required')


def main(config):
    validate(config)
    root, started = Path(config['root']), time.time()
    state = dict(status='RUNNING', phase='WAITING_PROTECTED_DRIFT', started=started,
                 active={}, completed=[], failures=[], costs=[])
    save(root / 'PROGRAM_STATE.json', state)
    def deadline():
        if time.time() >= config['deadline']:
            raise TimeoutError('Campaign deadline including resource wait; no retry')
    while True:
        deadline()
        previous = json.loads(Path(config['wait_for_program']).read_text())
        if previous['status'] in ('COMPLETE', 'INCOMPLETE') and not previous.get('active'):
            # An empty stale active map alone does not prove all GPU workers ended.
            old_config = json.loads((Path(config['wait_for_program']).parent / 'CONFIG.private.json').read_text())
            ps = subprocess.check_output(['ps', '-eo', 'args='], text=True).splitlines()
            if not any(old_config['worker_entry'] in line for line in ps):
                break
        time.sleep(15)
    mount = subprocess.check_output(['findmnt', '-T', str(root), '-n', '-o', 'FSTYPE'], text=True).strip()
    if not mount.startswith('nfs') or shutil.disk_usage(root).free < 40 * 1024**3:
        raise ValueError('NAS admission failed')
    probe = root / ('.probe_' + str(time.time_ns()))
    probe.write_text('probe'); assert probe.read_text() == 'probe'; probe.unlink()
    while True:
        deadline()
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                        '--format=csv,noheader,nounits'], text=True).splitlines()
        free = {int(v.split(',')[0]): int(v.split(',')[1]) for v in rows}
        if all(free[g] >= 32768 for g in (0, 1)):
            break
        state['phase'] = 'WAITING_GPU_01_MEMORY'; save(root / 'PROGRAM_STATE.json', state)
        time.sleep(30)
    save(root / 'GPU_LAUNCH_CHECK.json', dict(gpus=[0, 1], free_mib={g: free[g] for g in (0, 1)},
                                            mount=mount, checked_at=time.time()))
    run(config)
    state = json.loads((root / 'PROGRAM_STATE.json').read_text())
    state.update(resource_wait_seconds=state['started']-started, whole_round_wall_seconds=time.time()-started)
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
