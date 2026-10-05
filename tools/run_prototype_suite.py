"""Sequential bounded suite with neutral child arguments and fail-stop receipts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_prototype_single import save


def run(config):
    root = Path(config['suite_root'])
    status_path = root / 'STATUS.json'
    if status_path.exists():
        raise ValueError('Existing suite cannot be retried automatically')
    started = time.monotonic()
    state = dict(status='RUNNING', pid=os.getpid(), started=time.time(), completed=[])
    save(status_path, state)
    try:
        for job in config['jobs']:
            remaining = config['total_wall_seconds'] - (time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('Suite budget exhausted')
            job = dict(job, max_wall_seconds=remaining)
            name = Path(job['output']).name
            with (root / (name + '.log')).open('w') as log:
                child = subprocess.Popen([sys.executable, '-u', config['neutral_worker']], stdin=subprocess.PIPE,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(active=name, child_pid=child.pid)
                save(status_path, state)
                try:
                    child.communicate(json.dumps(job).encode(), timeout=remaining)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    raise
            if child.returncode:
                raise RuntimeError(f'{name}: exit {child.returncode}; no automatic retry')
            receipt = json.loads((Path(job['output']) / 'STATUS.json').read_text())
            if receipt['status'] != 'COMPLETE':
                raise RuntimeError('Missing completion receipt')
            state['completed'].append(dict(name=name, **receipt))
            save(status_path, state)
        state.update(status='COMPLETE', elapsed_seconds=time.monotonic()-started, ended=time.time())
    except BaseException as exc:
        state.update(status='INCOMPLETE', error=str(exc), elapsed_seconds=time.monotonic()-started)
        save(status_path, state); raise
    save(status_path, state)


if __name__ == '__main__':
    run(json.load(sys.stdin))
