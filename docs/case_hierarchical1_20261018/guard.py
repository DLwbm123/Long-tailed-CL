"""Detached execution with a hard wall limit; never retries the worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
root=Path(cfg['output']).parent
start=time.time()
with (root/'stdout.private.log').open('xb') as log:
    worker=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    identity=dict(pid=worker.pid,started=start,entry=str(Path(__file__).with_name('run.py')),
                  start_ticks=Path(f'/proc/{worker.pid}/stat').read_text().split()[21],
                  gpu=os.environ['CUDA_VISIBLE_DEVICES'],attempts=1)
    write(root/'launch.private.json',identity)
    try:
        rc=worker.wait(timeout=cfg['max_wall_seconds']-30)
        reason='worker_exit'
    except subprocess.TimeoutExpired:
        reason='hard_wall_limit'
        os.killpg(worker.pid,signal.SIGTERM)
        try:rc=worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid,signal.SIGKILL)
            rc=worker.wait(timeout=10)
    write(root/'terminal.private.json',dict(**identity,finished=time.time(),
          elapsed_seconds=time.time()-start,returncode=rc,reason=reason))
    if rc!=0:
        state=Path(cfg['output'])/'status.private.json'
        old=json.loads(state.read_text()) if state.exists() else {}
        if old.get('state')!='FAILED':
            old.update(state='STOPPED',reason=reason,returncode=rc,updated=time.time())
            write(state,old)
