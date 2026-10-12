"""Two evaluation-only workers; no training phase, no retry and GPU01 only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from run_module_cycle import save


def run(c):
    if c['gpus']!=[0,1] or [j['original_job']['dataset'] for j in c['jobs']]!=['ISIC','HK']:
        raise ValueError('Only paired ISIC/HK GPU01 diagnostic authorized')
    root=Path(c['root']);begun=time.monotonic();lock=threading.Lock()
    s=dict(status='RUNNING',phase='DIAGNOSTIC_VALIDATION',active={},costs=[],failures=[],started=time.time())
    def write():
        s['elapsed_seconds']=time.monotonic()-begun;save(root/'PROGRAM_STATE.json',s)
    def execute(slot):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(slot),Q157_SLOT=str(slot))
        start=time.monotonic()
        name=c['jobs'][slot]['original_job']['dataset']
        with (root/'logs'/(name+'.private.log')).open('x') as log:
            p=subprocess.Popen([sys.executable,'-u','/tmp/q157w.py'],stdout=log,stderr=subprocess.STDOUT,env=env)
            with lock:s['active'][str(slot)]=dict(job=name,operation='diagnostic_evaluate',pid=p.pid);write()
            code=p.wait()
        cost=dict(job=name,operation='diagnostic_evaluate',gpu=slot,returncode=code,residence_seconds=time.monotonic()-start)
        with lock:
            s['active'].pop(str(slot));s['costs'].append(cost)
            if code:s['failures'].append(cost)
            write()
    write();threads=[threading.Thread(target=execute,args=(i,)) for i in (0,1)]
    for t in threads:t.start()
    for t in threads:t.join()
    if len(s['costs'])!=2:
        s['failures'].append(dict(reason='DIAGNOSTIC_WORKER_DISPATCH_INCOMPLETE'))
    s.update(status='INCOMPLETE' if s['failures'] else 'COMPLETE',phase='AWAIT_PUBLIC_DELIVERY',ended=time.time(),whole_round_wall_seconds=time.time()-c['launched_at'])
    write()


if __name__=='__main__':
    run(json.loads(Path(os.environ['Q128_CONFIG']).read_text()))
