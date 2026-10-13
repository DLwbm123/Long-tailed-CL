"""Four one-shot evaluation workers, two GPU01 slots, no training or retry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from run_module_cycle import save


def run(c):
    expected=[ds+'_'+arm for ds in ('ISIC','HK') for arm in ('lca_supported','local_transport_lca_supported')]
    if c['gpus']!=[0,1] or [j['job_name'] for j in c['jobs']]!=expected:
        raise ValueError('Exactly four frozen checkpoint diagnostics on GPU01')
    root=Path(c['root']);begun=time.monotonic();lock=threading.Lock()
    s=dict(status='RUNNING',phase='FIXED_STATE_DIAGNOSTIC',active={},costs=[],failures=[],started=time.time())
    def write():
        s['elapsed_seconds']=time.monotonic()-begun;save(root/'PROGRAM_STATE.json',s)
    def execute(index,slot):
        job=expected[index];start=time.monotonic();code=-1
        try:
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(slot),Q165_JOB=str(index))
            with (root/'logs'/(job+'.private.log')).open('x') as log:
                p=subprocess.Popen([sys.executable,'-u',c['worker_entry']],stdout=log,stderr=subprocess.STDOUT,env=env)
                with lock:
                    s['active'][str(slot)]=dict(job=job,operation='checkpoint_switch',pid=p.pid);write()
                try:code=p.wait(timeout=max(1,c['deadline']-time.time()))
                except subprocess.TimeoutExpired:
                    p.terminate();p.wait();code=124
        finally:
            cost=dict(job=job,operation='checkpoint_switch',gpu=slot,returncode=code,residence_seconds=time.monotonic()-start)
            with lock:
                s['active'].pop(str(slot),None);s['costs'].append(cost)
                if code:s['failures'].append(cost)
                write()
    write()
    for batch in (0,2):
        if s['failures'] or time.time()>=c['deadline']:break
        threads=[threading.Thread(target=execute,args=(batch+slot,slot)) for slot in (0,1)]
        for t in threads:t.start()
        for t in threads:t.join()
    s.update(status='COMPLETE' if len(s['costs'])==4 and not s['failures'] else 'INCOMPLETE',
        phase='AWAIT_PUBLIC_DELIVERY',ended=time.time(),whole_round_wall_seconds=time.time()-c['launched_at'])
    write()


if __name__=='__main__':run(json.loads(Path(os.environ['Q128_CONFIG']).read_text()))
