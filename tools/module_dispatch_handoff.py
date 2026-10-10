"""Adopt one externally dispatched, unchanged trajectory without training it twice."""
import json
import os
from pathlib import Path
import subprocess
import time

from run_module_cycle import save


def wait_external(job):
    root=Path(job['output']).parent.parent
    assignment=json.loads((root/'DISPATCH_HANDOFF.private.json').read_text())
    if assignment['job']!=job:
        raise ValueError('External trajectory differs from the frozen queued configuration')
    started=time.monotonic()
    save(root/'DISPATCH_WAIT.private.json',dict(status='WAITING',pid=os.getpid(),gpu_compute=False))
    while True:
        if time.time()>=job['original_deadline']:
            raise TimeoutError('Original campaign deadline')
        p=root/'EXTERNAL_TRAIN.private.json'
        if p.exists():
            state=json.loads(p.read_text())
            if state['status'] in ('COMPLETE','INCOMPLETE'):
                code=state['cost']['returncode']
                if code==0:
                    status=json.loads((Path(job['output'])/'STATUS.json').read_text())
                    if (status['status']!='TRAINED' or status['steps']!=job['expected_steps'] or
                            status['policy_updates']!=0):
                        raise ValueError('External trajectory did not finish its frozen budget')
                save(root/'DISPATCH_WAIT.private.json',dict(status='ADOPTED' if code==0 else 'FAILED',
                    pid=os.getpid(),gpu_compute=False,residence_seconds=time.monotonic()-started))
                return code
        time.sleep(.5)


def run(assignment):
    root=Path(assignment['root']);job=assignment['job'];gpu=assignment['gpu']
    if gpu not in (0,1,2) or assignment['source_commit']!=job['source_commit']:
        raise ValueError('Resource assignment or scientific source differs')
    if Path(job['output']).exists():
        raise ValueError('External trajectory already started; no retry or overwrite')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    started=time.monotonic()
    with (root/'logs'/'external_candidate_train.private.log').open('x') as log:
        p=subprocess.Popen([os.sys.executable,'-u',assignment['worker_entry']],stdin=subprocess.PIPE,
                           stdout=log,stderr=subprocess.STDOUT,env=env,text=True)
        save(root/'EXTERNAL_TRAIN.private.json',dict(status='RUNNING',pid=p.pid,gpu=gpu,
             job=job['job_name'],started_at=time.time(),source_commit=job['source_commit']))
        p.communicate(json.dumps(job))
    cost=dict(job=job['job_name'],operation='train',gpu=gpu,returncode=p.returncode,
              residence_seconds=time.monotonic()-started)
    save(root/'EXTERNAL_TRAIN.private.json',dict(status='COMPLETE' if p.returncode==0 else 'INCOMPLETE',
         pid=p.pid,gpu=gpu,job=job['job_name'],ended_at=time.time(),source_commit=job['source_commit'],cost=cost))


if __name__=='__main__':
    run(json.loads(Path(os.environ['Q137_HANDOFF_CONFIG']).read_text()))
