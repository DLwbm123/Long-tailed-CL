"""Three neutral workers, fixed priority queue, and all-trained evaluation barrier."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def save(path,value):
    temporary=path.with_suffix('.part');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def run(config):
    root=Path(config['root']);start=time.monotonic();lock=threading.Lock()
    state=dict(status='RUNNING',phase='PREFLIGHT',started=time.time(),completed=[],failures=[],active={},costs=[])
    def write():
        state['elapsed_seconds']=time.monotonic()-start;save(root/'PROGRAM_STATE.json',state)
    def execute(job,gpu,operation):
        value=job.copy();value.update(operation=operation,evaluation_gate=str(root/'PROGRAM_STATE.json'))
        name=job['job_name']+'_'+operation
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        begun=time.monotonic()
        with (root/'logs'/f'{name}.private.log').open('w') as log:
            p=subprocess.Popen([sys.executable,'-u',config['worker_entry']],stdin=subprocess.PIPE,
                stdout=log,stderr=subprocess.STDOUT,env=env,text=True)
            with lock:
                state['active'][str(gpu)]=dict(job=job['job_name'],operation=operation,pid=p.pid);write()
            p.communicate(json.dumps(value))
        record=dict(job=job['job_name'],operation=operation,gpu=gpu,returncode=p.returncode,
                    residence_seconds=time.monotonic()-begun)
        with lock:
            state['active'].pop(str(gpu),None);state['costs'].append(record)
            if p.returncode:state['failures'].append(record)
            else:state['completed'].append(name)
            write()
    def phase(jobs,operation):
        queue=list(jobs)
        def worker(gpu):
            while True:
                with lock:
                    if state['failures'] or not queue:return
                    if time.time()>=config['deadline']:
                        state['failures'].append(dict(error='CAMPAIGN_DEADLINE',operation=operation));write();return
                    job=queue.pop(0)
                execute(job,gpu,operation)
        threads=[threading.Thread(target=worker,args=(gpu,)) for gpu in config['gpus']]
        for t in threads:t.start()
        for t in threads:t.join()
        return not state['failures']
    write()
    if phase(config['preflights'],'train'):
        state['phase']='TRAIN';write()
        if phase(config['jobs'],'train'):
            for job in config['jobs']:
                s=json.loads((Path(job['output'])/'STATUS.json').read_text())
                if s['status']!='TRAINED' or s['steps']!=job['expected_steps']:
                    raise ValueError('Incomplete training barrier')
            state['phase']='EVALUATE';state['training_sealed_at']=time.time();write()
            if phase(config['jobs'],'evaluate'):
                for candidate,control in config.get('paired_global_controls',[]):
                    a=json.loads((root/'runs'/candidate/'metrics.json').read_text())
                    b=json.loads((root/'runs'/control/'metrics.json').read_text())
                    if len(a['stages'])!=len(b['stages']) or any(
                            x['global_per_class_recall']!=y.get('global_per_class_recall',y['per_class_recall'])
                            for x,y in zip(a['stages'],b['stages'])):
                        raise ValueError('Candidate global predictions differ from transport control')
                if config.get('paired_global_controls'):state['paired_global_controls_equal']=True
                state['status']='COMPLETE';state['phase']='AWAIT_PUBLIC_DELIVERY';state['ended']=time.time();write();return
    state['status']='INCOMPLETE';state['phase']='STOPPED_DEPENDENCIES';state['ended']=time.time();write()


if __name__=='__main__':
    path=Path(os.environ['Q128_CONFIG']);config=json.loads(path.read_text())
    try:run(config)
    except BaseException as exc:
        root=Path(config['root']);p=root/'PROGRAM_STATE.json'
        s=json.loads(p.read_text()) if p.exists() else {}
        s.update(status='INCOMPLETE',coordinator_error=str(exc),ended=time.time());save(p,s);raise
