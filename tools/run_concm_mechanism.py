"""One statistics seal; fixed MPC then DSM then combined evaluations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from run_module_cycle import save


def run(c):
    root=Path(c['root']);begun=time.monotonic();lock=threading.Lock()
    expected=[ds+'_'+a for ds in ('ISIC','HK') for a in ('base','local_transport')]
    if c['gpus']!=[0,1] or [j['job_name'] for j in c['jobs']]!=expected:raise ValueError('Four fixed trajectories only')
    s=dict(status='RUNNING',phase='PREPARE',active={},costs=[],cpu_costs=[],failures=[],started=time.time())
    def write():
        s['elapsed_seconds']=time.monotonic()-begun;save(root/'PROGRAM_STATE.json',s)
    def execute(index,slot,operation):
        name=expected[index];variant=s.get('variant','');start=time.monotonic();process_start=None;code=-1;error=None
        try:
            if operation!='prepare':
                lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                memory={int(v.split(',')[0]):int(v.split(',')[1]) for v in lines.splitlines()}
                if memory[slot]<32768:raise RuntimeError('GPU memory admission failed; no fallback or retry')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if operation=='prepare' else str(slot),
                Q171_JOB=str(index),Q171_OPERATION=operation,Q171_VARIANT=s.get('variant',''),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (root/'logs'/f'{variant}_{name}_{operation}.private.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-u',c['worker_entry']],stdout=log,stderr=subprocess.STDOUT,env=env)
                process_start=time.monotonic()
                with lock:s['active'][str(slot)]=dict(job=name,variant=variant,operation=operation,pid=p.pid);write()
                try:code=p.wait(timeout=max(1,c['deadline']-time.time()))
                except subprocess.TimeoutExpired:
                    p.terminate();p.wait();code=124
        except BaseException as exc:error=type(exc).__name__
        finally:
            cost=dict(job=name,variant=variant,operation=operation,gpu=None if operation=='prepare' else slot,
                returncode=code,residence_seconds=0. if process_start is None else time.monotonic()-process_start,
                process_started=process_start is not None,dispatch_wall_seconds=time.monotonic()-start)
            if error:cost['dispatch_error_type']=error
            with lock:
                s['active'].pop(str(slot),None)
                s['cpu_costs' if operation=='prepare' else 'costs'].append(cost)
                if code:s['failures'].append(cost)
                write()
    write()
    operations=[('', 'prepare')]+[(v,op) for v in ('MECHANISM',) for op in ('preflight','evaluate')]
    s['preflights_sealed_at']={}
    for variant,operation in operations:
        if s['failures'] or time.time()>=c['deadline']:break
        s['phase']=operation.upper();s['variant']=variant;write()
        for batch in (0,2):
            if s['failures'] or time.time()>=c['deadline']:break
            ts=[threading.Thread(target=execute,args=(batch+slot,slot,operation)) for slot in (0,1)]
            for t in ts:t.start()
            for t in ts:t.join()
        if s['failures']:break
        if operation=='prepare':s['sidecars_sealed_at']=time.time()
        if operation=='preflight':s['preflights_sealed_at'][variant]=time.time()
        write()
    complete=len(s['cpu_costs'])==4 and len(s['costs'])==8 and not s['failures']
    s.update(status='COMPLETE' if complete else 'INCOMPLETE',phase='AWAIT_PUBLIC_DELIVERY',ended=time.time(),
        whole_round_wall_seconds=time.time()-c['launched_at']);write()


if __name__=='__main__':run(json.loads(Path(os.environ['Q128_CONFIG']).read_text()))
