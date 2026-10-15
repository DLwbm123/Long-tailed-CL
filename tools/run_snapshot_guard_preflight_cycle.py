"""Exactly four preflights; terminal state has no transition to training or evaluation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main(c):
    assert c['gpus']==[0,1] and c['operation']=='preflight_only' and c['formal_training_authorized'] is False
    assert len(c['preflights'])==4 and {(j['name'],j['module']) for j in c['preflights']}=={(d,m) for d in ('ISIC','HK') for m in ('base','local_transport')}
    root=Path(c['root']);start=time.time();state=dict(status='RUNNING',phase='PREFLIGHT_ONLY',started_at=start,deadline=c['deadline'],active={},completed=[],costs=[],failures=[],formal_updates=0,training_allowed=False)
    def save():
        p=root/'PROGRAM_STATE.json';t=root/'PROGRAM_STATE.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
    assert subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip().startswith('nfs') and shutil.disk_usage(root).free>2*1024**3
    queue=list(c['preflights']);live={};free={};last_admission=0.;save()
    while queue or live:
        now=time.time();dirty=False
        if queue and len(live)<2 and now-last_admission>=30:
            last_admission=now
            try:free={int(v.split(',')[0]):int(v.split(',')[1]) for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
            except subprocess.CalledProcessError:free={}
        if now>=c['deadline']:
            for j in queue:state['failures'].append(dict(id=j['job_name'],reason='NOT_RUN_PREFLIGHT_DEADLINE'))
            queue.clear()
            dirty=True
        for gpu in (0,1):
            if not queue or gpu in live or free.get(gpu,-1)<24576:continue
            job=queue.pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),Q247_JOB=json.dumps(job))
            log=(root/'logs'/(job['job_name']+'.private.log')).open('x');begun=time.time()
            proc=subprocess.Popen(['/tmp/q107v/bin/python','-u','/tmp/q247w.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
            live[gpu]=(proc,job,begun,log);state['active'][str(gpu)]=dict(pid=proc.pid,id=job['job_name'],gpu=gpu,started_at=begun)
            dirty=True
        for gpu,(proc,job,begun,log) in list(live.items()):
            if now>=c['deadline'] and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            rc=proc.poll()
            if rc is None:continue
            end=time.time();log.close();state['costs'].append(dict(id=job['job_name'],operation='preflight',gpu=gpu,pid=proc.pid,started_at=begun,ended_at=end,GPU_process_residence_seconds=end-begun,returncode=rc))
            good=False
            if rc==0:
                try:
                    r=json.loads((Path(job['output'])/'PREFLIGHT_SCALARS.json').read_text());good=r['status']=='PASS' and r['formal_updates']==0 and r['parameters_exactly_unchanged'] and r['buffers_exactly_unchanged'] and r['guard_highest_bank']['status']=='PASS'
                except Exception:pass
            if good:state['completed'].append(job['job_name'])
            else:state['failures'].append(dict(id=job['job_name'],reason='PREFLIGHT_WORKER_OR_RECEIPT_FAILED',returncode=rc))
            del live[gpu];del state['active'][str(gpu)]
            free.pop(gpu,None);last_admission=0.
            dirty=True
        if dirty:save()
        if queue or live:time.sleep(.25)
    state.update(status='PREFLIGHT_COMPLETE_STOPPED' if len(state['completed'])==4 and not state['failures'] else 'PREFLIGHT_INCOMPLETE_STOPPED',phase='STOPPED_NO_TRAIN',ended_at=time.time(),whole_round_wall_seconds=time.time()-c['launched_at']);save()


if __name__=='__main__':main(json.loads(Path(os.environ['Q247_CONFIG']).read_text()))
