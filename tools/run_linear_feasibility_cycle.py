"""Single bounded queue; immutable source, neutral worker entry, no retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path,value):
    temp=path.with_suffix('.part');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    worker=os.environ.get('Q356_WORKER','/tmp/q356w.py')
    cfg=json.loads((root/'CONFIG.private.json').read_text())
    lock=json.loads((source/os.environ.get('Q356_PROTOCOL','docs/real_query_node1_20261016/PROTOCOL_LOCK.json')).read_text())
    assert not (root/'PROGRAM_COMPACT.json').exists(),'Duplicate launch forbidden'
    start=time.monotonic();deadline=min(start+lock['overall_wall_seconds'],lock['original_overall_deadline_epoch'])
    state=dict(status='RUNNING',source_commit=os.environ['Q356_SHA'],started_at=time.time(),active=None,cases=[],costs=[],failures=[])
    update=lambda:save(root/'PROGRAM_COMPACT.json',state)
    update()
    for job in lock['jobs']:
        begin=time.monotonic();gpu=job['role'].startswith('extract_')
        if begin>=deadline:state['failures'].append(dict(id=job['id'],reason='OVERALL_WALL_CAP'));break
        if job['role']=='extract_gate':
            candidates=[json.loads((root/(j['id']+'.json')).read_text()) for j in lock['jobs'] if j['role']=='candidate']
            assert len(candidates)==2 and all(v['status']=='COMPLETED' for v in candidates)
            save(root/'CANDIDATES_LOCK.json',dict(locked_at=time.time(),candidates=[dict(dataset=v['dataset'],sealed_at=v['sealed_at']) for v in candidates]))
        if gpu:
            free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            available={int(v.split(',')[0]):int(v.split(',')[1]) for v in free.splitlines()}
            if available[cfg['gpu']]<lock['minimum_free_gpu_MiB']:
                state['failures'].append(dict(id=job['id'],reason='GPU_ADMISSION_FAILED',available_MiB=available[cfg['gpu']]));break
        env=os.environ.copy();env.update(Q356_JOB=json.dumps(job),CUDA_VISIBLE_DEVICES=str(cfg['gpu']) if gpu else '')
        with (root/(job['id']+'.stdout.private.log')).open('x') as fo,(root/(job['id']+'.stderr.private.log')).open('x') as fe:
            p=subprocess.Popen([sys.executable,'-u',worker],env=env,stdout=fo,stderr=fe)
            state['active']=dict(id=job['id'],pid=p.pid,role=job['role'],started_at=time.time());update()
            commands=subprocess.check_output(['ps','-p',str(p.pid),'-o','args='],text=True).strip()
            assert worker in commands and not any(v in commands.lower() for v in ('wangbomin','longtailed','query_node'))
            stopped=None;stop_reason=None
            while True:
                pid,status,usage=os.wait4(p.pid,os.WNOHANG)
                if pid:break
                now=time.monotonic()
                rss=0
                try:
                    rss=int(next(line.split()[1] for line in Path('/proc',str(p.pid),'status').read_text().splitlines() if line.startswith('VmRSS:')))/1024
                except (FileNotFoundError,StopIteration):pass
                if stopped is None and (rss>lock['worker_RSS_limit_MiB'] or now>=deadline or now-begin>lock['case_wall_seconds']):
                    stop_reason='RSS_LIMIT' if rss>lock['worker_RSS_limit_MiB'] else 'TIMEOUT'
                    p.terminate();stopped=now
                if stopped is not None and now-stopped>10:p.kill()
                time.sleep(.5)
            rc=os.waitstatus_to_exitcode(status);p.returncode=rc
        elapsed=time.monotonic()-begin
        state['costs'].append(dict(id=job['id'],returncode=rc,process_residence_seconds=elapsed,
            CPU_only_residence_seconds=0. if gpu else elapsed,GPU_process_residence_seconds=elapsed if gpu else 0.,
            CPU_user_seconds=usage.ru_utime,CPU_system_seconds=usage.ru_stime,peak_RSS_MiB=usage.ru_maxrss/1024.,timeout=stop_reason=='TIMEOUT',stop_reason=stop_reason))
        ok=rc==0 and (root/(job['id']+'.json')).is_file()
        state['cases'].append(dict(id=job['id'],status='COMPLETED' if ok else 'FAILED'))
        state['active']=None;update()
        if not ok:state['failures'].append(dict(id=job['id'],reason=stop_reason or 'WORKER_FAILED'));break
    completed={v['id'] for v in state['cases']}
    state['not_run']=[j['id'] for j in lock['jobs'] if j['id'] not in completed]
    state.update(status='FAILED_STOPPED' if state['failures'] else 'COMPLETE_AWAITING_DELIVERY',finished_at=time.time(),
                 supervisor_wall_seconds=time.monotonic()-start,active=None)
    update()


if __name__=='__main__':
    try:main()
    except BaseException as e:
        root=Path(os.environ['Q356_ROOT']);p=root/'PROGRAM_COMPACT.json'
        state=json.loads(p.read_text()) if p.exists() else dict(cases=[],costs=[],failures=[])
        state.update(status='SUPERVISOR_FAILED',error=repr(e),finished_at=time.time())
        save(p,state);raise
