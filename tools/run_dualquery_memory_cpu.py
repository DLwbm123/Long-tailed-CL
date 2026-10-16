"""Frozen CPU case supervisor. Runtime paths are private environment variables, never argv."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    root=Path(os.environ['Q349_ROOT']);source=Path(os.environ['Q349_SOURCE'])
    lock=json.loads((source/'docs/dualquery_memory_design1_20261016/DESIGN_LOCK.json').read_text())
    if os.environ.get('Q349_MODE')=='worker':
        begun=time.monotonic();cfg=json.loads(os.environ['Q349_CASE'])
        sys.path.insert(0,str(source/'tools'))
        from check_dualquery_memory import run
        result=run(cfg,lock,lambda value:save(root/(cfg['id']+'.progress.json'),value))
        result['operation_seconds_including_import']=time.monotonic()-begun
        save(root/(cfg['id']+'.json'),result)
        return
    assert not (root/'CPU_PROGRAM.json').exists(), 'Refusing duplicate launch'
    start=time.monotonic();program=dict(status='RUNNING_CPU_ONLY',source_commit=os.environ['Q349_SHA'],
      started_at=time.time(),deadline_at=time.time()+lock['overall_wall_seconds'],active=None,cases=[],costs=[],failures=[])
    def update():save(root/'CPU_PROGRAM.json',program)
    update()
    for cfg in lock['cases']:
        if time.monotonic()-start>=lock['overall_wall_seconds']:
            program['cases'].append(dict(id=cfg['id'],status='NOT_RUN_OVERALL_CAP',result=None));update();continue
        env=os.environ.copy();env.update(Q349_MODE='worker',Q349_CASE=json.dumps(cfg))
        begin=time.monotonic()
        with (root/(cfg['id']+'.stdout.private.log')).open('w') as fo,(root/(cfg['id']+'.stderr.private.log')).open('w') as fe:
            p=subprocess.Popen([sys.executable,'-u','/tmp/q349d.py'],env=env,stdout=fo,stderr=fe)
            program['active']=dict(id=cfg['id'],pid=p.pid,started_at=time.time());update()
            killed_at=None;hard=False
            while True:
                pid,status,usage=os.wait4(p.pid,os.WNOHANG)
                if pid:break
                if (time.monotonic()-begin>lock['case_hard_wall_seconds'] or time.monotonic()-start>lock['overall_wall_seconds']) and killed_at is None:
                    p.terminate();hard=True;killed_at=time.monotonic()
                if killed_at is not None and time.monotonic()-killed_at>10:p.kill()
                time.sleep(.25)
            rc=os.waitstatus_to_exitcode(status);p.returncode=rc
        cost=dict(id=cfg['id'],returncode=rc,CPU_process_residence_seconds=time.monotonic()-begin,
                  CPU_user_seconds=usage.ru_utime,CPU_system_seconds=usage.ru_stime,
                  peak_RSS_MiB=usage.ru_maxrss/1024.,timeout=hard,GPU_seconds=0.)
        program['costs'].append(cost)
        exists=(root/(cfg['id']+'.json')).is_file()
        case_status='COMPLETED' if rc==0 and exists else ('CPU_HARD_CAP' if hard else 'WORKER_FAILED')
        program['cases'].append(dict(id=cfg['id'],status=case_status,result_file=cfg['id']+'.json' if exists else None))
        if case_status!='COMPLETED':program['failures'].append(dict(id=cfg['id'],reason=case_status))
        program['active']=None;update()
        if cfg['kind']=='unit' and case_status!='COMPLETED':
            for pending in lock['cases'][1:]:
                program['cases'].append(dict(id=pending['id'],status='NOT_RUN_UNIT_FAILED',result_file=None))
            update();break
    program.update(status='COMPLETE_CPU_CHECKS',finished_at=time.time(),supervisor_wall_seconds=time.monotonic()-start,active=None)
    update()


if __name__=='__main__':main()
