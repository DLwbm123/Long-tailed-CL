"""Four independent trajectory workers with one cumulative GPU-hour ledger."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil

ROOT=Path(os.environ['N78_ROOT']);PUB=ROOT/'public';PRIVATE=ROOT/'private'
CFG=json.loads((PRIVATE/'INPUT.json').read_text());MODE=os.environ['N78_MODE']
PYTHON=CFG.get('python','/tmp/m62v/bin/python');ENTRY=CFG.get('entry','/tmp/p78.py')
METHODS=tuple(CFG.get('methods',('S','H','K','G','R','E')))
FROZEN=CFG.get('frozen_references',[('F_S','S'),('F_R','R')])
RUN_IDS=tuple(map(int,CFG.get('run_specs',{s:{} for s in (1993,1994,1995)})))
N_STAGES=4*len(RUN_IDS)*(len(METHODS)+len(FROZEN));N_CLASSES=20*len(RUN_IDS)*(len(METHODS)+len(FROZEN))
GPUS=tuple(CFG.get('gpu_indices',(0,1,2,3)))
GPU_LIMIT=min(57600.,float(CFG.get('gpu_budget_seconds',57600.)))
if CFG.get('experiment')=='NB-RL-A4':
    assert CFG.get('campaign_budget_authorized') is True and 'gpu_budget_seconds' in CFG and GPU_LIMIT>0,'BLOCKED_MISSING_CAMPAIGN_BUDGET'
BASE=json.loads((PUB/'RESOURCE_LEDGER.json').read_text())
PROCESS=[];RUNNING={};START=time.time();T0=CFG['wall_T0_unix']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def residence(now=None):
    now=time.time() if now is None else now
    return BASE['GPU_process_residence_seconds']+sum(p.get('ended',now)-p['started'] for p in PROCESS)

def limits(role=None):
    now=time.time();elapsed=now-T0
    assert elapsed<57600-30 and residence(now)<GPU_LIMIT-60,'INCOMPLETE_BUDGET_16H'
    if MODE in ('probe','check'):assert elapsed<9000,'BLOCKED_P0_BUDGET'
    if role=='train':assert elapsed<45000,'INCOMPLETE_BUDGET_TRAIN_12_5H'
    if role in ('frozen','evaluate'):assert elapsed<52200,'INCOMPLETE_BUDGET_FORWARD_14_5H'

def stop_children():
    for pid,(proc,log,rec) in list(RUNNING.items()):
        if proc.poll() is None:os.killpg(pid,signal.SIGTERM)
    end=time.time()+10
    while any(p.poll() is None for p,log,rec in RUNNING.values()) and time.time()<end:time.sleep(.1)
    for pid,(proc,log,rec) in list(RUNNING.items()):
        if proc.poll() is None:os.killpg(pid,signal.SIGKILL)
        proc.wait();log.close();rec['ended']=time.time();rec['returncode']=proc.returncode
    RUNNING.clear()

def ledger():
    counts=dict(BASE['counts']);peak=BASE['peak_GPU_allocated_bytes'];image_opens={}
    for rec in PROCESS:
        path=PRIVATE/'workers'/rec['worker']/'RESOURCE_LEDGER.json'
        if path.exists():
            v=read(path);peak=max(peak,v['peak_GPU_allocated_bytes']);image_opens[rec['worker']]=v['image_opens_this_process']
            for key,value in v['counts'].items():counts[key]=counts.get(key,0)+value
    primary=0
    for path in ROOT.rglob('*'):
        try:
            if path.is_file() and not path.is_symlink():primary+=path.stat().st_size
        except FileNotFoundError:pass
    backup={}
    for path in (PRIVATE/'transfer_acks').glob('*.json'):
        for item in read(path).get('items',[]):backup[item['file']]=item['bytes']
    value=dict(BASE,phase='parallel_'+MODE,counts=counts,wall_seconds=time.time()-T0,GPU_process_residence_seconds=residence(),
               primary_bytes=primary,independent_backup_bytes=sum(backup.values()),persistent_including_backup_bytes=primary+sum(backup.values()),
               peak_GPU_allocated_bytes=peak,unix=time.time(),gpu_limit=len(GPUS),budget_GPU_seconds=GPU_LIMIT,image_opens_by_worker=image_opens,
               GPU_accounting='P0 prior plus sum of every child launch-to-exit duration, including import, IO and teardown',
               active_workers=[r['worker'] for p,l,r in RUNNING.values()])
    write(PUB/'RESOURCE_LEDGER.json',value)
    write(PRIVATE/('PROCESS_LEDGER_'+MODE+'.json'),dict(T0=T0,processes=PROCESS,GPU_seconds=value['GPU_process_residence_seconds']))
    assert value['persistent_including_backup_bytes']<16*1024**3,'BLOCKED_STORAGE'
    return value

def run_jobs(role,jobs,prefix):
    pending=list(enumerate(jobs));finished=[];last=0
    while pending or RUNNING:
        limits(role)
        used={r['GPU'] for p,l,r in RUNNING.values()}
        for gpu in GPUS:
            if gpu in used or not pending:continue
            index,job=pending.pop(0);worker=prefix+f'{index:02d}';out=PRIVATE/'workers'/worker
            assert not out.exists(),'BLOCKED_WORKER_REUSE'
            env=dict(os.environ,N78_ROLE=role,N78_WORKER=worker,N78_JOB=json.dumps(job),CUDA_VISIBLE_DEVICES=str(gpu),
                     OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
            log=(PRIVATE/(worker+'.log')).open('w');started=time.time()
            deadline=min(T0+57600,T0+(45000 if role=='train' else 52200 if role in ('frozen','evaluate') else 57600))
            seconds=max(1,int(deadline-time.time()-10))
            proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=5',str(seconds),PYTHON,ENTRY],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rec=dict(worker=worker,role=role,job=job,GPU=gpu,pid=proc.pid,started=started)
            PROCESS.append(rec);RUNNING[proc.pid]=(proc,log,rec)
            print('START',worker,role,gpu,proc.pid,flush=True)
        for pid,(proc,log,rec) in list(RUNNING.items()):
            code=proc.poll()
            if code is None:continue
            rec['ended']=time.time();rec['returncode']=code;log.close();del RUNNING[pid]
            assert code==0,f"BLOCKED_WORKER_{rec['worker']}_EXIT_{code}"
            filename={'parallel_probe':'PARALLEL_PROBE.json','next_probe':'PARALLEL_PROBE.json','verify':'CURSOR_AND_TRANSITION_CHECK.json'}.get(role,'WORKER_COMPLETE.json')
            result=read(PRIVATE/'workers'/rec['worker']/filename);finished.append((rec,result))
            print('END',rec['worker'],flush=True)
        if time.time()-last>15:ledger();last=time.time()
        if pending or RUNNING:time.sleep(2)
    ledger();return finished

def enqueue(paths,label):
    key=str(time.time_ns());items=[]
    for path in paths:
        path=Path(path)
        if path.parent==PUB:
            dest=PRIVATE/'metadata'/key/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);path=dest
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        items.append(dict(file=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=h.hexdigest()))
    write(PRIVATE/'transfer_requests'/(key+'.json'),dict(id=key,label=label,items=items,unix=time.time()))
    return key

def backlog(wait=False):
    while True:
        limits();pending=[]
        for path in (PRIVATE/'transfer_requests').glob('*.json'):
            request=read(path);ack=PRIVATE/'transfer_acks'/path.name
            if ack.exists():
                value=read(ack);assert value['status']=='PASS' and value['items']==request['items']
            else:
                assert time.time()-request['unix']<900,'BLOCKED_BACKUP_BACKLOG'
                pending.append(path)
        if not pending or not wait:return
        time.sleep(2)

def merge_records(records):
    for name in ('STAGE_RECEIPTS.jsonl','INITIALIZATION_BY_TRAJECTORY.jsonl','gradient_diagnostics.jsonl','reward_diagnostics.jsonl'):
        paths=[PRIVATE/'workers'/r['worker']/name for r,v in records]
        with (PUB/name).open('w') as f:
            for path in paths:
                if path.exists():f.write(path.read_text())

def main():
    if MODE=='check':
        results=run_jobs('verify',[{}],'q')
        write(PUB/'PARALLEL_NATIVE_RECOVERY_CHECK.json',dict(status='PASS',native_checkpoint=True,result=results[0][1],GPU_seconds=residence()))
        return
    if MODE=='probe':
        next_round=CFG.get('experiment') in ('NB-RL-A2','NB-RL-A3','NB-RL-A4')
        results=run_jobs('next_probe' if next_round else 'parallel_probe',
                         CFG.get('probe_jobs',[dict(method=m) for m in METHODS]) if next_round else [{}]*4,'p')
        assert sum(v['steps'] for r,v in results)==(8*len(CFG.get('probe_jobs',METHODS)) if next_round else 40)
        write(PUB/'PARALLEL_PROBE_SUMMARY.json',dict(status='PASS',workers=[v for r,v in results],
              engineering_steps=ledger()['counts']['engineering_steps'],GPU_seconds=residence(),validation_images=0))
        return
    assert MODE=='formal'
    assert not (PUB/'COMPLETE.json').exists()
    write(PUB/'RUN_STATUS.json',dict(status='RUNNING',phase='train',unix=time.time(),supervisor_pid=os.getpid()))
    lock=read(PUB/'PROTOCOL_LOCK.json');assert lock['epochs'] in (3,5) and lock['budgets']['gpu_count']==len(GPUS)
    assert read(PUB/'P0_ENGINEERING_REPORT.json')['status']=='PASS'
    recovery_files=[ROOT/p for p in CFG.get('recovery_files',
        ['public/PARALLEL_NATIVE_RECOVERY_CHECK.json','private/workers/q00/cursor_resume.pt'])]
    enqueue(list((ROOT/'source').glob('*.py'))+[PUB/'PROTOCOL_LOCK.json',PUB/'P0_ENGINEERING_REPORT.json',PUB/'PARALLEL_AUTHORIZATION.json']+recovery_files,'parallel_source_and_lock')
    jobs=[dict(seed=s,method=m) for s in RUN_IDS for m in METHODS]
    trained=run_jobs('train',jobs,'t');entries=[e for r,v in trained for e in v['entries']]
    assert len(entries)==4*len(RUN_IDS)*len(METHODS) and sum(v['counts']['formal_steps'] for r,v in trained)==read(PUB/'METHOD_MATRIX.json')['tiers'][str(lock['epochs'])]
    merge_records(trained)
    initials=[json.loads(line) for line in (PUB/'INITIALIZATION_BY_TRAJECTORY.jsonl').read_text().splitlines()]
    for seed in RUN_IDS:
        assert len({v['delta_sha256'] for v in initials if v['seed']==seed})==1
        if CFG.get('experiment') in ('NB-RL-A2','NB-RL-A3','NB-RL-A4'):
            assert len({v['network_delta_sha256'] for v in entries if v['seed']==seed and v['task']==1})==1,'BLOCKED_TASK1_MISMATCH'
        for task in range(1,5):
            for epoch in range(1,lock['epochs']+1):
                prefix=f'{seed}_t{task}_e{epoch}.jsonl';reference=(PRIVATE/'batch_checks'/(METHODS[0]+'_'+prefix)).read_text()
                for method in METHODS[1:]:
                    assert (PRIVATE/'batch_checks'/(method+'_'+prefix)).read_text()==reference,'BLOCKED_BATCH_AUGMENTATION_MISMATCH'
    write(PUB/'TRAINED_MATRIX_LOCK.json',dict(status='LOCKED',entries=entries,epochs=lock['epochs'],new_optimizer_steps=sum(v['counts']['formal_steps'] for r,v in trained),
          batch_augmentation_equality=True,initialization_equality=True,unix=time.time()))
    frozen=run_jobs('frozen',[dict(seed=s,method=m,owner=o) for s in RUN_IDS for m,o in FROZEN],'f')
    entries += [e for r,v in frozen for e in v['entries']]
    merge_records(trained+frozen)
    assert len(entries)==N_STAGES and len({(e['method'],e['seed'],e['task']) for e in entries})==N_STAGES
    write(PUB/'ALL_STATES_LOCK.json',dict(status='LOCKED',entries=entries,unix=time.time(),validation_images_read=0))
    evaluated=run_jobs('evaluate',[dict(indices=list(range(i,N_STAGES,len(GPUS)))) for i in range(len(GPUS))],'e')
    predictions=[e for r,v in evaluated for e in v['entries']]
    assert len(predictions)==N_STAGES
    write(PUB/'PREDICTIONS_LOCK.json',dict(status='LOCKED',entries=predictions,unix=time.time()))
    run_jobs('analyze',[{}],'a');backlog(wait=True)
    final=ledger();assert final['counts']['formal_steps']==read(PUB/'METHOD_MATRIX.json')['tiers'][str(lock['epochs'])]
    write(PUB/'ACCESS_LEDGER.json',dict(final,scope='Per-process current task fit, then all-seen validation only after global ALL_STATES_LOCK'))
    write(PUB/'BACKUP_REPORT.json',dict(status='PASS',all_requests_acknowledged=True,independent_SHA_reread=True,
         P0_full_recovery_check='INDEPENDENT_RESTORE_CHECK.json',P0_next_update_check='CURSOR_AND_TRANSITION_CHECK.json'))
    write(PUB/'COMPLETE.json',dict(execution_status='COMPLETE',matrix_status='COMPLETE',stage_rows=N_STAGES,class_rows=N_CLASSES,
          epochs=lock['epochs'],new_steps=final['counts']['formal_steps'],NEXT_DECISION='STOP',
          publication_status='AUTHORIZED_PENDING_DELIVERY' if CFG.get('publication_authorized') else 'AWAITING_AUTHORIZATION'))
    write(PUB/'RUN_STATUS.json',dict(status='COMPLETE',unix=time.time()))
    enqueue(list(PUB.iterdir()),'completion_receipt');backlog(wait=True)
    write(PUB/'FINAL_BACKUP_ACK.json',dict(status='PASS',unix=time.time(),all_requests_acknowledged=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        stop_children();ledger()
        status='INCOMPLETE_BUDGET' if 'BUDGET' in str(error) else 'BLOCKED'
        write(PUB/'RUN_STATUS.json',dict(status=status,reason=str(error),unix=time.time()))
        if MODE=='formal':
            write(PUB/'NEXT_DECISION.json',dict(NEXT_DECISION='STOP',matrix_status=status,utility_status='NOT_EVALUABLE',reason=str(error)))
            (PUB/'FINAL_REPORT_ZH.md').write_text('# '+CFG.get('experiment','NB-RL-A1')+' 未完成\n\n'+status+'：'+str(error)+'\n\n完整矩阵未完成；方法效用 NOT_EVALUABLE。保留已有状态与账本，不填补缺失结果。\n')
            try:enqueue(list(PUB.iterdir()),'completion_receipt');backlog(wait=True)
            except Exception:pass
        raise
