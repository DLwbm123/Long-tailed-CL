"""One bounded three-GPU matrix; audited engineering recovery preserves all costs."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_prototype_single import save


def recovery_plan(c, previous):
    records = previous['jobs'] if previous else []
    if previous:
        if previous['status'] != 'INCOMPLETE' or not c.get('recovery_note'):
            raise ValueError('Recovery requires a diagnosed failure and an explicit repair note')
        if any('elapsed_seconds' not in r for r in records):
            raise ValueError('Reconcile interrupted process residence before recovery')
        if previous['gpu_seconds_limit'] != c['gpu_seconds_limit']:
            raise ValueError('Recovery cannot reset or change the GPU budget')
    done = {r.get('logical_id',r['id']):r['id'] for r in records
            if r.get('status')=='COMPLETE' and r.get('exit_code')==0}
    pending = []
    for job in c['jobs']:
        name = job['id']
        if name in done: continue
        count = sum(r.get('logical_id',r['id'])==name for r in records)
        pending.append(dict(job,attempt_id=name if not count else f'{name}_repair{count}'))
    spent = sum(r.get('elapsed_seconds',0.) for r in records)
    if spent+sum(j['cap_seconds']+60 for j in pending) > c['gpu_seconds_limit']:
        raise ValueError('Remaining budget cannot reserve all unfinished trajectories')
    return records,pending,done


def process_check(root, records):
    rows = [line.split(None,2) for line in subprocess.check_output(
        ['ps','-eo','pid=,ppid=,args='],text=True).splitlines()]
    owned = {os.getpid()} | {r['pid'] for r in records if 'exit_code' not in r}
    for _ in range(8): owned.update(int(pid) for pid,ppid,args in rows if int(ppid) in owned)
    commands = [args for pid,ppid,args in rows if int(pid) in owned]
    gpu_rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name',
        '--format=csv,noheader'],text=True).splitlines()
    gpu_names = [line for line in gpu_rows if int(line.split(',')[0]) in owned]
    forbidden = ('wangbomin','LongTailed','Long-tailed','prototype','cf_group','cf_linear','cf_weighted','cf_prototype')
    if any(word in line for line in commands+gpu_names for word in forbidden):
        raise ValueError('Visible process command policy violation')
    save(root/'public/PROCESS_CHECK.json',dict(passed=True,commands=commands,gpu_processes=gpu_names,checked=time.time()))


def run(c):
    root = Path(c['root'])
    lock = (root/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'HOLD.json').exists(): raise ValueError('Launch hold must be explicitly released')
    state_path = root/'PROGRAM_STATE.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    records,pending,done = recovery_plan(c,previous)
    if previous:
        save(root/f'RECOVERY_{len(records)}.json',dict(previous=previous,note=c['recovery_note'],time=time.time()))
    active = {}; started = previous['started'] if previous else time.time()
    deadline = previous['deadline'] if previous else started+c['wall_seconds']; outcome = 'RUNNING'
    initial_records = len(records)
    def record(**extra):
        now = time.time()
        total = sum(r.get('elapsed_seconds',now-r['started']) for r in records)
        state = dict(status=outcome,started=started,deadline=deadline,gpu_seconds=total,
            gpu_seconds_limit=c['gpu_seconds_limit'],jobs=records,pending=[j['id'] for j in pending],
            publication_verified=False,source_commit=c['source_commit'],**extra)
        save(root/'PROGRAM_STATE.json',state);save(root/'public/BUDGET_LEDGER.json',state)
        return total
    def reap():
        for name,(p,r) in list(active.items()):
            code = p.poll()
            if code is None: continue
            r.update(exit_code=code,elapsed_seconds=time.time()-r['started'])
            active.pop(name)
            receipt = root/r['id']/'STATUS.json'
            r['status'] = json.loads(receipt.read_text())['status'] if receipt.exists() else 'MISSING_RECEIPT'
    try:
        if len({j['id'] for j in pending}) != len(pending): raise ValueError('Duplicate jobs')
        record()
        checked = set()
        while pending or active:
            reap(); total = record()
            if any(r.get('exit_code',0) != 0 or r.get('status','COMPLETE') != 'COMPLETE' for r in records[initial_records:]):
                raise RuntimeError('Job failed; diagnose before audited recovery')
            if time.time() >= deadline or total >= c['gpu_seconds_limit']:
                raise TimeoutError('Campaign budget exhausted')
            if any(time.time()-r['started'] >= r['cap_seconds'] for p,r in active.values()):
                raise TimeoutError('Job residence cap exhausted')
            free = {int(i):int(m) for i,m in (line.split(',') for line in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
            occupied = {r['gpu'] for p,r in active.values()}
            preflight_ok = any(r.get('logical_id',r['id'])=='PREFLIGHT' and r.get('status')=='COMPLETE' for r in records)
            for gpu in c['gpus']:
                if not pending or gpu in occupied: continue
                spec = pending[0]
                if spec['id'] != 'PREFLIGHT' and not preflight_ok: continue
                if free[gpu] < spec['minimum_free_mib']: continue
                if time.time()+spec['cap_seconds'] > deadline: raise TimeoutError('Cannot admit job before deadline')
                pending.pop(0)
                name = spec['attempt_id']
                config = dict(spec['config'],output=str(root/name),max_wall_seconds=spec['cap_seconds']-30)
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                with (root/(name+'.private.log')).open('w') as stream:
                    p = subprocess.Popen([c['python'],'-u',c['worker_entry']],stdin=subprocess.PIPE,
                        stdout=stream,stderr=subprocess.STDOUT,env=env,text=True,start_new_session=True)
                r = dict(id=name,logical_id=spec['id'],gpu=gpu,pid=p.pid,started=time.time(),
                         cap_seconds=spec['cap_seconds'],source_commit=c['source_commit'])
                records.append(r); active[name] = (p,r)
                p.stdin.write(json.dumps(config));p.stdin.close();record()
            if active and all(time.time()-r['started']>12 for p,r in active.values()):
                ids = {r['pid'] for p,r in active.values()}
                if not ids.issubset(checked):
                    process_check(root,records);checked |= ids
            if active or pending: time.sleep(5)
        outcome='AWAITING_PUBLICATION';record()
        from summarize_lt_benchmark import summarize
        locations = {r.get('logical_id',r['id']):r['id'] for r in records if r.get('status')=='COMPLETE'}
        save(root/'public/RESULT_LOCATIONS.json',locations)
        summarize(root,[dict(j,result_id=locations[j['id']]) for j in c['jobs']])
    except BaseException as exc:
        for p,r in active.values():
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
        for p,r in active.values():
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()
        reap();outcome='INCOMPLETE';record(error=str(exc));raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
