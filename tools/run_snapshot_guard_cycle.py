"""Four frozen guarded trajectories after delivered zero-update acceptance."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from run_module_cycle import save
from run_snapshot_guard_scope import validate, metadata, STEPS
from run_paired_transport_cycle import checkpoint_metadata

PREFLIGHT_SOURCE = '16ad48063895e58fadd93f6eba744d9d82599521'
RUNTIME_KEYS = {'output', 'source_commit', 'GPU_execution_authorized', 'formal_training_authorized',
                'operation', 'original_deadline', 'evaluation_gate'}


def validate_acceptance(c, previous, results, reuse):
    validate(c)
    if c.get('formal_training_authorized') is not True or c.get('development_evaluation_authorized') is not True:
        raise ValueError('Formal train and frozen development evaluation authorization required')
    if c.get('preflight_policy') != 'reuse_delivered_unchanged_production' or reuse.get('all_production_files_exact') is not True:
        raise ValueError('Only exact unchanged production may reuse delivered preflights')
    if reuse.get('preflight_source') != PREFLIGHT_SOURCE or reuse.get('preflight_public_commit') != 'df79726d74ae54f8cba103c2842394d09c6fc0b2':
        raise ValueError('Different preflight provenance')
    p = results['program']
    if p['status'] != 'PREFLIGHT_COMPLETE_STOPPED' or p['active'] or p['failures'] or len(p['costs']) != 4:
        raise ValueError('Four closed successful zero-update operations required')
    if any(x['returncode'] or x['operation'] != 'preflight' or x['gpu'] not in (0, 1) for x in p['costs']):
        raise ValueError('Preflight operation mismatch')
    if len(results['preflights']) != 4 or len(previous['jobs']) != 4:
        raise ValueError('Four paired receipts required')
    for job, old, row in zip(c['jobs'], previous['jobs'], results['preflights']):
        if {k:v for k,v in job.items() if k not in RUNTIME_KEYS} != {k:v for k,v in old.items() if k not in RUNTIME_KEYS}:
            raise ValueError('Scientific config differs from accepted preflight')
        if job.get('GPU_execution_authorized') is not True or job.get('formal_training_authorized') is not True:
            raise ValueError('Job authorization missing')
        r = row['result']; h = r['guard_highest_bank']; m = r['PMT_highest_bank']
        if (row['id'] != 'pre_'+job['job_name'] or r['source_commit'] != PREFLIGHT_SOURCE or r['status'] != 'PASS'
            or not r['real_first_batch'] or r['batch_n'] != 64 or r['encoder_dim'] != 1536
            or any(r[k] != 0 for k in ('adapter_updates','optimizer_updates','policy_updates','formal_updates'))
            or not all(r[k] is True for k in ('parameters_exactly_unchanged','buffers_exactly_unchanged','parameters_unchanged'))
            or r['test_accessed'] or r['validation_accessed'] or r['formal_training_started']
            or r['real_guard_certificate'] != 'CERTIFIED' or r['max_solve_residual'] > 1e-8
            or not math.isfinite(r['gradient_norm']) or r['gradient_norm'] <= 0
            or len(r['adapter_pool_gradients']) != 2
            or any(not math.isfinite(x) or x <= 0 for x in r['adapter_pool_gradients'].values())):
            raise ValueError('Real zero-update acceptance differs')
        ds = job['name']; old_n = {'ISIC':6,'HK':21}[ds]
        if (h['status'] != 'PASS' or not h['synthetic_only'] or h['dimension'] != 1536 or h['old_classes'] != old_n
            or h['fit_n'] != {'ISIC':9434,'HK':1038}[ds] or h['all_constraints_n'] != {'ISIC':66038,'HK':22836}[ds]
            or h['minimum_raw_slack'] < -2e-8 or h['stationarity'] > 1e-8 or h['complementarity'] > 1e-7
            or abs(h['primal_dual_gap']) > 1e-7 or h['optimizer_updates'] or h['formal_statistics_written']
            or not h['input_statistics_unchanged'] or not m['synthetic_only'] or not m['full_rank_memory_probe']
            or m['dimension'] != 1536 or m['old_classes'] != old_n or m['relative_residual'] > 1e-8):
            raise ValueError('Highest-bank acceptance differs')


def training_barrier(c):
    for job in c['jobs']:
        root = Path(job['output']); s = json.loads((root/'STATUS.json').read_text())
        if s['status'] != 'TRAINED' or s['steps'] != job['expected_steps'] or s['stages'] != len(job['task_sizes']) or s['policy_updates']:
            raise ValueError('Four exact trained trajectories required')
        for filename, fn in [('GUARD_TRAINING_SEAL.json', metadata), ('TRAINING_SEAL.json', checkpoint_metadata)]:
            expected = [fn(job,t,sum(STEPS[job['name']][:t])) for t in range(1,len(job['task_sizes'])+1)]
            if json.loads((root/filename).read_text()) != expected:
                raise ValueError('All-stage source/version/order/budget seals differ')


def run(c):
    root = Path(c['root']); lock = threading.Lock()
    state = dict(status='RUNNING',phase='TRAIN',started=time.time(),active={},completed=[],failures=[],costs=[],
                 preflights_reused=4,new_preflight_operations=0,source_commit=c['source_commit'],deadline=c['deadline'])
    def write():
        state['whole_round_wall_seconds'] = time.time()-c['launched_at']; save(root/'PROGRAM_STATE.json',state)
    def phase(operation):
        queue = list(c['jobs'])
        def worker(gpu):
            while True:
                with lock:
                    if state['failures'] or not queue:return
                    if time.time() >= c['deadline']:
                        state['failures'].append(dict(operation=operation,error='ORIGINAL_DEADLINE'));write();return
                    job = queue.pop(0)
                value = dict(job,operation=operation,evaluation_gate=str(root/'PROGRAM_STATE.json'))
                env = os.environ.copy();env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                begun = time.time(); mono = time.monotonic(); proc = None; error = None
                try:
                    with (root/'logs'/f'{job["job_name"]}_{operation}.private.log').open('x') as log:
                        proc = subprocess.Popen([sys.executable,'-u',c['worker_entry']],stdin=subprocess.PIPE,
                            stdout=log,stderr=subprocess.STDOUT,env=env,text=True,start_new_session=True)
                        with lock:
                            state['active'][str(gpu)] = dict(job=job['job_name'],operation=operation,pid=proc.pid);write()
                        try:proc.communicate(json.dumps(value),timeout=max(.01,c['deadline']-time.time()))
                        except subprocess.TimeoutExpired:
                            error='ORIGINAL_DEADLINE';os.killpg(proc.pid,signal.SIGTERM)
                            try:proc.communicate(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.communicate()
                    rc = proc.returncode
                except Exception as exc:
                    error = type(exc).__name__;rc = -1
                record = dict(job=job['job_name'],operation=operation,gpu=gpu,pid=None if proc is None else proc.pid,
                    started_at=begun,ended_at=time.time(),GPU_process_residence_seconds=None if proc is None else time.monotonic()-mono,
                    returncode=rc,error=error)
                with lock:
                    state['active'].pop(str(gpu),None);state['costs'].append(record)
                    if rc or error:state['failures'].append(record)
                    else:state['completed'].append(job['job_name']+'_'+operation)
                    write()
        threads=[threading.Thread(target=worker,args=(g,)) for g in c['gpus']]
        for t in threads:t.start()
        for t in threads:t.join()
        return not state['failures']
    write()
    try:
        if phase('train'):
            training_barrier(c)
            state.update(phase='EVALUATE',training_sealed_at=time.time());write()
            if phase('evaluate'):
                for j in c['jobs']:
                    s=json.loads((Path(j['output'])/'STATUS.json').read_text())
                    if s['status']!='COMPLETE' or s['steps']!=j['expected_steps'] or s['policy_updates']:
                        raise ValueError('Incomplete evaluation status')
                state.update(status='COMPLETE',phase='AWAIT_PUBLIC_DELIVERY')
        if state['status']!='COMPLETE':state.update(status='INCOMPLETE',phase='STOPPED_DEPENDENCIES')
    except Exception as exc:
        state['failures'].append(dict(operation='barrier',error=type(exc).__name__))
        state.update(status='INCOMPLETE',phase='STOPPED_DEPENDENCIES')
        (root/'coordinator_error.private.txt').write_text(str(exc))
    state['ended_at']=time.time();write()


def main(c):
    root=Path(c['root'])
    validate_acceptance(c,json.loads((root/'ACCEPTED_CONFIG.private.json').read_text()),
        json.loads((root/'ACCEPTED_PREFLIGHT.json').read_text()),json.loads((root/'SOURCE_REUSE.json').read_text()))
    state=dict(status='RUNNING',phase='WAITING_GPU01',active={},costs=[],failures=[],source_commit=c['source_commit'])
    save(root/'PROGRAM_STATE.json',state)
    while time.time()<c['deadline']:
        free={int(r.split(',')[0]):int(r.split(',')[1]) for r in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        if all(free.get(g,-1)>=24576 for g in c['gpus']):
            save(root/'GPU_LAUNCH_CHECK.json',dict(gpus=c['gpus'],free_MiB={g:free[g] for g in c['gpus']},at=time.time()))
            run(c);return
        time.sleep(min(30,max(0,c['deadline']-time.time())))
    state.update(status='INCOMPLETE',phase='STOPPED_DEPENDENCIES',ended_at=time.time(),failures=[dict(error='ORIGINAL_DEADLINE')])
    save(root/'PROGRAM_STATE.json',state)


if __name__=='__main__':
    config=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    try:main(config)
    except BaseException as exc:
        root=Path(config['root']);p=root/'PROGRAM_STATE.json'
        s=json.loads(p.read_text()) if p.exists() else dict(active={},costs=[],failures=[])
        s.update(status='INCOMPLETE',phase='STOPPED_DEPENDENCIES',ended_at=time.time())
        s['failures'].append(dict(operation='coordinator',error=type(exc).__name__));save(p,s)
        raise
