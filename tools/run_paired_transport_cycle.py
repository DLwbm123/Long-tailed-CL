"""Frozen paired moment transport, GPU01 and an six-trajectory training seal."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_module_cycle import run,save
from run_lca_supported_cycle import ORDERS,SIZES,STEPS,read_training
VERSION='paired_moment_transport_v1'

MODES=(('base','global'),('local_transport','global_regional'),('local_transport','global'))
MIN_FREE_MIB=24576


def admitted(available,gpus):return all(available.get(g,-1)>=MIN_FREE_MIB for g in gpus)


def label(module,mode):return 'pmt' if module=='base' else 'local_transport_pmt'+('_global_only' if mode=='global' else '')


def validate_job(j):
    ds=j.get('name');pre=j.get('preflight',False)
    if (ds not in ORDERS or j.get('paired_moment_transport') is not True or j.get('pmt_authorized') is not True
        or j.get('pmt_version')!=VERSION or (j.get('module'),j.get('pmt_mode')) not in MODES
        or j.get('method')!='static_pc' or j.get('pmt_relative_ridge')!=.001
        or j.get('evaluation_split')!='development_validation' or j.get('checkpoint_root') or j.get('lca_supported')
        or j.get('dynamic_anchor') or j.get('residual_drift_weight',0)!=0 or j.get('test_accessed',False)
        or j.get('order')!=ORDERS[ds] or j.get('task_sizes')!=SIZES[ds] or j.get('seed')!=74002 or j.get('split_seed')!=74002
        or j.get('epochs')!=2 or j.get('batch_size')!=64 or j.get('lr')!=.0003
        or j.get('workers')!=(0 if pre else 4) or j.get('expected_steps')!=(0 if pre else sum(STEPS[ds]))
        or j.get('max_steps')!=sum(STEPS[ds]) or not j.get('source_commit') or not j.get('control_checkpoint_root')):
        raise ValueError('Frozen PMT scope/budget/version differs')


def validate(c):
    if c.get('gpus')!=[0,1] or c.get('version')!=VERSION or len(c['jobs'])!=6 or len(c['preflights'])!=6:
        raise ValueError('Six new trajectories/preflights on GPU01 required')
    wanted={(ds,m,a) for ds in ORDERS for m,a in MODES}
    if {(j['name'],j['module'],j['pmt_mode']) for j in c['jobs']}!=wanted:raise ValueError('No additional method/budget arms')
    for j,p in zip(c['jobs'],c['preflights']):
        validate_job(j);validate_job(p)
        if p.get('preflight') is not True or p.get('job_name')!='pre_'+j['job_name']:raise ValueError('Genuine zero-update preflight required')
        for key in ('name','module','order','task_sizes','seed','split_seed','epochs','lr','batch_size','manifest','images','weight',
                    'legacy_repo','source_commit','pmt_version','pmt_mode','pmt_relative_ridge','control_checkpoint_root'):
            if j[key]!=p[key]:raise ValueError('Preflight/formal mismatch')
    if len({j['output'] for j in c['jobs']+c['preflights']})!=12:raise ValueError('Fresh distinct outputs required')


def checkpoint_metadata(c,task,steps):
    return {key:c[key] for key in ('name','module','order','seed','split_seed','source_commit','pmt_version','pmt_mode',
                                   'pmt_relative_ridge','task_sizes','epochs','batch_size','lr','max_steps')} | dict(task=task,steps=steps)


def check_barrier(c):
    campaign=json.loads(Path(os.environ['Q128_CONFIG']).read_text());validate(campaign)
    gate=json.loads(Path(c['evaluation_gate']).read_text())
    if gate['phase']!='EVALUATE' or not gate.get('training_sealed_at'):raise ValueError('Validation remains sealed')
    for j in campaign['jobs']:
        s=json.loads((Path(j['output'])/'STATUS.json').read_text())
        seals=json.loads((Path(j['output'])/'TRAINING_SEAL.json').read_text())
        if seals!=[checkpoint_metadata(j,t,sum(STEPS[j['name']][:t])) for t in range(1,len(j['task_sizes'])+1)]:raise ValueError('All checkpoint seals required')
        if s['status'] not in ('TRAINED','COMPLETE') or s['steps']!=j['expected_steps'] or s['stages']!=len(j['task_sizes']) or s['policy_updates']:
            raise ValueError('Six exact trained trajectories required before val')


def main(c):
    validate(c);root=Path(c['root']);begun=time.time()
    save(root/'PROGRAM_STATE.json',dict(status='RUNNING',phase='WAITING_GPU01',active={},costs=[],failures=[]))
    if not subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip().startswith('nfs') or shutil.disk_usage(root).free<40*1024**3:
        raise ValueError('NAS admission failed')
    probe=root/'.admission_probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    while True:
        if time.time()>=c['deadline']:raise TimeoutError('Original deadline; no automatic retry')
        available={int(r.split(',')[0]):int(r.split(',')[1]) for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        if admitted(available,c['gpus']):break
        time.sleep(30)
    save(root/'GPU_LAUNCH_CHECK.json',dict(gpus=[0,1],free_mib={g:available[g] for g in (0,1)},checked_at=time.time()))
    run(c);p=json.loads((root/'PROGRAM_STATE.json').read_text());p.update(resource_wait_seconds=p['started']-begun,whole_round_wall_seconds=time.time()-c['launched_at']);save(root/'PROGRAM_STATE.json',p)


if __name__=='__main__':
    c=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    try:main(c)
    except BaseException as e:
        p=Path(c['root'])/'PROGRAM_STATE.json';s=json.loads(p.read_text()) if p.exists() else dict(active={},costs=[],failures=[])
        s.update(status='INCOMPLETE',coordinator_error=str(e),ended=time.time());save(p,s);raise
