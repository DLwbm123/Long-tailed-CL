"""Authorized main four-grid only; historical controls are never GPU jobs."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_module_cycle import run,save
VERSION='supported_old_margin_v1'

ORDERS={'ISIC':[4,7,5,1,2,6,0,3], 'HK':[4,5,6,22,20,1,2,15,14,12,10,11,19,16,7,0,17,9,18,13,21,8,3]}
SIZES={'ISIC':[4,2,2],'HK':[13,2,2,2,2,2]}
STEPS={'ISIC':[100,72,296],'HK':[98,4,2,20,18,34]}


def validate_job(j):
    from lca_supported import config_enabled
    if not config_enabled(j):raise ValueError('Active LCA main-four-grid required')
    ds=j['name'];pre=j.get('preflight',False)
    if (j['order']!=ORDERS[ds] or j['task_sizes']!=SIZES[ds] or j['seed']!=74002 or j['split_seed']!=74002
        or j['epochs']!=2 or j['batch_size']!=64 or j['lr']!=.0003
        or j['workers']!=(0 if pre else 4) or j['expected_steps']!=(0 if pre else sum(STEPS[ds]))
        or j['max_steps']!=sum(STEPS[ds]) or j.get('checkpoint_root')
        or not j.get('source_commit')):raise ValueError('Frozen scientific budget/order differs')


def validate(c):
    if c['gpus']!=[0,1] or len(c['jobs'])!=4 or len(c['preflights'])!=4 or c.get('version')!=VERSION:
        raise ValueError('Exactly four new main-grid trajectories on GPU01')
    if {(j['name'],j['module']) for j in c['jobs']}!={(ds,m) for ds in ORDERS for m in ('base','local_transport')}:
        raise ValueError('No contribution ablation or historical control retraining')
    for j,p in zip(c['jobs'],c['preflights']):
        validate_job(j);validate_job(p)
        for k in ('name','module','order','task_sizes','seed','split_seed','epochs','lr','batch_size',
                  'manifest','images','weight','legacy_repo','source_commit','lca_version','lca_map_ridge','lca_curvature_cap'):
            if j[k]!=p[k]:raise ValueError('Preflight differs from formal scientific configuration')
    if len({j['output'] for j in c['jobs']+c['preflights']})!=8:raise ValueError('Fresh separate preflight/train outputs required')


def read_training(c):
    with (Path(c['manifest'])/'train.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        p=Path(r['relative_path'])
        if r['split']!='train' or p.is_absolute() or '..' in p.parts or not r['identity_component']:
            raise ValueError('Invalid training manifest')
        r['label']=int(r['original_label'])
    if not rows or len({r['relative_path'] for r in rows})!=len(rows) or {r['label'] for r in rows}!=set(c['order']):
        raise ValueError('Training support/order differs')
    return rows


def checkpoint_metadata(c,task,steps):
    return {k:c[k] for k in ('name','module','order','seed','split_seed','source_commit','lca_version','lca_map_ridge','lca_curvature_cap')} | dict(task=task,steps=steps)


def check_barrier(c):
    campaign=json.loads(Path(os.environ['Q128_CONFIG']).read_text());validate(campaign)
    gate=json.loads(Path(c['evaluation_gate']).read_text())
    if gate['phase']!='EVALUATE' or not gate.get('training_sealed_at'):raise ValueError('Validation remains sealed')
    for j in campaign['jobs']:
        s=json.loads((Path(j['output'])/'STATUS.json').read_text())
        if s['status'] not in ('TRAINED','COMPLETE') or s['steps']!=j['expected_steps'] or s['stages']!=len(j['task_sizes']) or s['policy_updates']:
            raise ValueError('Exact four-trajectory training barrier not met')


def main(c):
    validate(c);root=Path(c['root']);begun=time.time()
    save(root/'PROGRAM_STATE.json',dict(status='RUNNING',phase='WAITING_GPU01',active={},costs=[],failures=[]))
    if not subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip().startswith('nfs') or shutil.disk_usage(root).free<30*1024**3:
        raise ValueError('NAS admission failed')
    probe=root/'.admission_probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    while True:
        if time.time()>=c['deadline']:raise TimeoutError('Original deadline; no automatic retry')
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        available={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows}
        if all(available[g]>=32768 for g in c['gpus']):break
        time.sleep(30)
    save(root/'GPU_LAUNCH_CHECK.json',dict(gpus=[0,1],free_mib={g:available[g] for g in (0,1)},checked_at=time.time()))
    run(c);p=json.loads((root/'PROGRAM_STATE.json').read_text());p.update(resource_wait_seconds=p['started']-begun,whole_round_wall_seconds=time.time()-c['launched_at']);save(root/'PROGRAM_STATE.json',p)


if __name__=='__main__':
    config=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    try:main(config)
    except BaseException as e:
        p=Path(config['root'])/'PROGRAM_STATE.json';s=json.loads(p.read_text()) if p.exists() else dict(active={},costs=[],failures=[])
        s.update(status='INCOMPLETE',coordinator_error=str(e),ended=time.time());save(p,s);raise
