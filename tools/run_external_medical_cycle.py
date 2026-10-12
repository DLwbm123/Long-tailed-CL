"""Four native-method jobs; closed protected deliveries and GPU01 admission."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from external_medical_compare import validate_access
from run_external_medical import validate, VERSION
from run_module_cycle import run, save


def validate_campaign(config):
    if config['gpus']!=[0,1] or config['version']!=VERSION or len(config['preflights'])!=4:
        raise ValueError('Only four GPU01 preflights authorized')
    expected=[('ISIC','AIR'),('HK','AIR'),('ISIC','APART_full'),('HK','APART_full')]
    if [(j['dataset'],j['method']) for j in config['jobs']]!=expected:
        raise ValueError('Frozen four-trajectory order differs')
    for j,p in zip(config['jobs'],config['preflights']):
        validate(j);validate(p)
        if p.get('preflight') is not True or j.get('preflight'):
            raise ValueError('Preflight-only shadow scope differs')
        for k in ('dataset','method','seed','order','task_sizes','version','protocol_lock',
                  'manifest','images','weight','source_commit','expected_steps','expected_analytic_batches'):
            if j[k]!=p[k]:raise ValueError('Preflight config differs from formal job')
    if len({j['output'] for j in config['jobs']+config['preflights']})!=8:
        raise ValueError('Fresh output scopes required')


def main(config):
    validate_campaign(config);root=Path(config['root']);begun=time.time()
    receipts={}
    for name,local in config['protected_delivery_receipts'].items():
        remote=json.loads((root.parent/name/'DELIVERY.private.json').read_text())
        receipts[name]=(local,remote)
        state=json.loads((root.parent/name/'PROGRAM_STATE.json').read_text())
        if state['status']!='COMPLETE' or state['active']:raise ValueError('Protected round not complete')
    validate_access(config['jobs'][0],'train',receipts)
    display=subprocess.check_output(['ps','-eo','args'],text=True)
    if any('/tmp/'+entry in display for entry in ('q151w.py','q153w.py')):
        raise ValueError('Protected workers still active')
    mount=subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip()
    if not mount.startswith('nfs') or shutil.disk_usage(root).free<40*1024**3:
        raise ValueError('NAS mount/free admission failed')
    probe=root/('.probe_'+str(time.time_ns()));probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    while True:
        if time.time()>=config['deadline']:raise TimeoutError('Original round deadline')
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        free={int(line.split(',')[0]):int(line.split(',')[1]) for line in lines}
        if all(free[g]>=32768 for g in (0,1)):break
        save(root/'PROGRAM_STATE.json',dict(status='RUNNING',phase='WAITING_GPU_01_MEMORY',active={},costs=[],failures=[]))
        time.sleep(30)
    save(root/'GPU_LAUNCH_CHECK.json',dict(mount=mount,free_mib={g:free[g] for g in (0,1)},checked_at=time.time()))
    run(config)
    s=json.loads((root/'PROGRAM_STATE.json').read_text());s.update(resource_wait_seconds=s['started']-begun,
        whole_round_wall_seconds=time.time()-config['launched_at']);save(root/'PROGRAM_STATE.json',s)


if __name__=='__main__':
    config=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    try:main(config)
    except BaseException as exc:
        path=Path(config['root'])/'PROGRAM_STATE.json'
        state=json.loads(path.read_text()) if path.exists() else dict(active={},costs=[],failures=[])
        state.update(status='INCOMPLETE',coordinator_error=str(exc),ended=time.time());save(path,state);raise
