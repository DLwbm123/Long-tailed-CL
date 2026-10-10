"""Prepare data, preserve the existing CIFAR queue, then run the frozen pair."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_module_cycle import run,save


def main(config):
    root=Path(config['root']);started=time.time()
    state=dict(status='RUNNING',phase='PREPARING_DATA',started=started,active={},costs=[],failures=[])
    save(root/'PROGRAM_STATE.json',state)
    begun=time.monotonic()
    with (root/'data_prepare.private.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',config['prepare_entry']],stdout=log,stderr=subprocess.STDOUT)
        state['data_prepare_pid']=p.pid;save(root/'PROGRAM_STATE.json',state);code=p.wait()
    save(root/'DATA_COST.json',dict(returncode=code,cpu_process_residence_seconds=time.monotonic()-begun))
    if code:raise RuntimeError('Data preparation failed; preserve outputs, no automatic retry')
    state['phase']='WAITING_EXISTING_CIFAR';save(root/'PROGRAM_STATE.json',state)
    # shortcut: single dependency queue; add a shared scheduler only for multiple concurrent campaigns.
    while True:
        if time.time()>=config['deadline']:raise TimeoutError('Original campaign deadline')
        previous=json.loads(Path(config['wait_for_program']).read_text())
        if previous['status'] in ('COMPLETE','INCOMPLETE') and not previous.get('active'):break
        time.sleep(15)
    mount=subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip()
    assert mount.startswith('nfs') and shutil.disk_usage(root).free>60*1024**3
    probe=root/('.gpu_probe_'+str(time.time_ns()));probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    assert config['gpus']==[0,1]
    while True:
        if time.time()>=config['deadline']:raise TimeoutError('Original campaign deadline')
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        available={int(v.split(',')[0]):int(v.split(',')[1]) for v in rows}
        if all(available[g]>=32768 for g in config['gpus']):break
        state['phase']='WAITING_GPU_01_MEMORY';save(root/'PROGRAM_STATE.json',state);time.sleep(30)
    save(root/'GPU_LAUNCH_CHECK.json',dict(gpus=[0,1],free_mib={str(g):available[g] for g in (0,1)},mount=mount,checked_at=time.time()))
    run(config)


if __name__=='__main__':
    config=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    try:main(config)
    except BaseException as exc:
        p=Path(config['root'])/'PROGRAM_STATE.json';state=json.loads(p.read_text())
        state.update(status='INCOMPLETE',coordinator_error=str(exc),ended=time.time());save(p,state);raise
