"""GPU01-only sealed four-reconstruction queue, without retries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from paper_medical_models import VERSION
from run_paper_medical import validate
from run_module_cycle import run,save


def main(c):
    expected=[('ISIC','DAT_paper'),('HK','DAT_paper'),('ISIC','Hao_paper'),('HK','Hao_paper')]
    if c['gpus']!=[0,1] or c['version']!=VERSION or [(j['dataset'],j['method']) for j in c['jobs']]!=expected or len(c['preflights'])!=4:
        raise ValueError('Frozen four-trajectory GPU01 scope')
    for j,p in zip(c['jobs'],c['preflights']):
        validate(j);validate(p)
        if not p.get('preflight') or j.get('preflight'):raise ValueError('Zero update scope')
        if any(j[k]!=p[k] for k in ('dataset','method','seed','order','task_sizes','source_commit','expected_steps','weight','resnet_weight')):raise ValueError('Preflight match')
    root=Path(c['root']);begun=time.time()
    if not subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip().startswith('nfs') or shutil.disk_usage(root).free<40*1024**3:
        raise ValueError('NAS admission')
    probe=root/('.probe_'+str(time.time_ns()));probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    for name,receipt in c['protected_delivery_receipts'].items():
        d=json.loads((root.parent/name/'DELIVERY.private.json').read_text())
        if any(d[k]!=receipt[k] for k in ('status','commit','anonymous_http_status','nas_saved')) or d['status']!='DELIVERED':raise ValueError('Prior delivery not closed')
    command=subprocess.check_output(['ps','-eo','args'],text=True)
    if any('/tmp/'+v in command for v in ('q155w.py','q157w.py')):raise ValueError('Protected workers active')
    while True:
        if time.time()>=c['deadline']:raise TimeoutError('Original campaign deadline')
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(v.split(',')[0]):int(v.split(',')[1]) for v in output.splitlines()}
        if all(free[g]>=32768 for g in (0,1)):break
        save(root/'PROGRAM_STATE.json',dict(status='RUNNING',phase='WAITING_GPU01',active={},costs=[],failures=[]));time.sleep(30)
    run(c)
    p=json.loads((root/'PROGRAM_STATE.json').read_text());p.update(whole_round_wall_seconds=time.time()-c['launched_at'],resource_wait_seconds=p['started']-begun)
    save(root/'PROGRAM_STATE.json',p)


if __name__=='__main__':
    c=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    try:main(c)
    except BaseException as e:
        path=Path(c['root'])/'PROGRAM_STATE.json';p=json.loads(path.read_text()) if path.exists() else dict(active={},costs=[],failures=[])
        p.update(status='INCOMPLETE',coordinator_error=str(e),ended=time.time());save(path,p);raise
