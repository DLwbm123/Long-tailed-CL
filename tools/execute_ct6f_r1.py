"""One finite worker sequence after strict engineering and archive admission."""
import os,time,subprocess,json,hashlib
from pathlib import Path

def main():
    r=Path('/tmp/p25root');pub=r/'output/public';read=lambda name:json.loads((pub/name).read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==read('RECOVERY_LOCK_R1.json')['driver_sha256']
    assert read('ENGINEERING_GATE.json')['status']=='PASS' and read('STORAGE_GATE.json')['status']=='PASS'
    assert not (pub/'DRIVER_STARTED_R1.json').exists()
    (pub/'DRIVER_STARTED_R1.json').write_text(json.dumps(dict(unix=time.time(),pid=os.getpid())))
    env=dict(os.environ,P25_CONFIG=str(r/'config.json'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0');receipts=[]
    for mode in ('train','evaluate','report'):
        env['P25_MODE']=mode
        if mode=='report':env['CUDA_VISIBLE_DEVICES']=''
        t=time.time()
        with (r/'output'/f'{mode}_r1.log').open('xb') as f:
            try:code=subprocess.run(['/tmp/n154env/bin/python','-u','/tmp/p25r.py' if mode=='report' else '/tmp/p25resume.py'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,timeout=600 if mode=='report' else None).returncode
            except subprocess.TimeoutExpired:code=124
        (r/'output'/f'{mode}_r1.exit').write_text(str(code)+'\n');receipts.append(dict(stage=mode,exit_code=code,started=t,finished=time.time(),process_residence_seconds=time.time()-t))
        (pub/'PROCESS_RECEIPTS_R1.json').write_text(json.dumps(receipts,indent=2))
        if code:
            (pub/'DRIVER_FAILURE_R1.json').write_text(json.dumps(dict(stage=mode,exit_code=code),indent=2));break
    (r/'output/private/STOP_TRANSFER.json').write_text(json.dumps(dict(status='STOP',exit_code=code)))
if __name__=='__main__':main()
