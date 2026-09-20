"""One finite worker sequence after strict engineering and archive admission."""
import os,time,subprocess,json,hashlib
from pathlib import Path

def main():
    r=Path('/tmp/p30root');pub=r/'output/public';read=lambda name:json.loads((pub/name).read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==read('LAUNCH_LOCK.json')['driver_sha256']
    assert read('STORAGE_GATE.json')['status']=='PASS'
    assert not (pub/'DRIVER_STARTED.json').exists()
    (pub/'DRIVER_STARTED.json').write_text(json.dumps(dict(unix=time.time(),pid=os.getpid())))
    env=dict(os.environ,P30_CONFIG=str(r/'config.json'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0');receipts=[]
    for mode in ('qualify','evaluate','report'):
        env['P30_MODE']=mode
        if mode=='report':env['CUDA_VISIBLE_DEVICES']=''
        t=time.time()
        with (r/'output'/f'{mode}.log').open('xb') as f:
            try:code=subprocess.run(['/tmp/n154env/bin/python','-u','/tmp/p30r.py' if mode=='report' else '/tmp/p30.py'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,timeout=600 if mode=='report' else None).returncode
            except subprocess.TimeoutExpired:code=124
        (r/'output'/f'{mode}.exit').write_text(str(code)+'\n');receipts.append(dict(stage=mode,exit_code=code,started=t,finished=time.time(),process_residence_seconds=time.time()-t))
        (pub/'PROCESS_RECEIPTS.json').write_text(json.dumps(receipts,indent=2))
        if code:
            (pub/'DRIVER_FAILURE.json').write_text(json.dumps(dict(stage=mode,exit_code=code),indent=2));break
    (r/'output/private/STOP_TRANSFER.json').write_text(json.dumps(dict(status='STOP',exit_code=code)))
if __name__=='__main__':main()
