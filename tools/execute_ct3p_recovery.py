"""Fixed CT3-P recovery driver; only remaining work, no automatic retries."""
import hashlib,json,math,os,subprocess,time
from pathlib import Path

def main():
    root=Path(os.environ['P22_ROOT']);pub=root/'output/public';base=json.loads((root/'runtime_locked.json').read_text())
    lock=json.loads((pub/'RECOVERY_LOCK_R2.json').read_text());limit=lock['GPU_limit_seconds']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==lock['recovery_driver_sha256']
    assert json.loads((pub/'RECOVERY_ADMISSION_R2.json').read_text())['status']=='PASS'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
    receipts=[]
    try:
        for phase in ('train','evaluate','oracle','report'):
            ledger=json.loads((pub/'RESOURCE_AND_ACCESS_LEDGER.json').read_text())
            remaining=math.floor((7200-ledger.get('CPU_analytic_seconds',0)-600-30) if phase=='report' else (limit-ledger['GPU_process_residence_seconds']-30))
            assert remaining>0,'BLOCKED_REMAINING_BUDGET'
            cfg=dict(base,mode=phase);path=root/('runtime_'+phase+'.json');path.write_text(json.dumps(cfg,indent=2));env['P22_CONFIG']=str(path)
            env['P22_RECOVERY_MODE']='resume' if phase=='train' else phase
            if phase=='report':env['CUDA_VISIBLE_DEVICES']=''
            began=time.time()
            with (root/'output'/(phase+'_r2.log')).open('xb') as f:
                result=subprocess.run(['timeout','-k','10',str(remaining),'/tmp/n154env/bin/python','-u','/tmp/p22r.py' if phase=='report' else '/tmp/p22resume.py'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f)
            (root/'output'/(phase+'_r2.exit')).write_text(str(result.returncode)+'\n')
            receipts.append(dict(phase=phase,exit_code=result.returncode,started=began,finished=time.time(),hard_timeout_seconds=remaining))
            (pub/'PHASE_RECEIPTS_R2.json').write_text(json.dumps(receipts,indent=2))
            assert result.returncode==0,('PHASE_FAILED',phase,result.returncode)
        (pub/'PIPELINE_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_CT3P',NEXT_DECISION='STOP',unix=time.time()),indent=2))
    except BaseException as e:
        (pub/'PIPELINE_FAILURE_R2.json').write_text(json.dumps(dict(status='BLOCKED',reason=str(e),unix=time.time(),preserve_partial=True,automatic_retry=False),indent=2));raise
    finally:(root/'output/private/STOP_TRANSFER.json').write_text(json.dumps(dict(status='STOP')))
if __name__=='__main__':main()
