"""Run the one locked F hypothesis, at most two workers, then report and stop."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from run_medical_v2 import sha,write_json

def verify(config):
    for name,h in config['code_sha256'].items():assert sha(Path(config['code_root'])/name)==h,'BLOCKED_CODE_DRIFT '+name
    assert sha(Path(config['output'])/'PROTOCOL_LOCK.json')==config['protocol_sha256']
    summary=json.loads((Path(config['protocol'])/'V2_DATASET_SUMMARY.json').read_text())
    for split,h in summary['manifest_sha256'].items():assert sha(Path(config['protocol'])/(split+'.csv'))==h
    assert json.loads((Path(config['v3_complete_output'])/'FINAL_STATUS.json').read_text())['status']=='COMPLETE_P0_P1_P2'

def worker(config,phase,seed=1993):
    import torch
    torch.set_num_threads(4);verify(config)
    if phase=='engineering':
        from test_stationary_medical_v4 import engineering
        return engineering(config)
    from stationary_medical_v4 import trajectory
    return trajectory(config,seed,pilot=phase=='pilot')

def supervisor(config):
    out=Path(config['output']);verify(config)
    assert json.loads((out/'engineering/ENGINEERING.json').read_text())['status']=='PASS'
    start=time.time();active={};jobs=[]
    def launch(phase,seed):
        env=os.environ.copy();env['N4_CONFIG']=config['runtime_path'];env['N4_JOB']=json.dumps(dict(phase=phase,seed=seed))
        log=(out/f'{phase}_{seed}.log').open('a')
        p=subprocess.Popen([sys.executable,config['neutral_worker']],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[p.pid]=(p,log,phase,seed,time.time());return p
    def collect(p):
        p,log,phase,seed,t=active.pop(p.pid);log.close()
        jobs.append(dict(phase=phase,seed=seed,seconds=time.time()-t,returncode=p.returncode))
        write_json(out/'RESOURCE.json',dict(jobs=jobs,wall_seconds=time.time()-start,process_seconds=sum(j['seconds'] for j in jobs),budget_gpu_hours=None))
        assert p.returncode==0,'BLOCKED_WORKER '+str(seed)+' '+phase
    def sample():
        s=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
        with (out/'gpu_resource.csv').open('a') as f:f.write(s)
    try:
        write_json(out/'STATUS.json',{'status':'RUNNING_PILOT','test_predictions':0})
        if not (out/'1993_F/PILOT.json').exists():
            p=launch('pilot',1993)
            while p.poll() is None:sample();time.sleep(15)
            collect(p)
        pilot=json.loads((out/'1993_F/PILOT.json').read_text());e=pilot['epoch'];rate=e['components']['training_seconds']/e['components']['train_images']
        counts=[10529,3263,2835,1306,458,256,44,27]
        images=sum(sum(counts[c] for c in order[4:]) for order in config['class_orders'].values())
        projected=1.5*(rate*images*11+max(0,e['seconds']-e['components']['training_seconds'])*60+180)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
        need=2*(pilot['peak_allocated_bytes']+1024**3)+512*1024**2
        slots=2 if free>=need else 1
        assert free>pilot['peak_allocated_bytes']+1024**3,'BLOCKED_GPU_MEMORY'
        disk=shutil.disk_usage(out).free;required=128*1024**2+1024**3
        assert disk>required,'BLOCKED_STORAGE'
        write_json(out/'RESOURCE_GATE.json',{'status':'PASS','slots':slots,'estimated_gpu_process_seconds':projected,'budget_gpu_hours':None,
                   'pilot_seconds_per_image':rate,'peak_gpu_bytes':pilot['peak_allocated_bytes'],'free_gpu_bytes':free,
                   'free_disk_bytes':disk,'required_disk_bytes':required,'checkpoint_format':'S0_HEAD_DELTA_V1','pilot_checkpoint_bytes':pilot['checkpoint_bytes'],
                   'retention':'six session checkpoints + three resumes; original parents preserved','new_epochs':60,'pilot_is_first_formal_epoch':True})
        queue=[1993,1994,1995];write_json(out/'STATUS.json',{'status':'RUNNING','slots':slots,'planned_epochs':60,'test_predictions':0})
        while queue or active:
            while queue and len(active)<slots:launch('train',queue.pop(0))
            sample();time.sleep(15)
            for p,*_ in list(active.values()):
                if p.poll() is not None:collect(p)
        from stationary_medical_v4 import report
        report(config)
    except Exception as error:
        write_json(out/'STATUS.json',{'status':'BLOCKED','error':repr(error),'traceback':traceback.format_exc(),'active_workers':[x[3] for x in active.values()],'test_predictions':0})
        raise
