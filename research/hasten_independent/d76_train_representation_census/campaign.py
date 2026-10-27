import os,json,subprocess,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
def run(name,config,label):
 e=os.environ.copy();e['RUN_CONFIG']=config
 with (r/(label+'.private.log')).open('a') as f:p=subprocess.Popen([cfg['python'],'/run/experiment/'+name],env=e,cwd=cfg['runtime'],stdout=f,stderr=subprocess.STDOUT)
 write(r/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=label));write(r/'STATUS.json',dict(time=time.time(),phase='RUNNING',job=label));assert p.wait()==0
if not (r/'SOFTWARE_CHECK.json').exists():
 e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']='';p=subprocess.run([cfg['python'],'/run/experiment/check_diagnostics.py'],env=e,text=True,capture_output=True);(r/'SOFTWARE_CHECK.private.log').write_text(p.stdout+p.stderr);assert p.returncode==0;write(r/'SOFTWARE_CHECK.json',json.loads(p.stdout))
run('preflight.py','/run/experiment/config.private.json','PREFLIGHT')
run('train.py','/run/experiment/HK/config.private.json','FULL_TRAIN_CENSUS')
write(r/'STATUS.json',dict(time=time.time(),phase='COMPLETE',optimizer_steps=0,VAL_TEST_reads=0))
