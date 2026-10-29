import os,json,subprocess,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output'])
for entry,tag in [('check_decision.py','SOFTWARE'),('preflight.py','NATIVE'),('train.py','TRAIN')]:
 env=os.environ.copy()
 if tag=='SOFTWARE':env['CUDA_VISIBLE_DEVICES']=''
 write(root/'STATUS.json',dict(phase=tag,time=time.time()))
 with (root/(tag+'.private.log')).open('w') as f:
  p=subprocess.Popen([cfg['python'],'/run/experiment/'+entry],env=env,cwd=cfg['runtime'],stdout=f,stderr=subprocess.STDOUT)
  write(root/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=tag));code=p.wait()
 assert code==0,(tag,code)
 if tag=='SOFTWARE':write(root/'SOFTWARE_CHECK.json',json.loads((root/'SOFTWARE.private.log').read_text()))
write(root/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
