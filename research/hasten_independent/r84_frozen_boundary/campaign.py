import os,json,subprocess,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
for entry,tag in [('software.py','SOFTWARE'),('preflight.py','NATIVE'),('train.py','TRAIN')]:
 if tag=='SOFTWARE' and (r/'SOFTWARE_CHECK.json').exists():
  assert json.loads((r/'SOFTWARE_CHECK.json').read_text())['passed'];continue
 e=os.environ.copy()
 if tag=='SOFTWARE':e['CUDA_VISIBLE_DEVICES']=''
 write(r/'STATUS.json',dict(phase=tag,time=time.time()))
 with (r/(tag+'.private.log')).open('w') as f:
  p=subprocess.Popen([cfg['python'],'/run/experiment/'+entry],env=e,cwd=cfg['runtime'],stdout=f,stderr=subprocess.STDOUT);write(r/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=tag));code=p.wait()
 assert code==0,(tag,code)
 if tag=='SOFTWARE':write(r/'SOFTWARE_CHECK.json',json.loads((r/'SOFTWARE.private.log').read_text()))

for entry in ['verify.py','analyze.py']:
 write(r/'STATUS.json',dict(phase=entry,time=time.time()))
 p=subprocess.Popen([cfg['python'],'/run/experiment/'+entry],env=os.environ.copy(),cwd=cfg['runtime'])
 write(r/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=entry));assert p.wait()==0,entry
write(r/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
