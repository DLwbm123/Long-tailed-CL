import json,os,subprocess,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
for entry,tag in [('software.py','SOFTWARE'),('preflight.py','SAVED_FEATURE_NATIVE'),('train.py','TRAIN'),('verify.py','VERIFY'),('analyze.py','ANALYZE')]:
 write(r/'STATUS.json',dict(phase=tag,time=time.time()))
 e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']=''
 with (r/(tag+'.private.log')).open('w') as f:
  p=subprocess.Popen([cfg['python'],'/run/experiment/'+entry],env=e,cwd=cfg['runtime'],stdout=f,stderr=subprocess.STDOUT)
  write(r/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=tag));assert p.wait()==0,(entry,p.returncode)
 if tag=='SOFTWARE':write(r/'SOFTWARE_CHECK.json',json.loads((r/'SOFTWARE.private.log').read_text()))
write(r/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
