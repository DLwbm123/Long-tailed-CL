import json,os,subprocess,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output'])
def run(entry,spec,label):
 env=os.environ.copy();env['RUN_CONFIG']=str(spec)
 with (root/(label+'.private.log')).open('a') as f:
  p=subprocess.Popen([cfg['python'],entry],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=cfg['runtime']);write(root/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=label));write(root/'STATUS.json',dict(phase='RUNNING',job=label,time=time.time()));code=p.wait();assert code==0,(label,code)
if not (root/'PREFLIGHT.json').exists() or not (root/'DECISION_PREFLIGHT.json').exists() or not (root/'BIRTH_PREFLIGHT.json').exists():run('/tmp/q1095/preflight.py',Path(os.environ['RUN_CONFIG']),'preflight')
for ds in ['NIH','ISIC','HK']:
 d=Path(cfg['domains'][ds]['output'])
 if (d/'STATUS.json').exists() and json.loads((d/'STATUS.json').read_text())['phase']=='COMPLETE':continue
 run('/tmp/q1095/train.py',d/'config.private.json',ds)
write(root/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
