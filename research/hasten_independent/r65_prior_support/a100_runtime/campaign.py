import json,os,subprocess,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output'])
if not (root/'A100_MIGRATION_PREFLIGHT.json').exists():
 with (root/'migration_preflight.private.log').open('a') as f:
  p=subprocess.run([cfg['python'],'/tmp/q1108/migration_preflight.py'],env=os.environ.copy(),cwd=cfg['runtime'],stdout=f,stderr=subprocess.STDOUT);assert p.returncode==0
for ds in ['NIH','ISIC','HK']:
 d=root/ds
 if (d/'STATUS.json').exists() and json.loads((d/'STATUS.json').read_text())['phase']=='COMPLETE':continue
 env=os.environ.copy();env['RUN_CONFIG']=str(d/'config.private.json')
 with (root/(ds+'.a100.private.log')).open('a') as f:
  p=subprocess.Popen([cfg['python'],'/tmp/q1108/train.py'],env=env,cwd=cfg['runtime'],stdout=f,stderr=subprocess.STDOUT);write(root/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=ds));write(root/'STATUS.json',dict(phase='RUNNING_A100_ONLY',job=ds,time=time.time()));assert p.wait()==0
write(root/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
