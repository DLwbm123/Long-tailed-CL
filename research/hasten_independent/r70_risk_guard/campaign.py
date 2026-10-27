import os,json,time,subprocess
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
def run(entry,config,label):
 e=os.environ.copy();e['RUN_CONFIG']=config
 with (r/(label+'.private.log')).open('a') as f:p=subprocess.Popen([cfg['python'],'/run/experiment/'+entry],env=e,cwd=cfg['runtime'],stdout=f,stderr=subprocess.STDOUT)
 write(r/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=label));write(r/'STATUS.json',dict(phase='RUNNING',job=label,time=time.time()));assert p.wait()==0
if not (r/'GRPO_PREFLIGHT.json').exists():run('preflight.py','/run/experiment/config.private.json','PREFLIGHT')
if not (r/'HK/aggregate.json').exists():run('train.py','/run/experiment/HK/config.private.json','HK_POSTTRAIN')
write(r/'STATUS.json',dict(phase='COMPLETE',time=time.time(),scope='HK74002 supervised-risk direction factorial'))
