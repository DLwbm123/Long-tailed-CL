import os,json,subprocess,time,signal
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output'])
def run(entry,spec,label):
 env=os.environ.copy();env['RUN_CONFIG']=str(spec)
 with (root/(label+'.private.log')).open('a') as f:
  p=subprocess.Popen([cfg['python'],entry],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=root/'runtime');(root/'ACTIVE_CHILD.private.json').write_text(json.dumps(dict(pid=p.pid,job=label)));(root/'STATUS.json').write_text(json.dumps(dict(phase='RUNNING',job=label,time=time.time())))
  while p.poll() is None:
   try:rss=int(next(v for v in Path(f'/proc/{p.pid}/status').read_text().splitlines() if v.startswith('VmRSS:')).split()[1])*1024
   except (FileNotFoundError,StopIteration):rss=0
   if rss>8*2**30:os.killpg(os.getpgrp(),signal.SIGTERM)
   time.sleep(2)
  assert p.returncode==0,(label,p.returncode)
if not (root/'PREFLIGHT.json').exists():run('/tmp/q1083/preflight.py',Path(os.environ['RUN_CONFIG']),'preflight')
for ds in ['NIH','ISIC','HK']:
 d=root/ds
 if (d/'STATUS.json').exists() and json.loads((d/'STATUS.json').read_text()).get('phase')=='COMPLETE':continue
 run('/tmp/q1083/worker.py',d/'config.private.json',ds)
(root/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE',time=time.time())))
