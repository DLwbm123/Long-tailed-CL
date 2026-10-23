import json,os,subprocess,time,signal
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
for name in ['NIH','ISIC','HK']:
 arm='geometry_priority';d=out/name
 if (d/'STATUS.json').exists() and json.loads((d/'STATUS.json').read_text()).get('phase')=='COMPLETE':continue
 spec=json.loads((d/'config.private.json').read_text())
 templates=d/'runtime/knowledge/virtual_templates.json'
 templates.parent.mkdir(parents=True,exist_ok=True)
 key=spec.get('runtime_dataset',spec['dataset'])
 templates.write_text(json.dumps({key:[spec['parent_template']]}))
 assert json.loads(templates.read_text())[key]==[spec['parent_template']]
 env=os.environ.copy();env['RUN_CONFIG']=str(d/'config.private.json')
 (out/'STATUS.json').write_text(json.dumps(dict(phase='RUNNING',arm=arm,dataset=name,time=time.time())))
 with (d/'worker.private.log').open('a') as f:
  child=subprocess.Popen([cfg['python'],'/tmp/q1063/worker.py'],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=d/'runtime')
  while child.poll() is None:
   try:rss=int(next(v for v in Path(f'/proc/{child.pid}/status').read_text().splitlines() if v.startswith('VmRSS:')).split()[1])*1024
   except (FileNotFoundError,StopIteration):rss=0
   if rss>8*2**30:os.killpg(os.getpgrp(),signal.SIGTERM)
   time.sleep(1)
  code=child.returncode
 if code:
  (out/'STATUS.json').write_text(json.dumps(dict(phase='FAILED',arm=arm,dataset=name,exit_code=code,time=time.time())))
  raise RuntimeError(f'Dataset {name} failed with {code}')
(out/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE',datasets=['NIH','ISIC','HK'],time=time.time())))
