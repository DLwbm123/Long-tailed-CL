import json,os,subprocess,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
for name in ['ISIC','HK']:
 d=out/name
 if (d/'STATUS.json').exists() and json.loads((d/'STATUS.json').read_text()).get('phase')=='COMPLETE':continue
 env=os.environ.copy();env['RUN_CONFIG']=str(d/'config.private.json')
 (out/'STATUS.json').write_text(json.dumps(dict(phase='RUNNING',dataset=name,time=time.time())))
 with (d/'worker.private.log').open('a') as f:
  code=subprocess.call([cfg['python'],'/tmp/q1033/worker.py'],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=d/'runtime')
 if code:raise RuntimeError(f'Dataset {name} failed with {code}')
(out/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE',datasets=['NIH_reused','ISIC','HK'],time=time.time())))
