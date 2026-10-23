import json,os,subprocess,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
for name in ['NIH','ISIC','HK']:
 d=out/name
 if (d/'STATUS.json').exists() and json.loads((d/'STATUS.json').read_text()).get('phase')=='COMPLETE':continue
 spec=json.loads((d/'config.private.json').read_text())
 templates=d/'runtime/knowledge/virtual_templates.json'
 templates.parent.mkdir(parents=True,exist_ok=True)
 key=spec.get('runtime_dataset',spec['dataset'])
 templates.write_text(json.dumps({key:[spec['parent_template']]}))
 assert json.loads(templates.read_text())[key]==[spec['parent_template']]
 env=os.environ.copy();env['RUN_CONFIG']=str(d/'config.private.json')
 (out/'STATUS.json').write_text(json.dumps(dict(phase='RUNNING',dataset=name,time=time.time())))
 with (d/'worker.private.log').open('a') as f:
  code=subprocess.call([cfg['python'],'/tmp/q1035/worker.py'],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=d/'runtime')
 if code:
  (out/'STATUS.json').write_text(json.dumps(dict(phase='FAILED',dataset=name,exit_code=code,time=time.time())))
  raise RuntimeError(f'Dataset {name} failed with {code}')
(out/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE',datasets=['NIH','ISIC','HK'],time=time.time())))
