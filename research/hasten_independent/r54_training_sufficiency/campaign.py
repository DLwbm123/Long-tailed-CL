import os,json,time,subprocess,signal
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);jobs=[]
for ds in ['NIH','ISIC','HK']:
 for mode in ['native','flat','factorized']:
  for fold in [0,1]:jobs.append(('audit',mode,ds,fold,root/'audit'/mode/ds/str(fold),'/tmp/q1080/audit_worker.py'))
for mode in ['native','flat','factorized']:
 for ds in ['NIH','ISIC','HK']:jobs.append(('full',mode,ds,None,root/'full'/mode/ds,'/tmp/q1080/worker.py'))
for phase,mode,ds,fold,d,entry in jobs:
 if (d/'STATUS.json').exists() and json.loads((d/'STATUS.json').read_text()).get('phase')=='COMPLETE':continue
 spec=d/'config.private.json';c=json.loads(spec.read_text());runtime=d/'runtime';runtime.mkdir(exist_ok=True);(runtime/'knowledge').mkdir(exist_ok=True);(runtime/'knowledge/virtual_templates.json').write_text(json.dumps({c.get('runtime_dataset',c['dataset']):[c['parent_template']]}))
 (root/'STATUS.json').write_text(json.dumps(dict(phase=phase,mode=mode,dataset=ds,fold=fold,time=time.time())));env=os.environ.copy();env['RUN_CONFIG']=str(spec)
 with (d/'worker.private.log').open('a') as log:
  p=subprocess.Popen([cfg['python'],entry],env=env,cwd=runtime,stdout=log,stderr=subprocess.STDOUT)
  (root/'ACTIVE_CHILD.private.json').write_text(json.dumps(dict(pid=p.pid,phase=phase,mode=mode,dataset=ds,fold=fold)))
  while p.poll() is None:
   try:rss=int(next(s for s in Path(f'/proc/{p.pid}/status').read_text().splitlines() if s.startswith('VmRSS:')).split()[1])*1024
   except (FileNotFoundError,StopIteration):rss=0
   if rss>8*2**30:os.killpg(os.getpgrp(),signal.SIGTERM)
   time.sleep(2)
  assert p.returncode==0,(phase,mode,ds,fold,p.returncode)
(root/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE',time=time.time())))
