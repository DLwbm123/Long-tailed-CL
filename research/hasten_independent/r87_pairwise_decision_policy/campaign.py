import os,json,time,subprocess
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
for entry,tag in [('software.py','SOFTWARE'),('preflight.py','NATIVE'),('train.py','TRAIN'),('verify.py','VERIFY'),('analyze.py','ANALYZE')]:
 write(r/'STATUS.json',dict(phase=tag,time=time.time()))
 with (r/(tag+'.private.log')).open('w') as f:
  p=subprocess.Popen([cfg['python'],'/run/experiment/'+entry],env=os.environ.copy(),cwd='/run/experiment',stdout=f,stderr=subprocess.STDOUT);write(r/'ACTIVE_CHILD.private.json',dict(pid=p.pid,job=tag));ps=subprocess.check_output(['ps','-ww','-o','pid,ppid,stat,args','-p',str(os.getpid())+','+str(p.pid)],text=True);assert not any(v in ps.lower() for v in json.loads(os.environ.get('FORBIDDEN_PROCESS_TOKENS_JSON','[]')));(r/(tag+'_ps.private.txt')).write_text(ps);assert p.wait()==0,(entry,p.returncode)
 if tag=='SOFTWARE':write(r/'SOFTWARE_CHECK.json',json.loads((r/'SOFTWARE.private.log').read_text()))
write(r/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
