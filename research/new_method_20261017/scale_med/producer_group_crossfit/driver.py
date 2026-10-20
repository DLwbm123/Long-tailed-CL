"""One shared deadline across all prechecks and formal phases."""
import json,os,subprocess,time,resource
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['root']);start=time.monotonic();receipts=[];error=None
try:
 for phase in ['synthetic','inject','prep','preflight','train','evaluate']:
  env=os.environ.copy();env['RUN_CONFIG']=str(root/(phase+'.config.private.json'));log=(root/(phase+'.coordinate.private.log')).open('w')
  p=subprocess.Popen([cfg['python'],cfg['coordinate']],env=env,stdout=log,stderr=subprocess.STDOUT);(root/(phase+'.LAUNCH.private.json')).write_text(json.dumps(dict(pid=p.pid,parent_pid=os.getpid())));p.wait();log.close()
  out=root/phase;r=json.loads((out/'COORDINATOR_EXIT.json').read_text());receipts.append(dict(phase=phase,**r));expected=1 if phase=='inject' else 0
  assert r['worker_exit']==expected and r['stop_reason'] is None,'PHASE_EXIT_OR_GUARD'
  if phase=='inject':
   assert all((out/n).exists() for n in ['FAILURE.json','EXIT.json','WORKER_RESOURCE_RECEIPT.json','COORDINATOR_EXIT.json'])
   assert json.loads((out/'FAILURE.json').read_text())['error']=='RuntimeError'
  if phase in ['synthetic','preflight']:assert json.loads((out/'aggregate.json').read_text())['pass_all']
except BaseException as e:
 error=type(e).__name__+':'+str(e);(root/'DRIVER_FAILURE.json').write_text(json.dumps(dict(error=error,no_implicit_retry=True)))
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);(root/'DRIVER_EXIT.json').write_text(json.dumps(dict(phases=receipts,wall_seconds=time.monotonic()-start,driver_CPU_seconds=u.ru_utime+u.ru_stime,error=error,result_available=(root/'evaluate/aggregate.json').exists()),indent=2))
