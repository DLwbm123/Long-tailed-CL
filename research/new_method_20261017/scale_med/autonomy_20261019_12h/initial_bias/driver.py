"""One sealed experiment, fixed sequential training then frozen stage fitting."""
import os,json,subprocess,time,resource
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['root']);start=time.monotonic();cells=[]
for name in ['train','fit']:
 env=os.environ.copy();env['RUN_CONFIG']=str(root/(name+'.config.private.json'));log=(root/(name+'.coordinate.private.log')).open('w');p=subprocess.Popen([cfg['modern_python'],cfg['coordinate']],env=env,stdout=log,stderr=subprocess.STDOUT);(root/('DRIVER_'+name+'_LAUNCH.private.json')).write_text(json.dumps(dict(pid=p.pid,parent_pid=os.getpid())));code=p.wait();log.close();phase=json.loads((root/(name+'.config.private.json')).read_text());out=Path(phase['output']);receipt=json.loads((out/'COORDINATOR_EXIT.json').read_text());cells.append(dict(phase=name,**receipt))
 if code or receipt['worker_exit']!=0:break
(root/'DRIVER_EXIT.json').write_text(json.dumps(dict(phases=cells,wall_seconds=time.monotonic()-start,driver_CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,scientific_result_available=(root/'fit/aggregate.json').exists()),indent=2))
