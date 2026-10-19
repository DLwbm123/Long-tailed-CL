"""Bound one CPU child, count complete child and coordinator resources."""
import json,os,resource,subprocess,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
assert not (out/'LAUNCH.private.json').exists()
start=time.monotonic();log=(out/'worker.private.log').open('w');p=subprocess.Popen([cfg['python'],cfg['entry']],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
(out/'LAUNCH.private.json').write_text(json.dumps(dict(parent_pid=os.getpid(),pid=p.pid,argv=[cfg['python'],cfg['entry']],source_commit=cfg['source_commit'])))
reason=None;terminated=None
while True:
 got,status,usage=os.wait4(p.pid,os.WNOHANG)
 if got:break
 try:
  fields=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split();cpu=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK');own=resource.getrusage(resource.RUSAGE_SELF)
  if cpu+own.ru_utime+own.ru_stime>cfg['cpu_budget']-2:reason=reason or 'CPU_LIMIT'
  if int(fields[21])*os.sysconf('SC_PAGE_SIZE')>2**30:reason=reason or 'RSS_LIMIT'
 except FileNotFoundError:pass
 if time.monotonic()-start>cfg['wall_budget']-3:reason=reason or 'WALL_LIMIT'
 if sum(f.stat().st_size for f in Path(cfg['root']).rglob('*') if f.is_file())>16*2**20:reason=reason or 'OUTPUT_LIMIT'
 if reason:
  if terminated is None:p.terminate();terminated=time.monotonic()
  elif time.monotonic()-terminated>1:p.kill()
 time.sleep(.2)
p.returncode=os.waitstatus_to_exitcode(status);log.close();own=resource.getrusage(resource.RUSAGE_SELF)
r=dict(worker_exit=p.returncode,stop_reason=reason,wall_seconds=time.monotonic()-start,worker_CPU_seconds=usage.ru_utime+usage.ru_stime,coordinator_CPU_seconds=own.ru_utime+own.ru_stime,worker_peak_RSS_bytes=usage.ru_maxrss*1024,source_commit=cfg['source_commit'])
(out/'COORDINATOR_EXIT.json').write_text(json.dumps(r,indent=2))
if not (out/'EXIT.json').exists():(out/'EXIT.json').write_text(json.dumps(dict(exit_code=p.returncode,decision='UNRESOLVED_RESOURCE_OR_SIGNAL')))
if p.returncode and not (out/'FAILURE.json').exists():(out/'FAILURE.json').write_text(json.dumps(dict(error=reason or 'PROCESS_FAILURE',no_retry=True)))
