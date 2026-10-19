"""No time/CPU/GPU duration cap; bound this window's new disk usage."""
import os,json,time,subprocess,resource
from pathlib import Path
from space import check_space,used
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);assert not (out/'LAUNCH.private.json').exists();start=time.monotonic();log=(out/'worker.private.log').open('w');check_space(cfg['window_root'])
p=subprocess.Popen([cfg['python'],cfg['entry']],cwd=cfg['root'],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
(out/'LAUNCH.private.json').write_text(json.dumps(dict(parent_pid=os.getpid(),pid=p.pid,argv=[cfg['python'],cfg['entry']],source_commit=cfg['source_commit'])))
reason=None;stopped=None
while True:
 got,status,u=os.wait4(p.pid,os.WNOHANG)
 if got:break
 try:check_space(cfg['window_root'])
 except RuntimeError:reason='WINDOW_SPACE_GUARD'
 if reason:
  if stopped is None:p.terminate();stopped=time.monotonic()
  elif time.monotonic()-stopped>5:p.kill()
 time.sleep(2)
p.returncode=os.waitstatus_to_exitcode(status);log.close();own=resource.getrusage(resource.RUSAGE_SELF);r=dict(worker_exit=p.returncode,stop_reason=reason,wall_seconds=time.monotonic()-start,worker_CPU_seconds=u.ru_utime+u.ru_stime,coordinator_CPU_seconds=own.ru_utime+own.ru_stime,worker_peak_RSS_bytes=u.ru_maxrss*1024,window_bytes=used(cfg['window_root']),source_commit=cfg['source_commit'])
(out/'COORDINATOR_EXIT.json').write_text(json.dumps(r,indent=2))
if not (out/'EXIT.json').exists():(out/'EXIT.json').write_text(json.dumps(dict(exit_code=p.returncode,decision='UNRESOLVED_PROCESS_OR_SPACE')))
