"""Independent clock, process group and allocated-storage guard; retries share start."""
import os,json,time,signal,subprocess,resource,traceback
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);first=out/'FIRST_ACTUAL.private.json'
if first.exists():origin=json.loads(first.read_text())
else:
 origin=dict(start_unix=time.time());origin['deadline_unix']=min(origin['start_unix']+5400,1792728054.0);first.write_text(json.dumps(origin))
start=time.time();deadline=origin['deadline_unix'];worker=None;code=1;reason=None;usage=None
attempt=1+len(list(out.glob('CONTROL_EXIT_*.json')))
def used():return sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in out.rglob('*') if p.is_file() and not p.is_symlink())
def limits():
 resource.setrlimit(resource.RLIMIT_CPU,(72000,72000));resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3));os.setsid()
try:
 assert time.time()<deadline,'ORIGINAL_DEADLINE_EXHAUSTED'
 with (out/f'worker_{attempt}.private.log').open('w') as log:
  worker=subprocess.Popen([cfg['python'],cfg['entry']],stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits,cwd=out/'runtime')
 (out/'LAUNCH.private.json').write_text(json.dumps(dict(parent_pid=os.getpid(),worker_pid=worker.pid,attempt=attempt)))
 last_scan=0;space=used()
 while True:
  pid,st,u=os.wait4(worker.pid,os.WNOHANG)
  if pid:code=os.waitstatus_to_exitcode(st);usage=u;worker.returncode=code;break
  if time.time()-last_scan>10:space=used();last_scan=time.time()
  try:
   fields=Path(f'/proc/{worker.pid}/status').read_text().splitlines();rss=int(next(s for s in fields if s.startswith('VmRSS:')).split()[1])*1024
  except (FileNotFoundError,StopIteration):rss=0
  if reason is None:
   if time.time()>=deadline-30:reason='ORIGINAL_WALL_LIMIT'
   elif space+256*2**20>=2500000000:reason='STORAGE_LIMIT'
   elif rss>12*2**30:reason='RSS_LIMIT'
   if reason:os.killpg(worker.pid,signal.SIGTERM);term=time.time()
  elif time.time()-term>5:os.killpg(worker.pid,signal.SIGKILL)
  time.sleep(2)
except BaseException:
 reason=reason or 'CONTROL_ERROR';(out/f'control_{attempt}.private.txt').write_text(traceback.format_exc())
 if worker is not None and worker.returncode is None:
  try:os.killpg(worker.pid,signal.SIGKILL);_,st,usage=os.wait4(worker.pid,0);code=os.waitstatus_to_exitcode(st)
  except ProcessLookupError:pass
finally:
 rec=dict(attempt=attempt,exit_code=code,reason=reason,attempt_wall_seconds=time.time()-start,total_elapsed_seconds=time.time()-origin['start_unix'],worker_CPU_seconds=None if usage is None else usage.ru_utime+usage.ru_stime,peak_RSS_bytes=None if usage is None else usage.ru_maxrss*1024,allocated_bytes=used(),deadline_unix=deadline)
 (out/f'CONTROL_EXIT_{attempt}.json').write_text(json.dumps(rec,indent=2));(out/'CONTROL_EXIT.json').write_text(json.dumps(rec,indent=2))
