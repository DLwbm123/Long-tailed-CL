"""Original-clock process-group guard; source jobs and target share one clock."""
import os,json,time,signal,subprocess,resource,traceback,threading
from pathlib import Path
from common import allocated,write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True);clock=Path(cfg['clock']);assert clock.exists();origin=json.loads(clock.read_text());start=time.time();p=None;usage=None;code=1;reason=None;peak=allocated(cfg['budget_root']);attempt=1+len(list(root.glob('CONTROL_EXIT_*.json')))
def watchdog():
 global reason
 if p is not None and p.returncode is None:
  reason='ORIGINAL_WALL_WATCHDOG'
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
timer=threading.Timer(max(1,origin['deadline_unix']-time.time()-60),watchdog);timer.daemon=True;timer.start()
def limits():
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]))
 resource.setrlimit(resource.RLIMIT_CPU,(400000,400000));resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3));os.setsid()
try:
 assert time.time()<origin['deadline_unix']-90
 env=os.environ.copy();env.update(cfg.get('environment',{}))
 with (root/f'worker_{attempt}.private.log').open('w') as f:p=subprocess.Popen([cfg['python'],cfg['entry']],env=env,cwd=cfg['runtime'],preexec_fn=limits,stdout=f,stderr=subprocess.STDOUT)
 write(root/'LAUNCH.private.json',dict(controller_pid=os.getpid(),worker_pid=p.pid,attempt=attempt));last=0
 while True:
  pid,st,u=os.wait4(p.pid,os.WNOHANG)
  if pid:code=os.waitstatus_to_exitcode(st);usage=u;p.returncode=code;break
  if time.time()-last>10:
   used=allocated(cfg['budget_root']);peak=max(peak,used);last=time.time();write(root/'GUARD.private.json',dict(allocated_bytes=used,peak_allocated_bytes=peak,remaining_seconds=origin['deadline_unix']-time.time()))
  try:rss=int(next(v for v in Path(f'/proc/{p.pid}/status').read_text().splitlines() if v.startswith('VmRSS:')).split()[1])*1024
  except (FileNotFoundError,StopIteration):rss=0
  if reason is None:
   if time.time()>origin['deadline_unix']-90:reason='ORIGINAL_WALL_LIMIT'
   elif peak+512*2**20>=cfg['storage_budget_bytes']:reason='STORAGE_LIMIT'
   elif rss>cfg['rss_budget_bytes']:reason='RSS_LIMIT'
   elif (Path(cfg['budget_root'])/'RELAY_FAILURE.private.txt').exists():reason='RELAY_FAILURE'
   if reason:os.killpg(p.pid,signal.SIGTERM);term=time.time()
  elif time.time()-term>10:os.killpg(p.pid,signal.SIGKILL)
  time.sleep(2)
except BaseException:
 reason=reason or 'CONTROL_ERROR';(root/f'control_{attempt}.private.txt').write_text(traceback.format_exc())
 if p is not None and p.returncode is None:
  try:os.killpg(p.pid,signal.SIGKILL);_,st,usage=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(st)
  except ProcessLookupError:pass
finally:
 timer.cancel()
 rec=dict(attempt=attempt,exit_code=code,reason=reason,attempt_wall_seconds=time.time()-start,total_elapsed_seconds=time.time()-origin['start_unix'],worker_CPU_seconds=None if usage is None else usage.ru_utime+usage.ru_stime,peak_RSS_bytes=None if usage is None else usage.ru_maxrss*1024,peak_allocated_bytes=max(peak,allocated(cfg['budget_root'])),allocated_bytes=allocated(cfg['budget_root']),deadline_unix=origin['deadline_unix']);write(root/f'CONTROL_EXIT_{attempt}.json',rec);write(root/'CONTROL_EXIT.json',rec)
