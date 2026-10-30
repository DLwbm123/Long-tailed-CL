"""Original-clock process-group guard; source jobs and target share one clock."""
import os,json,time,signal,subprocess,resource,traceback,threading,fcntl
from pathlib import Path
from common import allocated,write
from memory_guard import group_rss
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());remote_budget_root=cfg['budget_root']
def used_storage():return allocated(remote_budget_root)+allocated(cfg['swap_cache_root'])
memory_lock=Path(cfg['shared_memory_lock']).open('a');fcntl.flock(memory_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True);clock=Path(cfg['clock']);assert clock.exists();origin=json.loads(clock.read_text());start=time.time();p=None;usage=None;code=1;reason=None;peak=used_storage();peak_group_rss=0;term=0.;attempt=1+max([int(p.stem.split('_')[-1]) for p in root.glob('CONTROL_EXIT_*.json')]+[0])
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
 write(root/'LAUNCH.private.json',dict(controller_pid=os.getpid(),worker_pid=p.pid,attempt=attempt));(root/f'launch_ps_{attempt}.private.txt').write_text(subprocess.check_output(['ps','-ww','-o','pid,ppid,stat,args','-p',str(os.getpid())+','+str(p.pid)],text=True));last=0
 while True:
  pid,st,u=os.wait4(p.pid,os.WNOHANG)
  if pid:code=os.waitstatus_to_exitcode(st);usage=u;p.returncode=code;break
  if time.time()-last>10:
   used=used_storage();peak=max(peak,used);last=time.time();write(root/'GUARD.private.json',dict(allocated_bytes=used,peak_allocated_bytes=peak,remaining_seconds=origin['deadline_unix']-time.time(),aggregate_RSS_bytes=group_rss(p.pid,extra_pid=os.getpid())[0],peak_aggregate_RSS_bytes=peak_group_rss))
  rss,memory_pids=group_rss(p.pid,extra_pid=os.getpid());peak_group_rss=max(peak_group_rss,rss)
  if reason is None:
   if time.time()>origin['deadline_unix']-90:reason='ORIGINAL_WALL_LIMIT'
   elif peak+512*2**20>=cfg['storage_budget_bytes'] or allocated(remote_budget_root)+512*2**20>=cfg['remote_output_budget_bytes']:reason='STORAGE_LIMIT'
   elif rss>=cfg['aggregate_rss_budget_bytes']:reason='RSS_LIMIT'
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
 # All scratch files are declared regenerable prefixes; no scientific artifact is stored here.
 for scratch_folder in [Path(cfg['swap_cache_root']),Path(cfg['output'])/'temporary_prefix_swap']:
  for scratch in scratch_folder.glob('prefix_*.npy'):scratch.unlink()
 rec=dict(attempt=attempt,exit_code=code,reason=reason,attempt_wall_seconds=time.time()-start,total_elapsed_seconds=time.time()-origin['start_unix'],worker_CPU_seconds=None if usage is None else usage.ru_utime+usage.ru_stime,peak_RSS_bytes=None if usage is None else usage.ru_maxrss*1024,peak_aggregate_RSS_bytes=peak_group_rss,peak_allocated_bytes=max(peak,used_storage()),allocated_bytes=used_storage(),deadline_unix=origin['deadline_unix']);write(root/f'CONTROL_EXIT_{attempt}.json',rec);write(root/'CONTROL_EXIT.json',rec)
