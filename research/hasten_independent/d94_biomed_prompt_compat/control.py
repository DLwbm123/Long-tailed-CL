"""Single original-clock child/group guard; no science deletion or automatic resume."""
import os,json,time,signal,subprocess,resource,threading,fcntl,traceback
from pathlib import Path
from common import write,allocated
from memory_guard import group_rss
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);lock=Path(cfg['shared_memory_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();p=None;usage=None;code=1;reason=None;peak=allocated(r);peakrss=0;term=0;attempt=1+max([int(x.stem.rsplit('_',1)[1]) for x in r.glob('CONTROL_EXIT_*.json')]+[0])
def timeout():
 global reason
 if p is not None and p.returncode is None:
  reason='ORIGINAL_WALL_LIMIT'
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
timer=threading.Timer(max(1,cfg['deadline']-time.time()-60),timeout);timer.daemon=True;timer.start()
def limits():
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]));resource.setrlimit(resource.RLIMIT_CPU,(400000,400000));resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3));os.setsid()
try:
 assert time.time()<cfg['deadline']-180
 env=os.environ.copy();env.update(cfg['environment'])
 with (r/f'worker_{attempt}.private.log').open('w') as f:p=subprocess.Popen([cfg['python'],cfg['entry']],env=env,cwd=cfg['runtime'],preexec_fn=limits,stdout=f,stderr=subprocess.STDOUT)
 write(r/'LAUNCH.private.json',{'attempt':attempt,'controller_pid':os.getpid(),'worker_pid':p.pid});last=0
 while True:
  pid,status,u=os.wait4(p.pid,os.WNOHANG)
  if pid:code=os.waitstatus_to_exitcode(status);usage=u;p.returncode=code;break
  rss,_=group_rss(p.pid,extra_pid=os.getpid());peakrss=max(peakrss,rss)
  if time.time()-last>10:
   peak=max(peak,allocated(r));last=time.time();write(r/'GUARD.private.json',{'remaining_seconds':cfg['deadline']-time.time(),'peak_group_RSS_bytes':peakrss,'group_RSS_bytes':rss,'peak_allocated_bytes':peak})
  if reason is None:
   if time.time()>cfg['deadline']-90:reason='ORIGINAL_WALL_LIMIT'
   elif peak+512*2**20>=cfg['storage_budget_bytes']:reason='STORAGE_LIMIT'
   elif rss>=cfg['aggregate_rss_budget_bytes']:reason='RAM_LIMIT'
   if reason:os.killpg(p.pid,signal.SIGTERM);term=time.time()
  elif time.time()-term>10:os.killpg(p.pid,signal.SIGKILL)
  time.sleep(2)
except BaseException:
 reason=reason or 'CONTROL_ERROR';(r/f'control_{attempt}.private.txt').write_text(traceback.format_exc())
 if p is not None and p.returncode is None:
  try:os.killpg(p.pid,signal.SIGKILL);_,status,usage=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(status)
  except ProcessLookupError:pass
finally:
 timer.cancel();rec={'attempt':attempt,'exit_code':code,'reason':reason,'attempt_wall_seconds':time.time()-start,'total_elapsed_seconds':time.time()-cfg['start'],'worker_CPU_seconds':None if usage is None else usage.ru_utime+usage.ru_stime,'worker_peak_RSS_bytes':None if usage is None else usage.ru_maxrss*1024,'peak_group_RSS_bytes':peakrss,'allocated_bytes':allocated(r),'peak_allocated_bytes':max(peak,allocated(r)),'deadline':cfg['deadline']};write(r/f'CONTROL_EXIT_{attempt}.json',rec);write(r/'CONTROL_EXIT.json',rec)
