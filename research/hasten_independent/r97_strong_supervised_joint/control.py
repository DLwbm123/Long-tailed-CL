"""Single original-clock child/group guard; no science deletion or automatic resume."""
import os,json,time,signal,subprocess,resource,threading,fcntl,traceback
from pathlib import Path
from common import write,allocated
from memory_guard import group_rss
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);lock=Path(cfg['shared_memory_lock']).open('a');
while True:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:
  write(r/'STATUS.json',{'phase':'QUEUED_SHARED_LOCK','remaining_seconds':cfg['deadline']-time.time(),'formal_optimizer':0})
  if time.time()>cfg['deadline']-7200:
   write(r/'STATUS.json',{'phase':'NOT_ADMITTED_QUEUE','formal_optimizer':0});write(r/'CONTROL_EXIT_1.json',{'attempt':1,'exit_code':0,'reason':'QUEUE_BUDGET_EXHAUSTED','total_elapsed_seconds':time.time()-cfg['start'],'worker_CPU_seconds':0,'queue_controller_CPU_seconds':time.process_time()});raise SystemExit(0)
  time.sleep(20)
mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
free=int(subprocess.check_output(['nvidia-smi','-i',str(cfg['physical_gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
if not (mem>=64*2**30 and free>=14*2**30):
 write(r/'STATUS.json',{'phase':'NOT_ADMITTED_RESOURCE','formal_optimizer':0});write(r/'CONTROL_EXIT_1.json',{'attempt':1,'exit_code':0,'reason':'RESOURCE_ADMISSION','MemAvailable':mem,'GPUfree':free,'total_elapsed_seconds':time.time()-cfg['start'],'worker_CPU_seconds':0});raise SystemExit(0)
assert __import__('shutil').disk_usage(r).free>=cfg['storage_budget_bytes']+512*2**20
write(r/'RESOURCE_ADMISSION.private.json',{'MemAvailable':mem,'GPUfree':free,'original_deadline':cfg['deadline']})
start=time.time();p=None;usage=None;code=1;reason=None;peak=allocated(r);peakrss=0;term=0;attempt=1+max([int(x.stem.rsplit('_',1)[1]) for x in r.glob('CONTROL_EXIT_*.json')]+[0])
def timeout():
 global reason
 if p is not None and p.returncode is None:
  reason='ORIGINAL_WALL_LIMIT'
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
timer=threading.Timer(max(1,cfg['deadline']-time.time()-60),timeout);timer.daemon=True;timer.start()
def limits():
 os.sched_setaffinity(0,set(cfg.get('cpu_affinity',sorted(os.sched_getaffinity(0))[-4:])));resource.setrlimit(resource.RLIMIT_CPU,(400000,400000));resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3));os.setsid()
try:
 assert time.time()<cfg['deadline']-180
 env=os.environ.copy();env.update(cfg['environment'])
 with (r/f'worker_{attempt}.private.log').open('w') as f:p=subprocess.Popen([cfg['python'],cfg['entry']],env=env,cwd=cfg['runtime'],preexec_fn=limits,stdout=f,stderr=subprocess.STDOUT)
 write(r/'LAUNCH.private.json',{'attempt':attempt,'controller_pid':os.getpid(),'worker_pid':p.pid});last=0
 while True:
  pid,status,u=os.wait4(p.pid,os.WNOHANG)
  if pid:code=os.waitstatus_to_exitcode(status);usage=u;p.returncode=code;break
  rss,_=group_rss(p.pid,extra_pid=os.getpid());peakrss=max(peakrss,rss);peer_rss=0
  for peer in cfg.get('peer_groups',[]):
   try:
    fields=(Path('/proc')/str(peer['pgid'])/'stat').read_text().rsplit(')',1)[1].split()
    if int(fields[19])==peer['start_ticks']:peer_rss+=group_rss(peer['pgid'],extra_pid=peer['controller'])[0]
   except (OSError,ValueError,IndexError):pass
  combined_rss=rss+peer_rss
  if time.time()-last>10:
   peak=max(peak,allocated(r));last=time.time();write(r/'GUARD.private.json',{'remaining_seconds':cfg['deadline']-time.time(),'peak_group_RSS_bytes':peakrss,'group_RSS_bytes':rss,'peer_RSS_bytes':peer_rss,'combined_RSS_bytes':combined_rss,'peak_allocated_bytes':peak})
  if reason is None:
   if time.time()>cfg['deadline']-90:reason='ORIGINAL_WALL_LIMIT'
   elif peak+512*2**20>=cfg['storage_budget_bytes']:reason='STORAGE_LIMIT'
   elif rss>=cfg['aggregate_rss_budget_bytes'] or combined_rss>=29_000_000_000:reason='RAM_LIMIT'
   if reason:os.killpg(p.pid,signal.SIGTERM);term=time.time()
  elif time.time()-term>10:os.killpg(p.pid,signal.SIGKILL)
  time.sleep(2)
except BaseException:
 reason=reason or 'CONTROL_ERROR';(r/f'control_{attempt}.private.txt').write_text(traceback.format_exc())
 if p is not None and p.returncode is None:
  try:os.killpg(p.pid,signal.SIGKILL);_,status,usage=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(status)
  except ProcessLookupError:pass
finally:
 timer.cancel();rec={'attempt':attempt,'exit_code':code,'reason':reason,'attempt_wall_seconds':time.time()-start,'total_elapsed_seconds':time.time()-cfg['start'],'queue_controller_CPU_seconds':time.process_time(),'worker_CPU_seconds':None if usage is None else usage.ru_utime+usage.ru_stime,'worker_peak_RSS_bytes':None if usage is None else usage.ru_maxrss*1024,'peak_group_RSS_bytes':peakrss,'allocated_bytes':allocated(r),'peak_allocated_bytes':max(peak,allocated(r)),'deadline':cfg['deadline']};write(r/f'CONTROL_EXIT_{attempt}.json',rec);write(r/'CONTROL_EXIT.json',rec)
