"""Supervise an existing worker without restart after an authorized wall-time amendment.
The legacy controller retains wait4 ownership and the exclusive lock. Its static
old timer is suspended only after this independent replacement is ready.
"""
import os,json,time,signal,threading,traceback
from pathlib import Path
from common import write,allocated
from memory_guard import group_rss
r=Path(os.environ['RUN_ROOT']);cfg=json.loads((r/'config.private.json').read_text());a=json.loads((r/'WALL24_AMENDMENT.private.json').read_text());clock=json.loads((r/'FIRST_ACTUAL.private.json').read_text());old=a['legacy_controller_pid'];worker=a['worker_pid'];reason=None;term=None;peak=0;peakrss=0
assert clock['start_unix']==a['original_start_unix'] and clock['deadline_unix']==a['new_deadline_unix']
def proc(pid):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 v=p.read_text().rsplit(')',1)[1].split();return {'state':v[0],'ppid':int(v[1]),'pgid':int(v[2]),'start_ticks':int(v[19])}
initial=proc(worker);assert initial and initial['ppid']==old and initial['pgid']==worker and initial['state']!='Z';assert proc(old)
def kill_group(sig):
 try:os.killpg(worker,sig)
 except ProcessLookupError:pass

def watchdog():
 global reason
 state=proc(worker)
 if state and state['state']!='Z' and state['start_ticks']==initial['start_ticks']:
  reason='AMENDED_WALL_WATCHDOG';kill_group(signal.SIGKILL)
timer=threading.Timer(max(1,clock['deadline_unix']-time.time()-60),watchdog);timer.daemon=True;timer.start()
write(r/'AMENDED_GUARD_READY.private.json',{'pid':os.getpid(),'worker_pid':worker,'legacy_controller_pid':old,'worker_start_ticks':initial['start_ticks'],'deadline_unix':clock['deadline_unix'],'original_start_unix':clock['start_unix']})
last=0
try:
 while True:
  state=proc(worker)
  if not state or state['state']=='Z':break
  assert state['start_ticks']==initial['start_ticks'] and state['pgid']==worker,'WORKER_IDENTITY_CHANGED'
  rss,_=group_rss(worker,extra_pid=old);own=proc(os.getpid());rss+=int(next(v for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('VmRSS:')).split()[1])*1024;peakrss=max(peakrss,rss)
  if time.time()-last>=10:
   disk=allocated(cfg['budget_root'])+allocated(cfg['swap_cache_root']);peak=max(peak,disk);last=time.time();write(r/'GUARD.private.json',{'amended_guard_pid':os.getpid(),'legacy_controller_suspended':proc(old)['state']=='T','worker_pid':worker,'allocated_bytes':disk,'peak_allocated_bytes':peak,'aggregate_RSS_bytes':rss,'peak_aggregate_RSS_bytes':peakrss,'remaining_seconds':clock['deadline_unix']-time.time(),'deadline_unix':clock['deadline_unix'],'original_start_unix':clock['start_unix']})
  if reason is None:
   if time.time()>=clock['deadline_unix']-90:reason='AMENDED_WALL_LIMIT'
   elif rss>=cfg['aggregate_rss_budget_bytes']:reason='RSS_LIMIT'
   elif peak+512*2**20>=cfg['storage_budget_bytes']:reason='STORAGE_LIMIT'
   elif (r/'RELAY_FAILURE.private.txt').exists():reason='RELAY_FAILURE'
   if reason:kill_group(signal.SIGTERM);term=time.time()
  elif term is not None and time.time()-term>10:kill_group(signal.SIGKILL)
  time.sleep(2)
 # Resume the original parent solely to reap its exited child and recover wait4 costs.
 os.kill(old,signal.SIGCONT)
 end=time.time()+30
 while not (r/'CONTROL_EXIT.json').exists() and time.time()<end:time.sleep(.5)
 if (r/'CONTROL_EXIT.json').exists():
  legacy=json.loads((r/'CONTROL_EXIT.json').read_text());write(r/'LEGACY_CONTROL_EXIT_AFTER_AMENDMENT.private.json',legacy)
  rec=dict(legacy);rec.update(deadline_unix=clock['deadline_unix'],original_start_unix=clock['start_unix'],wall_amendment_applied=True,legacy_static_deadline_unix=legacy['deadline_unix'],legacy_reason=legacy.get('reason'),reason=reason,amended_guard_pid=os.getpid(),peak_aggregate_RSS_bytes=max(legacy.get('peak_aggregate_RSS_bytes',0),peakrss),peak_allocated_bytes=max(legacy.get('peak_allocated_bytes',0),peak));write(r/'AMENDED_CONTROL_EXIT.json',rec);write(r/'CONTROL_EXIT.json',rec)
 else:write(r/'AMENDED_GUARD_REAP_FAILURE.private.json',{'worker_state':proc(worker),'reason':reason,'legacy_controller_state':proc(old)})
except BaseException:
 (r/'AMENDED_GUARD_FAILURE.private.txt').write_text(traceback.format_exc());kill_group(signal.SIGTERM);time.sleep(2);kill_group(signal.SIGKILL)
 try:os.kill(old,signal.SIGCONT)
 except ProcessLookupError:pass
 raise
finally:timer.cancel()
