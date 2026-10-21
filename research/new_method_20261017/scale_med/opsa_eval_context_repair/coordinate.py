"""Independent original clock/RSS/storage guard, neutral CPU/GPU process names."""
import os,json,time,resource,signal,subprocess,traceback
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);start=time.time();deadline=cfg['original_start_unix']+21590
assert start<deadline, 'ORIGINAL_CLOCK_EXPIRED'
(out/'FIRST_ACTUAL.private.json').write_text(json.dumps(dict(start_unix=start,deadline_unix=deadline)))
def used(r):return sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in Path(r).rglob('*') if p.is_file() and not p.is_symlink())
def limits():
 resource.setrlimit(resource.RLIMIT_CPU,(24000,24000));resource.setrlimit(resource.RLIMIT_FSIZE,(64*2**20,64*2**20));os.setsid()
worker=None;reason=None;code=1;usage=None;gpu_seen=False
try:
 with (out/'worker.private.log').open('w') as log:worker=subprocess.Popen([cfg['python'],cfg['entry']],stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
 (out/'LAUNCH.private.json').write_text(json.dumps(dict(parent_pid=os.getpid(),worker_pid=worker.pid)))
 while True:
  pid,status,u=os.wait4(worker.pid,os.WNOHANG)
  if pid:code=os.waitstatus_to_exitcode(status);usage=u;worker.returncode=code;break
  try:
   row=[v for v in Path(f'/proc/{worker.pid}/status').read_text().splitlines() if v.startswith('VmRSS:')];rss=int(row[0].split()[1])*1024 if row else 0
  except FileNotFoundError:continue
  if not gpu_seen:
   q=subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True)
   hits=[v for v in q.stdout.splitlines() if v.split(',')[0].strip()==str(worker.pid)]
   if hits:
    gpu_seen=True;(out/'GPU_ARGV.private.json').write_text(json.dumps(dict(rows=hits)))
    if any('wangbomin' in v or 'LongTailed' in v or 'opsa' in v.lower() for v in hits):reason='GPU_ARGV_NOT_NEUTRAL';os.killpg(worker.pid,signal.SIGTERM);term=time.time()
  if reason is None:
   if time.time()>=deadline:reason='WALL_LIMIT'
   elif rss>8*2**30:reason='RSS_LIMIT'
   elif used(out)+cfg['prior_chain_bytes']+32*2**20>=512*2**20:reason='ROUND_STORAGE_LIMIT'
   elif used(cfg['window_root'])+1001040384>=17000000000:reason='WINDOW_STORAGE_LIMIT'
   if reason:os.killpg(worker.pid,signal.SIGTERM);term=time.time()
  elif time.time()-term>5:os.killpg(worker.pid,signal.SIGKILL)
  time.sleep(2)
except BaseException:
 reason=reason or 'COORDINATOR_ERROR';(out/'COORDINATOR_FAILURE.private.txt').write_text(traceback.format_exc())
 if worker is not None and worker.returncode is None:
  try:os.killpg(worker.pid,signal.SIGKILL);_,status,usage=os.wait4(worker.pid,0);code=os.waitstatus_to_exitcode(status)
  except ProcessLookupError:pass
finally:
 u=resource.getrusage(resource.RUSAGE_SELF)
 rec=dict(exit_code=code,stop_reason=reason,wall_seconds=time.time()-start,worker_CPU_seconds=None if usage is None else usage.ru_utime+usage.ru_stime,worker_peak_RSS_bytes=None if usage is None else usage.ru_maxrss*1024,coordinator_CPU_seconds=u.ru_utime+u.ru_stime,round_bytes=used(out),window_bytes=used(cfg['window_root']),GPU_active_name_captured=gpu_seen,completed_unix=time.time(),deadline_unix=deadline)
 (out/'COORDINATOR_EXIT.json').write_text(json.dumps(rec,indent=2));(out/'EXIT.json').write_text(json.dumps(dict(exit_code=code)))
 if code or reason:(out/'FAILURE.json').write_text(json.dumps(dict(exit_code=code,reason=reason or 'WORKER_FAILURE',no_retry=True)))
