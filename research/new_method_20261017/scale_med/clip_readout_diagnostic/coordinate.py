import os,json,time,signal,subprocess,resource
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);start=time.time();reason=None;usage=None
def used():return sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in out.rglob('*') if p.is_file())
def limits():
 os.setsid();resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20))
with (out/'worker.private.log').open('w') as f:p=subprocess.Popen([cfg['python'],cfg['entry']],stdout=f,stderr=subprocess.STDOUT,preexec_fn=limits)
(out/'LAUNCH.private.json').write_text(json.dumps(dict(parent_pid=os.getpid(),worker_pid=p.pid,start_unix=start)))
while True:
 pid,status,u=os.wait4(p.pid,os.WNOHANG)
 if pid:usage=u;code=os.waitstatus_to_exitcode(status);break
 try:rss=next(int(x.split()[1])*1024 for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith('VmRSS:'))
 except (FileNotFoundError,StopIteration):rss=0
 if reason is None:
  if time.time()-start>590:reason='TIME_LIMIT'
  elif rss>2*2**30:reason='RSS_LIMIT'
  elif used()+8*2**20>=16*2**20:reason='STORAGE_LIMIT'
  if reason:os.killpg(p.pid,signal.SIGTERM);term=time.time()
 elif time.time()-term>5:os.killpg(p.pid,signal.SIGKILL)
 time.sleep(1)
u=resource.getrusage(resource.RUSAGE_SELF)
(out/'EXIT.json').write_text(json.dumps(dict(exit_code=code,stop_reason=reason,wall_seconds=time.time()-start,worker_CPU_seconds=usage.ru_utime+usage.ru_stime,worker_peak_RSS_bytes=usage.ru_maxrss*1024,coordinator_CPU_seconds=u.ru_utime+u.ru_stime,new_bytes=used()),indent=2))
