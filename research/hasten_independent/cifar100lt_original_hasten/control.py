"""Own-process guard, independent of SSH, with the original preparation clock."""
import os,json,time,signal,subprocess,resource,threading
from pathlib import Path
r=Path('/tmp/q1101');clock=json.loads((r/'FIRST_ACTUAL.private.json').read_text());deadline=clock['deadline_unix'];cap=4_000_000_000
def write(name,x):
    p=r/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def used():
    return sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in r.rglob('*') if p.is_file() and not p.is_symlink())
def limits():
    os.setsid();os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]));resource.setrlimit(resource.RLIMIT_CPU,(200000,200000));resource.setrlimit(resource.RLIMIT_FSIZE,(2**30,2**30))
assert used()+512*2**20<cap and time.time()<deadline-120
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
f=(r/'worker.private.log').open('a');p=subprocess.Popen(['/tmp/q107v/bin/python','/tmp/q1101/worker.py'],env=env,cwd=r,preexec_fn=limits,stdout=f,stderr=subprocess.STDOUT)
write('LAUNCH.private.json',dict(controller_pid=os.getpid(),worker_pid=p.pid,gpu=1,deadline_unix=deadline))
def hard_stop():
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
timer=threading.Timer(max(1,deadline-time.time()),hard_stop);timer.daemon=True;timer.start()
peak=0;reason=None;term=None
while True:
    pid,status,usage=os.wait4(p.pid,os.WNOHANG)
    if pid:
        code=os.waitstatus_to_exitcode(status);p.returncode=code;break
    n=used();peak=max(peak,n)
    try:rss=int(next(x for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
    except (OSError,StopIteration):rss=0
    write('GUARD.private.json',dict(time=time.time(),allocated_bytes=n,peak_allocated_bytes=peak,remaining_seconds=deadline-time.time(),rss_bytes=rss))
    if reason is None:
        if n+512*2**20>=cap:reason='STORAGE_PRESTOP'
        elif rss>8*2**30:reason='RSS_LIMIT'
        elif time.time()>deadline-90:reason='ORIGINAL_WALL_LIMIT'
        if reason:os.killpg(p.pid,signal.SIGTERM);term=time.time()
    elif time.time()-term>10:hard_stop()
    time.sleep(5)
timer.cancel();write('CONTROL_EXIT.json',dict(exit_code=code,reason=reason,elapsed_seconds=time.time()-clock['start_unix'],CPU_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,peak_allocated_bytes=max(peak,used()),allocated_bytes=used()))
