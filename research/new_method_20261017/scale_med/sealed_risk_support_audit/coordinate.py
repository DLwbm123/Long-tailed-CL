"""Independent wall/RSS/storage guard; all child arguments stay neutral."""
import json
import os
import resource
import signal
import subprocess
import time
import traceback
from pathlib import Path

cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
out = Path(cfg['output'])
start = time.time()
deadline = start+600
Path(out/'FIRST_ACTUAL.private.json').write_text(json.dumps(dict(start_unix=start, deadline_unix=deadline)))


def used(root):
    return sum(max(p.stat().st_size, p.stat().st_blocks*512) for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (16*2**20, 16*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    os.setsid()


reason = None
worker = None
exitcode = 1
waitusage = None
try:
    with (out/'worker.private.log').open('w') as log:
        worker = subprocess.Popen([cfg['python'], cfg['entry']], stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
        (out/'LAUNCH.private.json').write_text(json.dumps(dict(parent_pid=os.getpid(), worker_pid=worker.pid)))
        while True:
            pid, status, usage = os.wait4(worker.pid, os.WNOHANG)
            if pid:
                exitcode = os.waitstatus_to_exitcode(status)
                worker.returncode = exitcode
                waitusage = usage
                break
            if reason is None:
                try:
                    lines = [s for s in Path('/proc/'+str(worker.pid)+'/status').read_text().splitlines() if s.startswith('VmRSS:')]
                    rss = int(lines[0].split()[1])*1024 if lines else 0
                except FileNotFoundError:
                    continue
                if time.time() >= deadline:
                    reason = 'WALL_LIMIT'
                elif rss > 4*2**30:
                    reason = 'RSS_LIMIT'
                elif used(out)+16*2**20 >= 128*2**20:
                    reason = 'ROUND_STORAGE_LIMIT'
                if used(cfg['window_root'])+1001040384 >= 17000000000:
                    reason = 'WINDOW_STORAGE_LIMIT'
                if reason:
                    os.killpg(worker.pid, signal.SIGTERM)
                    terminate_time = time.time()
            elif time.time()-terminate_time > 5:
                os.killpg(worker.pid, signal.SIGKILL)
            time.sleep(1)
except BaseException:
    reason = reason or 'COORDINATOR_ERROR'
    (out/'FAILURE.private.txt').write_text(traceback.format_exc())
    if worker is not None and worker.returncode is None:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
            _, status, waitusage = os.wait4(worker.pid, 0)
            exitcode = os.waitstatus_to_exitcode(status)
        except ProcessLookupError:
            pass
finally:
    u = resource.getrusage(resource.RUSAGE_SELF)
    record = dict(exit_code=exitcode, stop_reason=reason, wall_seconds=time.time()-start,
                  worker_CPU_seconds=None if waitusage is None else waitusage.ru_utime+waitusage.ru_stime,
                  worker_peak_RSS_bytes=None if waitusage is None else waitusage.ru_maxrss*1024,
                  coordinator_CPU_seconds=u.ru_utime+u.ru_stime, round_bytes=used(out), window_bytes=used(cfg['window_root']),
                  completed_unix=time.time(), deadline_unix=deadline)
    (out/'COORDINATOR_EXIT.json').write_text(json.dumps(record, indent=2)+'\n')
    (out/'EXIT.json').write_text(json.dumps(dict(exit_code=exitcode))+'\n')
    if exitcode or reason:
        (out/'FAILURE.json').write_text(json.dumps(dict(reason=reason or 'WORKER_FAILURE', exit_code=exitcode))+'\n')
