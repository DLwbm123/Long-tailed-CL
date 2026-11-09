"""Independent hard clock/storage/RSS guard; worker survives the client session."""
import fcntl
import os
import resource
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .core import write


def allocated(root):
    total = 0
    for p in Path(root).rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():
                s = p.stat()
                total += max(s.st_size, s.st_blocks * 512)
        except FileNotFoundError:
            pass
    return total


def rss_tree(root):
    sizes, parents = {}, {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            text = (p / 'status').read_text()
            fields = dict(s.split(':', 1) for s in text.splitlines() if ':' in s)
            parents[int(p.name)] = int(fields['PPid'])
            sizes[int(p.name)] = int(fields.get('VmRSS', '0 kB').split()[0]) * 1024
        except (OSError, ValueError):
            pass
    members = {root}
    while True:
        new = members | {p for p, parent in parents.items() if parent in members}
        if new == members:
            break
        members = new
    return sum(sizes.get(p, 0) for p in members)


def run(cfg):
    out = Path(cfg['output'])
    lock = (out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(cfg['cpu_affinity']))
    p, reason, term = None, None, None
    peak_disk, peak_rss, exited = 0, 0, None
    try:
        assert time.time() < cfg['deadline'] - 3600
        assert shutil.disk_usage(out).free >= 5_000_000_000
        free = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)) * 2**20
        assert free >= 60 * 2**30, 'GPU1_CURRENT_FREE_INSUFFICIENT'

        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 2**30, 4 * 2**30))
            resource.setrlimit(resource.RLIMIT_CPU, (160000, 160000))

        env = {**os.environ, 'Q_MODULE': 'worker'}
        with (out / 'worker.private.log').open('w') as log:
            p = subprocess.Popen([cfg['python'], cfg['entry']], stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env=env, start_new_session=True, preexec_fn=limits)
        write(out / 'LAUNCH.private.json', {'controller': os.getpid(), 'worker': p.pid,
                                           'original_start': cfg['start'], 'deadline': cfg['deadline'], 'physical_gpu': 1})
        while True:
            pid, status, usage = os.wait4(p.pid, os.WNOHANG)
            if pid:
                p.returncode = os.waitstatus_to_exitcode(status)
                exited = {'exit_code': p.returncode, 'CPU_seconds': usage.ru_utime + usage.ru_stime}
                if p.returncode:
                    reason = reason or 'WORKER_FAILURE'
                break
            disk = allocated(out)
            rss = rss_tree(os.getpid())
            peak_disk, peak_rss = max(peak_disk, disk), max(peak_rss, rss)
            if time.time() >= cfg['deadline'] - 120:
                reason = reason or 'ORIGINAL_CLOCK'
            # 1GB headroom covers bounded writes and local-source/safe delivery copies.
            if disk + cfg.get('other_window_allocated_bytes', 0) >= 9_000_000_000:
                reason = reason or 'WINDOW_DISK_PRESTOP'
            if shutil.disk_usage(out).free < 1_000_000_000:
                reason = reason or 'FILESYSTEM_HEALTH'
            if rss >= 48_000_000_000:
                reason = reason or 'GROUP_RSS_LIMIT'
            write(out / 'GUARD.private.json', {'epoch': time.time(), 'reason': reason,
                                              'remaining_seconds': cfg['deadline'] - time.time(),
                                              'allocated_bytes': disk, 'peak_allocated_bytes': peak_disk,
                                              'group_RSS_bytes': rss, 'peak_group_RSS_bytes': peak_rss})
            if reason:
                sig = signal.SIGTERM if term is None else signal.SIGKILL if time.time() - term > 15 else None
                if sig:
                    try:
                        os.killpg(p.pid, sig)
                    except ProcessLookupError:
                        pass
                    term = term or time.time()
            time.sleep(2)
    except BaseException:
        import traceback
        reason = reason or 'CONTROL_FAILURE'
        (out / 'CONTROL_FAILURE.private.txt').write_text(traceback.format_exc())
    finally:
        if p is not None and exited is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
                _, status, usage = os.wait4(p.pid, 0)
                exited = {'exit_code': os.waitstatus_to_exitcode(status), 'CPU_seconds': usage.ru_utime + usage.ru_stime}
            except ProcessLookupError:
                pass
        write(out / 'CONTROL_EXIT.json', {'exit_code': int(reason is not None), 'reason': reason, 'worker': exited,
                                         'original_elapsed_seconds': time.time() - cfg['start'],
                                         'peak_disk_bytes': peak_disk, 'peak_group_RSS_bytes': peak_rss})
