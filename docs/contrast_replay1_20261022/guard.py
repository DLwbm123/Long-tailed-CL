"""Independent admission, original-clock deadline and whole-process-group guard."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

cfg_path = Path(os.environ['RUN_CONFIG']).resolve()
cfg = json.loads(cfg_path.read_text())
root = Path(cfg['output']).parent
entry = Path(cfg['neutral_entry'])


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def identity(pid):
    return dict(pid=pid, start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])


def usage():
    total=0
    for p in list(root.rglob('*'))+list(entry.rglob('*')):
        try:
            if p.is_file():
                st=p.stat();total+=max(st.st_size,st.st_blocks*512)
        except FileNotFoundError:
            pass
    return total


def free_gpu():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    return next(int(r.split(',')[1])*1024**2 for r in rows if int(r.split(',')[0]) == cfg['GPU_index'])


assert not (root / 'guardian.private.json').exists(), 'Do not relaunch this experiment'
cfg.setdefault('original_start_epoch', time.time())
cfg.setdefault('hard_deadline_epoch', cfg['original_start_epoch'] + cfg['max_wall_seconds'])
assert cfg['hard_deadline_epoch'] == cfg['original_start_epoch'] + cfg['max_wall_seconds']
write(cfg_path, cfg)
write(root / 'guardian.private.json', dict(**identity(os.getpid()),
      original_start_epoch=cfg['original_start_epoch'], hard_deadline_epoch=cfg['hard_deadline_epoch'],
      storage_initial_bytes=usage(), budget_bytes=cfg['output_budget_bytes']))
assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024>=cfg['required_host_free_bytes']
assert free_gpu() >= cfg['required_GPU_free_bytes'], 'GPU admission failed; no worker launched'
total_peak=0
total_rss=0
for name in ['run', 'verify']:
    if time.time() >= cfg['hard_deadline_epoch']-60:
        break
    with (root / (name+'.stdout.private.log')).open('xb') as log:
        p = subprocess.Popen([sys.executable, str(Path(__file__).with_name(name+'.py'))],
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              stdin=subprocess.DEVNULL)
        ident = identity(p.pid)
        write(root / (name+'_launch.private.json'), dict(**ident, started=time.time()))
        peak, rss, reason = 0, 0, 'process_exit'
        while p.poll() is None:
            size = usage()
            peak = max(peak, size)
            total_peak=max(total_peak,size)
            group_rss=0
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():continue
                try:
                    stat=proc.joinpath('stat').read_text().split()
                    if int(stat[4])!=p.pid:continue
                    lines=proc.joinpath('status').read_text().splitlines()
                    group_rss+=next((int(z.split()[1])*1024 for z in lines if z.startswith('VmRSS:')),0)
                except (FileNotFoundError,ProcessLookupError,PermissionError):pass
            rss=max(rss,group_rss)
            total_rss=max(total_rss,rss)
            write(root / 'guardian_resources.private.json', dict(updated=time.time(), phase=name,
                  new_output_bytes=size, peak_output_bytes=total_peak, RSS_sampled_peak_bytes=total_rss,
                  phase_peak_output_bytes=peak,phase_RSS_sampled_peak_bytes=rss,
                  hard_deadline_epoch=cfg['hard_deadline_epoch']))
            if size >= cfg['storage_stop_threshold_bytes'] or rss >= cfg['RSS_budget_bytes'] or time.time() >= cfg['hard_deadline_epoch']-60:
                reason = 'storage_prestop' if size >= cfg['storage_stop_threshold_bytes'] else ('RSS_limit' if rss>=cfg['RSS_budget_bytes'] else 'original_deadline')
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=10)
                break
            time.sleep(5)
        rc = p.wait()
        # Retire any child retaining the worker's process group before guard exit.
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        write(root / (name+'_terminal.private.json'), dict(**ident, returncode=rc, reason=reason,
              finished=time.time(), elapsed_from_first_run=time.time()-cfg['original_start_epoch'],
              peak_output_bytes=total_peak,RSS_sampled_peak_bytes=total_rss))
        if rc:
            break

write(root/'guardian_terminal.private.json',dict(finished=time.time(),elapsed_from_first_run=time.time()-cfg['original_start_epoch'],peak_output_bytes=total_peak,RSS_sampled_peak_bytes=total_rss))
