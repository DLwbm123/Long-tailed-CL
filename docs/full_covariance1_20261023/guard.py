"""Independent storage/RSS guard, additionally enclosed by an NFS-independent timeout."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent

def identity(pid):return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
def usage(locations):
    total=0
    for location in locations:
        for p in Path(location).rglob('*'):
            try:
                if p.is_file():
                    s=p.stat();total+=max(s.st_size,s.st_blocks*512)
            except FileNotFoundError:pass
    return total

assert not (root/'guardian.private.json').exists()
write(root/'guardian.private.json',dict(**identity(os.getpid()),original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch']))
peak=0;rsspeak=0
for phase in ['run','verify']:
    assert time.time()<cfg['hard_deadline_epoch']-60
    with (root/(phase+'.stdout.private.log')).open('xb') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name(phase+'.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=False,stdin=subprocess.DEVNULL)
        ident=identity(p.pid);write(root/(phase+'_launch.private.json'),dict(**ident,started=time.time()))
        # Keep timeout, guard and worker in one group so NFS waits cannot orphan the worker.
        reason='process_exit'
        while p.poll() is None:
            size=usage([str(root),cfg['neutral_entry']]);window=usage(cfg['window_accounting_roots']);rss=0
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():continue
                try:
                    st=(proc/'stat').read_text().split()
                    if int(st[4])!=os.getpgrp():continue
                    rss+=next((int(z.split()[1])*1024 for z in (proc/'status').read_text().splitlines() if z.startswith('VmRSS:')),0)
                except (FileNotFoundError,ProcessLookupError,PermissionError):pass
            peak=max(peak,size);rsspeak=max(rsspeak,rss)
            write(root/'guardian_resources.private.json',dict(updated=time.time(),phase=phase,new_output_bytes=size,peak_output_bytes=peak,window_new_bytes=window,RSS_sampled_peak_bytes=rsspeak,hard_deadline_epoch=cfg['hard_deadline_epoch']))
            if size>=cfg['storage_stop_threshold_bytes'] or window>=cfg['window_storage_stop_bytes'] or rss>=cfg['RSS_budget_bytes'] or time.time()>=cfg['hard_deadline_epoch']-60:
                reason='budget_prestop';p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill()
                break
            time.sleep(5)
        rc=p.wait();write(root/(phase+'_terminal.private.json'),dict(**ident,returncode=rc,reason=reason,finished=time.time(),elapsed_from_first_run=time.time()-cfg['original_start_epoch']))
        if rc:break
write(root/'guardian_terminal.private.json',dict(finished=time.time(),peak_output_bytes=peak,RSS_sampled_peak_bytes=rsspeak,elapsed_from_first_run=time.time()-cfg['original_start_epoch']))
