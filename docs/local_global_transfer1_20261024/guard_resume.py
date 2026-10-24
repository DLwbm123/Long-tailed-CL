"""Independent whole-group resource guard enclosed by an NFS-independent timeout."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def identity(pid):return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
def usage(locations):
    total=0
    for loc in locations:
        for p in Path(loc).rglob('*'):
            try:
                if p.is_file():s=p.stat();total+=max(s.st_size,s.st_blocks*512)
            except FileNotFoundError:pass
    return total

def group_resources():
    rss=0;pids=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            st=(p/'stat').read_text().split()
            if int(st[4])!=os.getpgrp():continue
            pids.append(int(p.name));rss+=next((int(x.split()[1])*1024 for x in (p/'status').read_text().splitlines() if x.startswith('VmRSS:')),0)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    gpu=sum(int(line.split(',')[1])*1024**2 for line in out.splitlines() if int(line.split(',')[0]) in pids)
    return rss,gpu,pids

assert not (root/'resume_guardian.private.json').exists(),'No automatic relaunch'
assert cfg['hard_deadline_epoch']==cfg['original_start_epoch']+cfg['max_wall_seconds']
assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024>=cfg['required_host_free_bytes']
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
assert next(int(line.split(',')[1])*1024**2 for line in rows.splitlines() if int(line.split(',')[0])==cfg['GPU_index'])>=cfg['required_GPU_free_bytes']
write(root/'resume_guardian.private.json',dict(**identity(os.getpid()),original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch']))
peak=rsspeak=gpupeak=0
for phase in ['resume_validation','verify']:
    if time.time()>=cfg['hard_deadline_epoch']-60:break
    with (root/(phase+'.stdout.private.log')).open('xb') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name(phase+'.py'))],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
        who=identity(p.pid);write(root/(phase+'_launch.private.json'),dict(**who,started=time.time()));reason='process_exit'
        while p.poll() is None:
            size=usage([str(root),cfg['neutral_entry']]);window=usage(cfg['window_accounting_roots']);rss,gpu,pids=group_resources();peak=max(peak,size);rsspeak=max(rsspeak,rss);gpupeak=max(gpupeak,gpu)
            write(root/'resume_guardian_resources.private.json',dict(phase=phase,updated=time.time(),new_output_bytes=size,peak_output_bytes=peak,window_new_bytes=window,RSS_sampled_peak_bytes=rsspeak,GPU_process_sampled_peak_bytes=gpupeak,group_pids=pids,hard_deadline_epoch=cfg['hard_deadline_epoch']))
            if size>=cfg['storage_stop_threshold_bytes'] or window>=cfg['window_storage_stop_bytes'] or rss>=cfg['RSS_budget_bytes'] or gpu>=cfg['gpu_reserved_budget_bytes'] or time.time()>=cfg['hard_deadline_epoch']-60:
                reason='budget_prestop';p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill()
                break
            time.sleep(5)
        rc=p.wait();write(root/(phase+'_terminal.private.json'),dict(**who,returncode=rc,reason=reason,finished=time.time(),elapsed_from_first=time.time()-cfg['original_start_epoch']))
        if rc:break
write(root/'resume_guardian_terminal.private.json',dict(finished=time.time(),elapsed_from_first=time.time()-cfg['original_start_epoch'],sampled_peak_output_bytes=peak,RSS_sampled_peak_bytes=rsspeak,GPU_process_sampled_peak_bytes=gpupeak))
