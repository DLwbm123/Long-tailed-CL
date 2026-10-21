"""Original-clock guard for one experiment; whole worker groups only."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from common import settings,write
cfg,out=settings();root=out.parent;config=Path(os.environ['RUN_CONFIG']).resolve()
assert not (root/'guardian.private.json').exists(),'Experiment already launched'
cfg.setdefault('original_start_epoch',time.time());cfg.setdefault('hard_deadline_epoch',cfg['original_start_epoch']+cfg['max_wall_seconds']);assert cfg['hard_deadline_epoch']==cfg['original_start_epoch']+cfg['max_wall_seconds'];write(config,cfg)

def identity(pid):return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
def usage():
    total=0
    for path in list(root.rglob('*'))+list(Path(__file__).parent.rglob('*')):
        try:
            if path.is_symlink():
                s=path.lstat();total+=max(s.st_size,s.st_blocks*512)
            elif path.is_file():
                s=path.stat();total+=max(s.st_size,s.st_blocks*512)
        except FileNotFoundError:pass
    return total

def group_rss(pgid):
    rss=0;members=[]
    for path in Path('/proc').glob('[0-9]*'):
        try:
            # comm may contain spaces; fields after the closing parenthesis start at field 3.
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=pgid:continue
            members.append(int(path.name))
            rss+=next((int(s.split()[1])*1024 for s in (path/'status').read_text().splitlines() if s.startswith('VmRSS:')),0)
        except (FileNotFoundError,ProcessLookupError):pass
    return rss,members

def retire(p):
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    deadline=time.time()+30
    while time.time()<deadline:
        if not group_rss(p.pid)[1]:return
        time.sleep(.5)
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass

write(root/'guardian.private.json',dict(**identity(os.getpid()),original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch'],storage_initial_bytes=usage()))
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
if not cfg.get('verify_only'):assert next(int(r.split(',')[1])*1024**2 for r in rows if int(r.split(',')[0])==cfg['GPU_index'])>=cfg['required_GPU_free_bytes'],'GPU admission failed'
peak=cfg.get('prior_peak_output_bytes',0);rss_peak=cfg.get('prior_RSS_group_sampled_peak_bytes',0)
for name in (['verify'] if cfg.get('verify_only') else ['resume','verify'] if cfg.get('resume_sealed_heads') else ['run','verify']):
    if time.time()>=cfg['hard_deadline_epoch']-60:break
    with (root/(name+'.stdout.private.log')).open('xb') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name(name+'.py'))],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ident=identity(p.pid);write(root/(name+'_launch.private.json'),dict(**ident,started=time.time()));reason='process_exit'
        while p.poll() is None:
            size=usage();rss,members=group_rss(p.pid);peak=max(peak,size);rss_peak=max(rss_peak,rss)
            write(root/'guardian_resources.private.json',dict(updated=time.time(),phase=name,new_output_bytes=size,peak_output_bytes=peak,RSS_group_sampled_peak_bytes=rss_peak,members=members,hard_deadline_epoch=cfg['hard_deadline_epoch']))
            if size>=cfg['storage_stop_threshold_bytes'] or rss>=cfg['RSS_budget_bytes'] or time.time()>=cfg['hard_deadline_epoch']-60:
                reason='storage_prestop' if size>=cfg['storage_stop_threshold_bytes'] else 'RSS_limit' if rss>=cfg['RSS_budget_bytes'] else 'deadline';retire(p);break
            time.sleep(5)
        rc=p.wait();retire(p)
        write(root/(name+'_terminal.private.json'),dict(**ident,returncode=rc,reason=reason,finished=time.time(),elapsed_from_first_run=time.time()-cfg['original_start_epoch'],peak_output_bytes=peak,RSS_group_sampled_peak_bytes=rss_peak))
        if rc:break
write(root/'guardian_terminal.private.json',dict(finished=time.time(),elapsed_from_first_run=time.time()-cfg['original_start_epoch'],peak_output_bytes=max(peak,usage()),RSS_group_sampled_peak_bytes=rss_peak))
