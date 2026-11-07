"""Independent hard guard for the bounded two-worker input check."""
import fcntl,importlib,json,os,resource,signal,subprocess,time,traceback
from pathlib import Path
from .readiness import write
rss_tree=importlib.import_module('r110_ecv_opportunity.controller').rss_tree


def run(cfg):
    root=Path(cfg['root_output']);lock=(root/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,set(cfg['cpu_affinity']));children={};exits={};reason=None;term=None;peak=0;disk_peak=0;start=time.time();cpu=time.process_time()
    def limits(cores):
        os.sched_setaffinity(0,set(cores));resource.setrlimit(resource.RLIMIT_CPU,(400000,400000));resource.setrlimit(resource.RLIMIT_FSIZE,(2**30,2**30))
    try:
        avail=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
        assert avail>=64*2**30
        for name in ['a','b']:
            w=json.loads((root/name/'CONFIG.private.json').read_text());free=int(subprocess.check_output(['nvidia-smi','-i',str(w['physical_gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
            assert free>=14*2**30
            env={**os.environ,**w['environment'],'Q_SOURCE':str(root.resolve()),'Q_MODULE':'readiness','Q_CONFIG':str(root/name/'CONFIG.private.json')}
            log=(root/name/'worker.private.log').open('w');p=subprocess.Popen([cfg['python'],str(root/'main.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda c=w['cpu_affinity']:limits(c));log.close();children[name]=p
        write(root/'LAUNCH.private.json',{'controller':os.getpid(),'workers':{n:p.pid for n,p in children.items()},'original_start':cfg['start'],'deadline':cfg['deadline']})
        while len(exits)<2:
            for n,p in children.items():
                if n in exits:continue
                pid,status,u=os.wait4(p.pid,os.WNOHANG)
                if pid:
                    p.returncode=os.waitstatus_to_exitcode(status);exits[n]={'exit_code':p.returncode,'CPU_seconds':u.ru_utime+u.ru_stime,'peak_RSS_bytes':u.ru_maxrss*1024}
                    if p.returncode:reason=reason or 'WORKER_FAILURE'
            rss={n:rss_tree(p.pid) for n,p in children.items() if n not in exits};total=rss_tree(os.getpid());peak=max(peak,total)
            allocated=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in root.rglob('*') if p.is_file() and not p.is_symlink());disk_peak=max(disk_peak,allocated)
            if time.time()>=cfg['deadline']-120:reason=reason or 'ORIGINAL_CLOCK'
            if total>=29_000_000_000 or any(x>=28_000_000_000 for x in rss.values()):reason=reason or 'RAM_LIMIT'
            if allocated+512*2**20>=20_000_000_000:reason=reason or 'DISK_LIMIT'
            write(root/'GUARD.private.json',{'remaining_seconds':cfg['deadline']-time.time(),'allocated_bytes':allocated,'peak_group_RSS_bytes':peak,'reason':reason,'exits':exits})
            if reason:
                sig=signal.SIGTERM if term is None else signal.SIGKILL if time.time()-term>10 else None
                if sig:
                    for n,p in children.items():
                        if n not in exits:
                            try:os.killpg(p.pid,sig)
                            except ProcessLookupError:pass
                    if term is None:term=time.time()
            time.sleep(2)
        if reason is None:
            results={n:json.loads((root/n/'READINESS_RESULT.json').read_text()) for n in children}
            passed=all(x['passed'] for x in results.values());write(root/'INPUT_ADMISSION.json',{'passed':passed,'both_streams':results,'scientific_P1b_started':False,'next_step':'freeze full P1b protocol and native budget' if passed else 'task construction must be revised before P1b; no training admission'})
            write(root/'STATUS.json',{'phase':'COMPLETE_INPUT_READINESS','passed':passed})
    except BaseException:
        reason=reason or 'CONTROL_FAILURE';(root/'CONTROL_FAILURE.private.txt').write_text(traceback.format_exc())
        for n,p in children.items():
            if n not in exits:
                try:os.killpg(p.pid,signal.SIGKILL);os.waitpid(p.pid,0)
                except ProcessLookupError:pass
        write(root/'STATUS.json',{'phase':'FAILED_PRESERVED','reason':reason})
    finally:write(root/'CONTROL_EXIT.json',{'exit_code':int(reason is not None),'reason':reason,'workers':exits,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'original_elapsed_seconds':time.time()-cfg['start'],'peak_group_RSS_bytes':peak,'peak_disk_bytes':disk_peak})

if __name__=='__main__':run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))
