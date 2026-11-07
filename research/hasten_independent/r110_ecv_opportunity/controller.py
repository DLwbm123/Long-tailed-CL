"""Two-worker admission barrier and independent original-clock/resource guard."""
import os,json,time,signal,subprocess,resource,fcntl,traceback
from pathlib import Path
from .common import write


def rss_tree(pid):
    p=Path(f'/proc/{pid}');total=0
    try:
        total=int(next(s for s in (p/'status').read_text().splitlines() if s.startswith('VmRSS:')).split()[1])*1024
        for child in (p/'task'/str(pid)/'children').read_text().split():total+=rss_tree(int(child))
    except (FileNotFoundError,StopIteration,ProcessLookupError):pass
    return total


def run(cfg):
    root=Path(cfg['root_output']);lock=(root/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,set(cfg['cpu_affinity']));start=time.time();cpu=time.process_time();children={};exits={};peak=0;disk_peak=0;reason=None;term=None
    def limits(cores):
        os.sched_setaffinity(0,set(cores));resource.setrlimit(resource.RLIMIT_CPU,(400000,400000));resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3))
    try:
        available=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
        assert available>=64*2**30
        for name in ['a','b']:
            w=json.loads((root/name/'CONFIG.private.json').read_text())
            free=int(subprocess.check_output(['nvidia-smi','-i',str(w['physical_gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
            assert free>=14*2**30,'GPU_HEADROOM'
            env={**os.environ,**w['environment'],'Q_SOURCE':str(root.resolve()),'Q_MODULE':'worker','Q_CONFIG':str(root/name/'CONFIG.private.json')}
            log=(root/name/'worker.private.log').open('w');p=subprocess.Popen([cfg['python'],str(root/'main.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda cores=w['cpu_affinity']:limits(cores));log.close();children[name]=p
        write(root/'LAUNCH.private.json',{'controller':os.getpid(),'workers':{k:p.pid for k,p in children.items()},'original_start':cfg['start'],'deadline':cfg['deadline']})
        while len(exits)<len(children):
            for name,p in children.items():
                if name in exits:continue
                pid,status,usage=os.wait4(p.pid,os.WNOHANG)
                if pid:
                    p.returncode=os.waitstatus_to_exitcode(status);exits[name]={'exit_code':p.returncode,'worker_CPU_seconds':usage.ru_utime+usage.ru_stime,'peak_RSS_bytes':usage.ru_maxrss*1024}
                    if p.returncode:reason=reason or 'WORKER_FAILURE'
            if not (root/'GLOBAL_NATIVE_ADMISSION.json').exists() and not (root/'GLOBAL_NATIVE_FAIL.json').exists() and all((root/n/'NATIVE_ADMISSION.json').exists() for n in ['a','b']):
                native=[json.loads((root/n/'NATIVE_ADMISSION.json').read_text()) for n in ['a','b']]
                passed=all(v['passed'] and v['full_P1_projected_seconds']<cfg['deadline']-time.time() for v in native)
                write(root/('GLOBAL_NATIVE_ADMISSION.json' if passed else 'GLOBAL_NATIVE_FAIL.json'),{'passed':passed,'both_streams':native,'epoch':time.time()})
            rss={n:rss_tree(p.pid) for n,p in children.items() if n not in exits}
            # rss_tree(controller) already includes live children: do not double count.
            total=rss_tree(os.getpid());peak=max(peak,total)
            allocated=0
            for path in root.rglob('*'):
                try:
                    if path.is_file() and not path.is_symlink():
                        stat=path.stat();allocated+=max(stat.st_size,stat.st_blocks*512)
                except FileNotFoundError:pass
            disk_peak=max(disk_peak,allocated)
            if time.time()>cfg['deadline']-120:reason=reason or 'ORIGINAL_CLOCK'
            if total>=29_000_000_000 or any(v>=28_000_000_000 for v in rss.values()):reason=reason or 'RAM_LIMIT'
            if allocated+512*2**20>=20_000_000_000:reason=reason or 'DISK_LIMIT'
            write(root/'GUARD.private.json',{'allocated_bytes':allocated,'peak_group_RSS_bytes':peak,'remaining_seconds':cfg['deadline']-time.time(),'reason':reason,'exits':exits})
            if reason:
                if term is None:
                    for name,p in children.items():
                        if name not in exits:
                            try:os.killpg(p.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                    term=time.time()
                elif time.time()-term>10:
                    for name,p in children.items():
                        if name not in exits:
                            try:os.killpg(p.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
            time.sleep(2)
        if reason is None and all(v['exit_code']==0 for v in exits.values()):
            if (root/'GLOBAL_NATIVE_FAIL.json').exists():write(root/'STATUS.json',{'phase':'COMPLETE_NOT_ADMITTED','science_optimizer':0})
            else:
                from .report import run as report
                report(root);write(root/'STATUS.json',{'phase':'COMPLETE_P1_READY_FOR_DELIVERY','automatic_P2':False})
        else:write(root/'STATUS.json',{'phase':'FAILED_PRESERVED','reason':reason})
    except BaseException:
        reason=reason or 'CONTROLLER_FAILURE';(root/'CONTROL_FAILURE.private.txt').write_text(traceback.format_exc())
        for name,p in children.items():
            if name not in exits:
                try:os.killpg(p.pid,signal.SIGKILL);_,status,usage=os.wait4(p.pid,0);exits[name]={'exit_code':os.waitstatus_to_exitcode(status),'worker_CPU_seconds':usage.ru_utime+usage.ru_stime}
                except ProcessLookupError:pass
        write(root/'STATUS.json',{'phase':'FAILED_PRESERVED','reason':reason})
    finally:
        write(root/'CONTROL_EXIT.json',{'exit_code':0 if reason is None else 1,'reason':reason,'workers':exits,'controller_CPU_seconds':time.process_time()-cpu,
                                      'controller_wall_seconds':time.time()-start,'original_elapsed_seconds':time.time()-cfg['start'],'peak_RSS_bytes':peak,'peak_disk_bytes':disk_peak})

if __name__=='__main__':run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))

