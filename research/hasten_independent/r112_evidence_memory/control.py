"""One original clock: construction -> native barrier -> full four-rule branches."""
import fcntl,importlib,json,os,resource,signal,subprocess,time,traceback
from pathlib import Path
from .supply import write,table
rss_tree=importlib.import_module('r110_ecv_opportunity.controller').rss_tree


def run(cfg):
    root=Path(cfg['root_output']);lock=(root/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,set(cfg['cpu_affinity']));peak=0;disk_peak=0;start=time.time();cpu=time.process_time();history={};active={};reason=None
    def limits(cores):
        os.sched_setaffinity(0,set(cores));resource.setrlimit(resource.RLIMIT_CPU,(400000,400000));resource.setrlimit(resource.RLIMIT_FSIZE,(2**30,2**30))
    def execute(phase):
        nonlocal peak,disk_peak,reason,active
        children={};exits={};term=None
        available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
        assert available>=64*2**30
        for n in ['a','b']:
            w=json.loads((root/n/(phase.upper()+'.private.json')).read_text());free=int(subprocess.check_output(['nvidia-smi','-i',str(w['physical_gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20;assert free>=14*2**30
            env={**os.environ,**w['environment'],'Q_SOURCE':str(root.resolve()),'Q_MODULE':'worker','Q_CONFIG':str(root/n/(phase.upper()+'.private.json'))}
            log=(Path(w['output'])/'worker.private.log').open('w');p=subprocess.Popen([cfg['python'],str(root/'main.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda c=w['cpu_affinity']:limits(c));log.close();children[n]=p
        active=children;write(root/(phase.upper()+'_LAUNCH.private.json'),{'controller':os.getpid(),'workers':{n:p.pid for n,p in children.items()},'original_start':cfg['start'],'deadline':cfg['deadline']})
        while len(exits)<2:
            for n,p in children.items():
                if n in exits:continue
                pid,status,u=os.wait4(p.pid,os.WNOHANG)
                if pid:
                    p.returncode=os.waitstatus_to_exitcode(status);exits[n]={'exit_code':p.returncode,'CPU_seconds':u.ru_utime+u.ru_stime,'peak_RSS_bytes':u.ru_maxrss*1024}
                    if p.returncode:reason=reason or phase.upper()+'_WORKER_FAILURE'
            if phase=='science' and not (root/'GLOBAL_NATIVE_ADMISSION.json').exists() and not (root/'GLOBAL_NATIVE_FAIL.json').exists() and all((root/n/'science/NATIVE_ADMISSION.json').exists() for n in ['a','b']):
                native={n:json.loads((root/n/'science/NATIVE_ADMISSION.json').read_text()) for n in ['a','b']};ok=all(x['passed'] and x['complete_scientific_projected_seconds']<cfg['deadline']-time.time() for x in native.values())
                write(root/('GLOBAL_NATIVE_ADMISSION.json' if ok else 'GLOBAL_NATIVE_FAIL.json'),{'passed':ok,'workers':native,'epoch':time.time()})
            total=rss_tree(os.getpid());peak=max(peak,total);rss={n:rss_tree(p.pid) for n,p in children.items() if n not in exits}
            allocated=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in root.rglob('*') if p.is_file() and not p.is_symlink());disk_peak=max(disk_peak,allocated)
            if time.time()>=cfg['deadline']-120:reason=reason or 'ORIGINAL_CLOCK'
            if total>=29_000_000_000 or any(v>=28_000_000_000 for v in rss.values()):reason=reason or 'RAM_LIMIT'
            if allocated+512*2**20>=20_000_000_000:reason=reason or 'DISK_LIMIT'
            write(root/'GUARD.private.json',{'phase':phase,'remaining_seconds':cfg['deadline']-time.time(),'allocated_bytes':allocated,'peak_group_RSS_bytes':peak,'reason':reason,'exits':exits})
            if reason:
                sig=signal.SIGTERM if term is None else signal.SIGKILL if time.time()-term>10 else None
                if sig:
                    for n,p in children.items():
                        if n not in exits:
                            try:os.killpg(p.pid,sig)
                            except ProcessLookupError:pass
                    if term is None:term=time.time()
            time.sleep(2)
        active={};history[phase]=exits
        if reason:raise RuntimeError(reason)
    try:
        execute('construction')
        results={n:json.loads((root/n/'construction/CONSTRUCTION_RESULT.json').read_text()) for n in ['a','b']};passed=all(x['passed'] for x in results.values())
        records=[x for n in ['a','b'] for x in json.loads((root/n/'construction/ROOT_CONSTRUCTION.json').read_text())]
        table(root/'CONSTRUCTION_COMPARISON.csv',records);write(root/'CONSTRUCTION_ADMISSION.json',{'passed':passed,'workers':results,'planned_roots':64,'scientific_training_started':False})
        if not passed:
            write(root/'STATUS.json',{'phase':'COMPLETE_CONSTRUCTION_NOT_ADMITTED','decision':'B','science_classifier_updates':0,'automatic_successor':False});return
        write(root/'SCIENTIFIC_FREEZE.json',{'epoch':time.time(),'original_start':cfg['start'],'deadline':cfg['deadline'],'source_manifest':json.loads((root/'SOURCE_FREEZE.json').read_text()),'specification':'PROTOCOL_NEW.md','construction_gate_passed':True,'max_science_updates':1170864,'controls':['E-ZERO','E-BAL','E-ANCHOR','E-LOCAL'],'no_protocol_retuning':True})
        write(root/'STATUS.json',{'phase':'NATIVE_ADMISSION_THEN_CONDITIONAL_P1C','construction_passed':True})
        execute('science')
        if (root/'GLOBAL_NATIVE_FAIL.json').exists():write(root/'STATUS.json',{'phase':'COMPLETE_NOT_RESOURCE_ADMITTED','science_classifier_updates':0,'automatic_successor':False});return
        from .report import run as report
        report(root);write(root/'STATUS.json',{'phase':'COMPLETE_P1C_READY_FOR_DELIVERY','automatic_successor':False})
    except BaseException:
        reason=reason or 'CONTROL_FAILURE';(root/'CONTROL_FAILURE.private.txt').write_text(traceback.format_exc())
        for p in active.values():
            try:os.killpg(p.pid,signal.SIGKILL);os.waitpid(p.pid,0)
            except ProcessLookupError:pass
        write(root/'STATUS.json',{'phase':'FAILED_PRESERVED','reason':reason,'automatic_successor':False})
    finally:write(root/'CONTROL_EXIT.json',{'exit_code':int(reason is not None),'reason':reason,'workers':history,'controller_CPU_seconds':time.process_time()-cpu,'controller_wall_seconds':time.time()-start,'original_elapsed_seconds':time.time()-cfg['start'],'peak_group_RSS_bytes':peak,'guard_observed_peak_disk_bytes':disk_peak})

if __name__=='__main__':run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))
