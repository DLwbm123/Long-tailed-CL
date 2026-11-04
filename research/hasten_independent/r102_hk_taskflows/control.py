"""Two-GPU bounded campaign with one global seal and original24h hard timeout."""
import os,json,time,resource,subprocess,signal,fcntl,traceback
from pathlib import Path
from common import write,allocated
from memory_guard import group_rss
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);start=time.time();lock=(r/'.duplicate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
children={};exits={};reason=None;term=0;peak=0;peakrss=0
def limits(affinity):
    os.sched_setaffinity(0,set(affinity));resource.setrlimit(resource.RLIMIT_CPU,(400000,400000));resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3));os.setsid()
try:
    free=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024;assert free>=64*2**30
    for folder in ['a','b']:
        wc=json.loads((r/folder/'RUN_CONFIG.private.json').read_text());gpu=int(subprocess.check_output(['nvidia-smi','-i',str(wc['physical_gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20;assert gpu>=14*2**30
    for folder in ['a','b']:
        wc=json.loads((r/folder/'RUN_CONFIG.private.json').read_text());env=os.environ.copy();env.update(wc['environment']);env['RUN_CONFIG']=str(r/folder/'RUN_CONFIG.private.json')
        f=(r/folder/'worker.private.log').open('w');p=subprocess.Popen([wc['python'],str(Path(c['runtime'])/'worker.py')],cwd=r,env=env,preexec_fn=lambda a=wc['cpu_affinity']:limits(a),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);f.close();children[folder]=p
    write(r/'LAUNCH.private.json',{'controller':os.getpid(),'workers':{k:v.pid for k,v in children.items()},'original_start':c['start'],'deadline':c['deadline']})
    while len(exits)<2:
        for folder,p in children.items():
            if folder in exits:continue
            pid,status,u=os.wait4(p.pid,os.WNOHANG)
            if pid:
                code=os.waitstatus_to_exitcode(status);p.returncode=code;exits[folder]={'exit_code':code,'worker_CPU_seconds':u.ru_utime+u.ru_stime,'worker_peak_RSS_bytes':u.ru_maxrss*1024}
                if code:reason=reason or 'WORKER_FAILURE'
        rss=group_rss(os.getpgrp(),extra_pid=os.getpid())[0]
        rss+=sum(group_rss(p.pid)[0] for k,p in children.items() if k not in exits);peakrss=max(peakrss,rss);peak=max(peak,allocated(r))
        if not (r/'GLOBAL_LEARNING_SEAL.json').exists() and all((r/k/'TRAIN_COMPLETE.json').exists() for k in ['a','b']):
            totals=[json.loads((r/k/'SEAL.json').read_text()) for k in ['a','b']];assert all(x['final_endpoints']==20 and x['total_optimizer_steps']==17920 for x in totals)
            write(r/'GLOBAL_LEARNING_SEAL.json',{'final_learning_endpoints':40,'teacher_endpoints':10,'module_total_steps':35840,'time':time.time(),'both_streams_sealed_before_any_module_F_VAL':True})
        if reason is None:
            if time.time()>c['deadline']-120:reason='ORIGINAL_CLOCK_LIMIT'
            elif rss>=29_000_000_000:reason='COMBINED_RAM_LIMIT'
            elif peak+512*2**20>=c['storage_budget_bytes']:reason='STORAGE_LIMIT'
        write(r/'GUARD.private.json',{'remaining_seconds':c['deadline']-time.time(),'combined_RSS_bytes':rss,'peak_sampled_RSS_bytes':peakrss,'allocated_bytes':peak,'exits':exits,'reason':reason})
        if reason:
            if term==0:
                for k,p in children.items():
                    if k not in exits:
                        try:os.killpg(p.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                term=time.time()
            elif time.time()-term>10:
                for k,p in children.items():
                    if k not in exits:
                        try:os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
        time.sleep(2)
    if all(v['exit_code']==0 for v in exits.values()) and reason is None:
        from combine import run
        run(r);write(r/'STATUS.json',{'phase':'COMPLETE_TWO_STREAMS_READY_FOR_DELIVERY','no_followup_experiment':True})
    else:write(r/'STATUS.json',{'phase':'FAILED_PRESERVED','reason':reason,'exits':exits})
except BaseException:
    (r/'CONTROL_FAILURE.private.txt').write_text(traceback.format_exc());reason=reason or 'CONTROLLER_FAILURE'
    for k,p in children.items():
        if k not in exits:
            try:os.killpg(p.pid,signal.SIGKILL);_,s,u=os.wait4(p.pid,0);exits[k]={'exit_code':os.waitstatus_to_exitcode(s),'worker_CPU_seconds':u.ru_utime+u.ru_stime,'worker_peak_RSS_bytes':u.ru_maxrss*1024}
            except ProcessLookupError:pass
    write(r/'STATUS.json',{'phase':'FAILED_PRESERVED','reason':reason})
finally:
    write(r/'CONTROL_EXIT.json',{'exit_code':0 if reason is None and len(exits)==2 and all(v['exit_code']==0 for v in exits.values()) else 1,'reason':reason,'workers':exits,'controller_CPU_seconds':time.process_time(),'controller_wall_seconds':time.time()-start,'original_elapsed_seconds':time.time()-c['start'],'peak_sampled_group_RSS_bytes':peakrss,'allocated_bytes':allocated(r),'original_deadline':c['deadline']})
