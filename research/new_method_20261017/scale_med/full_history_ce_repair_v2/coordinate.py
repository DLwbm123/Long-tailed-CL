"""One child, no retries; wait4 captures resource receipts even for fatal signals."""
import json,os,signal,subprocess,time,resource
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['root']);out=root/('engineering/precheck' if cfg['mode']=='PREFLIGHT' else 'run');out.mkdir(exist_ok=True);assert not (out/'LAUNCH.private.json').exists()
start=time.monotonic();log=(out/'worker.private.log').open('w');p=subprocess.Popen([cfg['python'],cfg['entry']],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
(out/'LAUNCH.private.json').write_text(json.dumps(dict(pid=p.pid,parent_pid=os.getpid(),argv=[cfg['python'],cfg['entry']],source_commit=cfg['source_commit'],started_at=time.time()),indent=2))
reason=None;terminated=None
while True:
    got,status,usage=os.wait4(p.pid,os.WNOHANG)
    if got:break
    if time.monotonic()-start>cfg['wall_budget']-10:reason=reason or 'WALL_LIMIT'
    try:
        fields=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split();child_CPU=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK');own_CPU=resource.getrusage(resource.RUSAGE_SELF)
        if child_CPU+own_CPU.ru_utime+own_CPU.ru_stime>cfg['cpu_budget']-2:reason=reason or 'COMBINED_CPU_LIMIT'
    except FileNotFoundError:pass
    try:total=sum(q.stat().st_size for q in root.rglob('*') if q.is_file())
    except FileNotFoundError:total=0
    if total>256*2**20:reason=reason or 'OUTPUT_LIMIT'
    if reason:
        if terminated is None:p.terminate();terminated=time.monotonic()
        elif time.monotonic()-terminated>5:p.kill()
    time.sleep(.5)
p.returncode=os.waitstatus_to_exitcode(status);log.close();own=resource.getrusage(resource.RUSAGE_SELF)
receipt=dict(worker_exit=p.returncode,stop_reason=reason,wall_seconds=time.monotonic()-start,worker_CPU_seconds=usage.ru_utime+usage.ru_stime,worker_peak_RSS_bytes=usage.ru_maxrss*1024,coordinator_CPU_seconds=own.ru_utime+own.ru_stime,worker_usage_scope='entire_child_including_imports',source_commit=cfg['source_commit'])
(out/'COORDINATOR_EXIT.json').write_text(json.dumps(receipt,indent=2))
if not (out/'EXIT.json').exists():(out/'EXIT.json').write_text(json.dumps(dict(exit_code=p.returncode,status='UNRESOLVED_RESOURCE_OR_SIGNAL',**receipt),indent=2))
