"""Single neutral worker with immutable resource caps and no retry."""
import json,os,subprocess,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);assert not (out/'LAUNCH.private.json').exists()
t=time.monotonic();log=(out/'worker.private.log').open('w');p=subprocess.Popen([cfg['python'],cfg['entry']],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
(out/'LAUNCH.private.json').write_text(json.dumps(dict(pid=p.pid,parent_pid=os.getpid(),argv=[cfg['python'],cfg['entry']],source_commit=cfg['source_commit'],started_at=time.time()),indent=2))
reason=None
while p.poll() is None:
    if time.monotonic()-t>1800:reason='WALL_LIMIT'
    if sum(q.stat().st_size for q in out.parent.rglob('*') if q.is_file())>256*2**20:reason='OUTPUT_LIMIT'
    if reason:
        p.terminate()
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        break
    time.sleep(2)
log.close();receipt=dict(worker_exit=p.returncode,stop_reason=reason,wall_seconds=time.monotonic()-t)
(out/'COORDINATOR_EXIT.json').write_text(json.dumps(receipt,indent=2))
if not (out/'EXIT.json').exists():(out/'EXIT.json').write_text(json.dumps(dict(exit_code=p.returncode,status='UNRESOLVED_RESOURCE_OR_SIGNAL',**receipt),indent=2))
