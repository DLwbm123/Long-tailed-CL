"""Bound one neutral worker; never retry or overwrite a prior output."""
import json, os, subprocess, time
from pathlib import Path

cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output'])
assert not (root/'LAUNCH.private.json').exists()
start=time.time();log=(root/'worker.private.log').open('w')
p=subprocess.Popen([cfg['python'],cfg['entry']],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
(root/'LAUNCH.private.json').write_text(json.dumps(dict(pid=p.pid,argv=[cfg['python'],cfg['entry']],started_at=start,source_commit=cfg['source_commit']),indent=2))
reason=None
while p.poll() is None:
    if time.time()-start>300:reason='WALL_TIME_LIMIT'
    if sum(f.stat().st_size for f in root.parent.rglob('*') if f.is_file())>32*2**20:reason='OUTPUT_LIMIT'
    if reason:
        p.terminate()
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        break
    time.sleep(2)
log.close()
(root/'COORDINATOR_EXIT.json').write_text(json.dumps(dict(worker_exit=p.returncode,stop_reason=reason,wall_seconds=time.time()-start),indent=2))
if not (root/'EXIT.json').exists():
    (root/'EXIT.json').write_text(json.dumps(dict(exit_code=p.returncode,status='UNRESOLVED_RESOURCE_LIMIT' if reason or p.returncode<0 else 'FAILED',stop_reason=reason,wall_seconds=time.time()-start),indent=2))
