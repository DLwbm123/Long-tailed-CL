"""Single authorized continuation with the unreset cumulative wall limit."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent
baseline=float(os.environ['CUMULATIVE_BASE_WALL']);start=time.time()
remaining=cfg['max_wall_seconds']-baseline-30
assert remaining>0
with (root/'cert.stdout.private.log').open('xb') as log:
 worker=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 identity=dict(pid=worker.pid,started=start,start_ticks=Path(f'/proc/{worker.pid}/stat').read_text().split()[21],attempts=3,entry=str(Path(__file__).with_name('run.py')))
 (root/'cert_launch.private.json').write_text(json.dumps(identity,indent=2))
 try:rc=worker.wait(timeout=remaining);reason='worker_exit'
 except subprocess.TimeoutExpired:
  reason='cumulative_hard_wall_limit';os.killpg(worker.pid,signal.SIGTERM)
  try:rc=worker.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);rc=worker.wait(timeout=10)
 terminal=dict(**identity,finished=time.time(),elapsed_seconds=time.time()-start,prior_and_repair_wall_seconds=baseline,returncode=rc,reason=reason)
 (root/'cert_terminal.private.json').write_text(json.dumps(terminal,indent=2))
 if rc!=0:
  path=Path(cfg['output'])/'status.private.json';state=json.loads(path.read_text())
  if state.get('state')!='FAILED':
   state.update(state='STOPPED',reason=reason,returncode=rc,updated=time.time(),cumulative_wall_seconds=baseline+time.time()-start);path.write_text(json.dumps(state,indent=2))
