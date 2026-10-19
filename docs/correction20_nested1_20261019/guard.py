"""One worker, hard wall and RSS budget; never retries."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent;start=time.time()
with (root/'stdout.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 identity=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=start,entry=str(Path(__file__).with_name('run.py')));(root/'launch.private.json').write_text(json.dumps(identity,indent=2));reason='worker_exit'
 while p.poll() is None:
  time.sleep(5)
  if p.poll() is not None:break
  try:
   rss=int(next(x.split()[1] for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
  except (FileNotFoundError,StopIteration):continue
  if time.time()-start>cfg['max_wall_seconds'] or rss>cfg['RSS_budget_bytes']:
   reason='hard_wall_or_RSS_limit';os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   break
 rc=p.wait();(root/'terminal.private.json').write_text(json.dumps(dict(**identity,finished=time.time(),wall_seconds=time.time()-start,returncode=rc,reason=reason),indent=2))
 if rc!=0:
  path=Path(cfg['output'])/'status.private.json';s=json.loads(path.read_text()) if path.exists() else {};s.update(state='FAILED',reason=reason,returncode=rc,updated=time.time());path.write_text(json.dumps(s,indent=2))
