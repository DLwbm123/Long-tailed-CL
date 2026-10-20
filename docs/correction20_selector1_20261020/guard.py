"""Neutral background guardian: preserve resource evidence and enforce disk/window only."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent;start=time.time();rss_peak=0

def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)

with (root/'stdout.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 ident=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=start,entry=str(Path(__file__).with_name('run.py')));write(root/'launch.private.json',ident);reason='worker_exit'
 while p.poll() is None:
  try:
   status=Path(f'/proc/{p.pid}/status').read_text().splitlines();rss_peak=max(rss_peak,max([int(v.split()[1])*1024 for v in status if v.startswith(('VmRSS:','VmHWM:'))] or [0]));size=sum(q.stat().st_size for rr in [root,*map(Path,cfg['prior_roots'])] for q in rr.rglob('*') if q.is_file());write(root/'guardian_resources.private.json',dict(RSS_sampled_peak_bytes=rss_peak,new_remote_output_bytes=size,updated=time.time()))
  except FileNotFoundError:
   if p.poll() is not None:break
   raise
  if size>=cfg['new_storage_cap_bytes']-100000000:
   reason='storage_limit';os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   break
  time.sleep(2)
 rc=p.wait();write(root/'terminal.private.json',dict(**ident,finished=time.time(),wall_seconds=time.time()-start,returncode=rc,reason=reason,RSS_sampled_peak_bytes=rss_peak))
 if rc!=0:
  path=Path(cfg['output'])/'status.private.json';s=json.loads(path.read_text()) if path.exists() else {};s.update(state='FAILED',reason=reason,returncode=rc,updated=time.time(),guardian_RSS_sampled_peak_bytes=rss_peak);write(path,s)

# Verification is part of this authorized round; it never fits a model.
if rc==0:
 with (root/'verify.stdout.private.log').open('xb') as v:
  verify_start=time.time();q=subprocess.run([sys.executable,str(Path(__file__).with_name('verify.py'))],stdout=v,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  write(root/'verification_terminal.private.json',dict(returncode=q.returncode,wall_seconds=time.time()-verify_start,finished=time.time()))
