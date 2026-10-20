"""Standalone deadline/storage guardian for the worker and its independent verifier."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent

def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)


def monitor(p,name):
 peak=0;reason='process_exit';size=0
 while p.poll() is None:
  try:
   lines=Path(f'/proc/{p.pid}/status').read_text().splitlines();peak=max(peak,max([int(v.split()[1])*1024 for v in lines if v.startswith(('VmRSS:','VmHWM:'))] or [0]))
  except FileNotFoundError:
   if p.poll() is not None:break
  size=sum(q.stat().st_size for rr in [root,*map(Path,cfg['prior_roots'])] for q in rr.rglob('*') if q.is_file());write(root/(name+'_resources.private.json'),dict(RSS_sampled_peak_bytes=peak,new_remote_output_bytes=size,updated=time.time(),hard_deadline_epoch=cfg['hard_deadline_epoch']))
  if size>=cfg['new_storage_cap_bytes']-100000000 or time.time()>=cfg['hard_deadline_epoch']-30:
   reason='storage_limit' if size>=cfg['new_storage_cap_bytes']-100000000 else 'hard_24h_deadline';os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   break
  time.sleep(2)
 return p.wait(),reason,peak

start=time.time()
with (root/'stdout.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);ident=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=start,entry=str(Path(__file__).with_name('run.py')));write(root/'launch.private.json',ident);rc,reason,peak=monitor(p,'guardian');write(root/'terminal.private.json',dict(**ident,finished=time.time(),wall_seconds=time.time()-start,returncode=rc,reason=reason,RSS_sampled_peak_bytes=peak))
if rc!=0:
 path=Path(cfg['output'])/'status.private.json';s=json.loads(path.read_text()) if path.exists() else {};s.update(state='FAILED',reason=reason,returncode=rc,updated=time.time());write(path,s)
else:
 with (root/'verify.stdout.private.log').open('xb') as log:
  start=time.time();p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('verify.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);rc,reason,peak=monitor(p,'verification_guardian');write(root/'verification_terminal.private.json',dict(pid=p.pid,returncode=rc,reason=reason,wall_seconds=time.time()-start,finished=time.time(),RSS_sampled_peak_bytes=peak))
