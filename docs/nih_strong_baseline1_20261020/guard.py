"""Independent GPU admission, 24-hour process-group timeout and storage guard."""
import os,sys,time,json,signal,subprocess
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent

def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)


def gpu_free():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();return next(int(v.split(',')[1])*1024**2 for v in lines if int(v.split(',')[0])==cfg['GPU_index'])


def monitor(p,name):
 peak=0;reason='process_exit'
 while p.poll() is None:
  try:
   ss=Path(f'/proc/{p.pid}/status').read_text().splitlines();peak=max(peak,max([int(s.split()[1])*1024 for s in ss if s.startswith(('VmRSS:','VmHWM:'))] or [0]))
  except FileNotFoundError:
   if p.poll() is not None:break
  size=sum(q.stat().st_size for q in root.rglob('*') if q.is_file());write(root/(name+'_resources.private.json'),dict(updated=time.time(),RSS_sampled_peak_bytes=peak,new_output_bytes=size,hard_deadline_epoch=cfg['hard_deadline_epoch']))
  if size>=cfg['storage_stop_threshold_bytes'] or time.time()>=cfg['hard_deadline_epoch']-60:
   reason='storage_limit' if size>=cfg['storage_stop_threshold_bytes'] else 'hard_24h_deadline';os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   break
  time.sleep(5)
 return p.wait(),reason,peak


# Audited I/O repair, not a new scientific run: packing shares the original deadline.
if cfg.get('packed_cache') and not cfg.get('resume_checkpoint'):
 with (root/'packing.stdout.private.log').open('xb') as log:
  start=time.time();p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('pack.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);rc,reason,peak=monitor(p,'packing_guardian');write(root/'packing_terminal.private.json',dict(returncode=rc,reason=reason,wall_seconds=time.time()-start,finished=time.time(),RSS_sampled_peak_bytes=peak))
  if rc!=0:sys.exit(rc)
elif cfg.get('packed_cache'):
 assert json.loads((Path(cfg['packed_cache'])/'PACK_COSTS.private.json').read_text())['state']=='PASS'
 assert json.loads((root/'input_qualification.private.json').read_text())['state']=='PASS'

while gpu_free()<cfg['required_GPU_free_bytes']:
 write(root/'admission.private.json',dict(state='WAITING_GPU',updated=time.time(),required_free_bytes=cfg['required_GPU_free_bytes'],free_bytes=gpu_free()))
 if time.time()>=cfg['hard_deadline_epoch']-60:raise TimeoutError('24h admission deadline; no training launched')
 time.sleep(30)
write(root/'admission.private.json',dict(state='ADMITTED',updated=time.time(),free_bytes=gpu_free(),required_free_bytes=cfg['required_GPU_free_bytes']))
if cfg.get('throughput_qualification'):
 with (root/'throughput.stdout.private.log').open('xb') as log:
  start=time.time();p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('benchmark.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);ident=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]);write(root/'throughput_launch.private.json',ident);rc,reason,peak=monitor(p,'throughput_guardian');write(root/'throughput_terminal.private.json',dict(**ident,returncode=rc,reason=reason,wall_seconds=time.time()-start,finished=time.time(),RSS_sampled_peak_bytes=peak))
  if rc!=0:sys.exit(rc)
 assert json.loads((root/'ENGINEERING_THROUGHPUT.json').read_text())['state']=='PASS'
start=time.time()
with (root/'stdout.private.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);ident=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=start,entry=str(Path(__file__).with_name('run.py')));write(root/'launch.private.json',ident);rc,reason,peak=monitor(p,'guardian');write(root/'terminal.private.json',dict(**ident,returncode=rc,reason=reason,finished=time.time(),wall_seconds=time.time()-start,RSS_sampled_peak_bytes=peak))
if rc==0:
 with (root/'verify.stdout.private.log').open('xb') as log:
  start=time.time();p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('verify.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);rc,reason,peak=monitor(p,'verification_guardian');write(root/'verification_terminal.private.json',dict(returncode=rc,reason=reason,wall_seconds=time.time()-start,finished=time.time(),RSS_sampled_peak_bytes=peak))
else:
 path=Path(cfg['output'])/'status.private.json';s=json.loads(path.read_text()) if path.exists() else {};s.update(state='FAILED',guardian_reason=reason,returncode=rc,updated=time.time());write(path,s)
