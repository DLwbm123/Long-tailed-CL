"""Bound a retained-checkpoint audit within the original static-series clock."""
import os,json,time,sys,subprocess,signal
from pathlib import Path
from common import write
cfgpath=Path(os.environ['RUN_CONFIG']);cfg=json.loads(cfgpath.read_text())
out=Path(cfg['audit_output']);out.mkdir(exist_ok=False)
series=Path(cfg['output']).parent
assert cfg['hard_deadline_epoch']==1792559828.1970181
assert cfg['original_start_epoch']==1792473428.1970181
assert time.time()<cfg['hard_deadline_epoch']-120
cfg['audit_deadline_epoch']=min(cfg['hard_deadline_epoch'],time.time()+900)
write(cfgpath,cfg)
free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
assert next(int(s.split(',')[1]) for s in free.splitlines() if int(s.split(',')[0])==0)>=8192
write(out/'guardian.private.json',dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch'],subtask_deadline_epoch=cfg['audit_deadline_epoch']))
with (out/'stdout.private.log').open('x') as log:
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('audit.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 ident=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]);write(out/'launch.private.json',ident)
 reason='process_exit';peak=0
 while p.poll() is None:
  size=sum(max(q.stat().st_size,q.stat().st_blocks*512) for q in series.rglob('*') if q.is_file());peak=max(peak,size)
  write(out/'resources.private.json',dict(series_bytes=size,peak_series_bytes=peak,updated=time.time()))
  if size>=48000000000 or time.time()>=cfg['audit_deadline_epoch']-60:
   reason='original_budget_or_audit_subtask_limit';os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   break
  time.sleep(5)
 rc=p.wait()
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 write(out/'terminal.private.json',dict(**ident,returncode=rc,reason=reason,finished=time.time(),series_elapsed_seconds=time.time()-cfg['original_start_epoch'],peak_series_bytes=peak))
