"""Supplement the window guard by conservatively charging default GPU caches."""
import os,json,time,signal,resource
from pathlib import Path
from space import used
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['root']);window=Path(cfg['window_root']);launch=json.loads((root/'run/LAUNCH.private.json').read_text());pid=launch['pid'];parent=launch['parent_pid'];start=time.monotonic();peak=0;reason=None
external=[Path.home()/p for p in ['.nv/ComputeCache','.cache/torch_extensions','.triton','.cache/matplotlib']]
while True:
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
 except FileNotFoundError:break
 if int(fields[1])!=parent or '/tmp/q595/entry.py' not in cmd:break
 charge=used(window)+sum(used(p) for p in external if p.exists())+1000000000;peak=max(peak,charge)
 if charge>=17000000000:
  reason='TOTAL_NEW_SPACE_CONSERVATIVE_GUARD';os.kill(pid,signal.SIGTERM);break
 time.sleep(2)
u=resource.getrusage(resource.RUSAGE_SELF)
(root/'SPACEWATCH_EXIT.json').write_text(json.dumps(dict(stop_reason=reason,conservative_charged_peak_bytes=peak,includes_full_default_cache_size_not_only_growth=True,local_reserved_bytes=1000000000,wall_seconds=time.monotonic()-start,CPU_seconds=u.ru_utime+u.ru_stime),indent=2))
