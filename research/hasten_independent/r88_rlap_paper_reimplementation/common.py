import json,time,os
from pathlib import Path

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def allocated(root):return sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
def guard(cfg):
 assert time.time()<json.loads(Path(cfg['clock']).read_text())['deadline_unix']-90,'ORIGINAL_DEADLINE'
 rss=int(next(z for z in Path('/proc/self/status').read_text().splitlines() if z.startswith('VmRSS:')).split()[1])*1024
 assert rss<=cfg.get('rss_budget_bytes',8*2**30),'WORKER_RSS_LIMIT'
