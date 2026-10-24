import json,time,os
from pathlib import Path

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def allocated(root):return sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
def guard(cfg):assert time.time()<json.loads(Path(cfg['clock']).read_text())['deadline_unix']-90,'ORIGINAL_DEADLINE'
def request_cache(cfg,job):
 root=Path(cfg['output']);dest=root/'cache'/str(job['id']);ready=dest/'READY.json'
 if not ready.exists():
  write(root/'REQUEST.json',job)
  while not ready.exists():
   guard(cfg);assert not (root/'RELAY_FAILURE.private.txt').exists(),'RELAY_FAILURE';time.sleep(2)
  if (root/'REQUEST.json').exists() and json.loads((root/'REQUEST.json').read_text())['id']==job['id']:(root/'REQUEST.json').unlink()
 return dest
