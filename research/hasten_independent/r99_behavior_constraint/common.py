import json,time,os,resource
from pathlib import Path

def write(path,obj):
 p=Path(path);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
def save(path,obj):
 import torch
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(obj,tmp);tmp.replace(p)
def guard(cfg):
 assert time.time()<cfg['deadline']-90,'ORIGINAL_CLOCK_LIMIT'
 import torch
 if torch.cuda.is_initialized():assert torch.cuda.max_memory_reserved()<12*2**30,'GPU_RESERVED_LIMIT'
 rss=int(next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')).split()[1])*1024;assert rss<28_000_000_000

def allocated(root):return sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
