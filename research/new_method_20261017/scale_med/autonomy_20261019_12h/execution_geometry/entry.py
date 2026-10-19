import os,json,time,resource,traceback
from pathlib import Path
import numpy as np
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
resource.setrlimit(resource.RLIMIT_FSIZE,(512*2**20,512*2**20));torch.set_num_threads(4)
def write(directory,name,obj):
 s=json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)));f=directory/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(s);t.replace(f)
def guarded(action,directory):
 directory.mkdir(exist_ok=True,parents=True);start=time.monotonic();code=1
 try:action();code=0
 except BaseException as error:
  write(directory,'FAILURE.json',dict(error=type(error).__name__,phase=cfg['mode'],no_silent_retry=True));(directory/'FAILURE.private.txt').write_text(traceback.format_exc())
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(directory,'WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized(),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0));write(directory,'EXIT.json',dict(exit_code=code))
 return code

def action():
 from audit import main
 main(cfg)
raise SystemExit(guarded(action,out))
