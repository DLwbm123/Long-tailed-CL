"""Engineering-only raw PNG packing. Preserves original encoded bytes and transforms."""
import time
START=time.time()
import os,sys,json,signal,resource,io
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from common import rows,Images,seed_all,write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());STOP=False

def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop)
root=Path(cfg['packed_cache']);root.mkdir(exist_ok=False);data,audit=rows(cfg);items=data['train']+data['val'];names=[];offsets=[];sizes=[];position=0

def read(z):
 if STOP:raise TimeoutError('packing termination requested')
 path,label,group=z
 with open(path,'rb') as f:b=f.read()
 return Path(path).name,b

try:
 with ThreadPoolExecutor(max_workers=16) as executor,(root/'images.private.pack').open('xb') as output:
  source=iter(items);pending=deque()
  for _ in range(64):
   z=next(source,None)
   if z is not None:pending.append(executor.submit(read,z))
  while pending:
   if STOP or time.time()>=cfg['hard_deadline_epoch']-60:raise TimeoutError('packing original 24h deadline')
   name,b=pending.popleft().result();assert position+len(b)<cfg['packed_bytes_cap'];offsets.append(position);names.append(name);sizes.append(len(b));assert output.write(b)==len(b);position+=len(b);z=next(source,None)
   if z is not None:pending.append(executor.submit(read,z))
   if len(names)%1024==0:write(root/'status.private.json',dict(state='PACKING',images=len(names),bytes=position,updated=time.time(),started=START))
  output.flush();os.fsync(output.fileno())
 assert len(names)==len(set(names))==68358
 np.savez(root/'pack_index.private.npz',names=np.array(names),offsets=np.array(offsets,dtype=np.int64),sizes=np.array(sizes,dtype=np.int64));write(root/'PACK_READY.private.json',dict(state='READY',images=len(names),bytes=position,created=time.time(),train_images=68058,val_images=300,source_images_modified=False))
 # One focused decoded-pixel and seeded-transform check for the adapter change.
 sample=[0,len(data['train'])//2,len(data['train'])-1];plain=Images(data['train'],'imagenet',True);packed=Images(data['train'],'imagenet',True,str(root))
 for i in sample:
  seed_all(74002+i);a=plain[i][0];seed_all(74002+i);b=packed[i][0];assert torch.equal(a,b)
 result=dict(state='PASS',images=len(names),bytes=position,adapter_qualification_samples=len(sample),seeded_transform_exact_match=True,new_encoding_features=0,process_wall_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(root/'PACK_COSTS.private.json',result);write(root/'status.private.json',dict(state='READY',images=len(names),bytes=position,updated=time.time()));print(json.dumps(result),flush=True)
except BaseException:
 import traceback
 write(root/'failure.private.json',dict(error=traceback.format_exc(),images=len(names),bytes=position,process_wall_seconds=time.time()-START));np.savez(root/'partial_index.private.npz',names=np.array(names),offsets=np.array(offsets,dtype=np.int64),sizes=np.array(sizes,dtype=np.int64));raise
