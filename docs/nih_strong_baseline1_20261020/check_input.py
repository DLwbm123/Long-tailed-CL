"""Check ordered random draws, pixel equality and replay against original item path."""
import os,json,time,resource
from pathlib import Path
import numpy as np
import torch
from common import Images,rows,seed_all,write
start=time.time();cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());torch.set_num_threads(1)
data,_=rows(cfg);items=[data['train'][i] for i in np.linspace(0,68057,32,dtype=int)];records=[]
for backbone in ['imagenet','clip']:
 for train in [True,False]:
  reference=Images(items,backbone,train,cfg['packed_cache']);fast=Images(items,backbone,train,cfg['packed_cache'])
  seed_all(cfg['seed']);t=time.perf_counter();expected=[reference[i] for i in range(32)];old_seconds=time.perf_counter()-t;rng=torch.get_rng_state()
  seed_all(cfg['seed']);t=time.perf_counter();actual=fast.__getitems__(list(range(32)));fast_seconds=time.perf_counter()-t
  assert torch.equal(rng,torch.get_rng_state())
  assert all(torch.equal(a[0],b[0]) and a[1:]==b[1:] for a,b in zip(expected,actual))
  seed_all(cfg['seed']);replay=Images(items,backbone,train,cfg['packed_cache'],replay_batches=1);dummy=replay.__getitems__(list(range(32)))
  assert torch.equal(rng,torch.get_rng_state()) and all(v[0].shape==(1,) for v in dummy)
  next_fast=replay.__getitems__(list(range(32)));seed_all(cfg['seed']);[reference[i] for i in range(32)];next_ref=[reference[i] for i in range(32)]
  assert all(torch.equal(a[0],b[0]) for a,b in zip(next_fast,next_ref))
  records.append(dict(backbone=backbone,train=train,images=32,pixels_exact=True,RNG_exact=True,replay_exact=True,old_seconds=old_seconds,fast_seconds=fast_seconds))
result=dict(state='PASS',records=records,optimizer_updates=0,engineering_pixel_transforms=32*5*4,engineering_replay_headers=32*4,process_wall_seconds=time.time()-start,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
write(Path(cfg['output']).parent/'input_qualification.private.json',result);print(json.dumps(result),flush=True)
