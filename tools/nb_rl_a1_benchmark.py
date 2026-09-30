import os,sys,time,copy,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(os.environ['N78_ROOT'])/'source'))
import run_nb_rl_a1 as r
import torch,numpy as np
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=4):
 run=r.Run()
 try:
  began=time.monotonic();model=r.Model(1993);torch.cuda.synchronize();init=time.monotonic()-began
  model.optimizer(3);model.task=1;model.epoch=1
  stage=r.load(r.PRIVATE/'p0_stage.pt.gz');model.bank=stage['bank'];W=torch.tensor(stage['W_start'],device='cuda',dtype=torch.float32)
  began=time.monotonic();anchor=copy.deepcopy(model.net).requires_grad_(False).eval();torch.cuda.synchronize();anchor_sec=time.monotonic()-began
  ds=r.BatchDataset(run,1993,1,epoch=1,train=True);counts=np.bincount([x['target'] for x in ds.rows],minlength=8)
  weights=np.zeros(8);weights[:2]=len(ds)/(2*counts[:2]);weights=torch.tensor(weights,device='cuda',dtype=torch.float32)
  iterator=iter(r.loader(ds,True));times=[];hashes=[];steps=[]
  for i in range(13):
   began=time.monotonic();_,x,y=next(iterator);hstart=time.monotonic();hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest();hashes.append(time.monotonic()-hstart)
   x=x.cuda();y=y.cuda();torch.cuda.synchronize();s=time.monotonic();r.step(run,model,anchor,x,y,W,weights,'R',2);torch.cuda.synchronize();steps.append(time.monotonic()-s);times.append(time.monotonic()-began)
  gen=np.random.default_rng(781);z=gen.normal(size=(len(ds),1536));z/=np.linalg.norm(z,axis=1)[:,None];labels=np.array([x['target'] for x in ds.rows]);ids=[x['sample_id'] for x in ds.rows]
  began=time.monotonic();bank=r.append(r.empty(1536),z,labels,range(2),1);r.ridge(bank);stat=time.monotonic()-began
  began=time.monotonic();a,b,v,_=r.translation(z,z,labels);bank=r.append(r.transport(r.empty(1536),a,b,v,1),z,labels,range(2),1);r.ridge(bank);endstat=time.monotonic()-began
  began=time.monotonic();state=model.snapshot();state.update(method='P0_ONLY',W_start=stage['W_start'],before=z,labels=labels,ids=ids,anchor_delta=model.delta(),prior_bank=r.empty(1536));snap=time.monotonic()-began
  timings={}
  for compressed in (True,False):
   path=r.PRIVATE/('p0_full_rolling.pt.gz' if compressed else 'p0_full_rolling.pt');began=time.monotonic()
   if compressed:r.dump(path,state)
   else:
    tmp=path.with_suffix('.part');torch.save(state,tmp,pickle_protocol=4);tmp.replace(path)
   duration=time.monotonic()-began;began=time.monotonic()
   loaded=r.load(path) if compressed else torch.load(path,map_location='cpu',weights_only=False)
   assert r.tensors_equal(state,loaded)
   timings['gzip' if compressed else 'native']=dict(save_seconds=duration,load_and_equal_seconds=time.monotonic()-began,bytes=path.stat().st_size)
   del loaded
  r.save('P0_END_TO_END.json',dict(status='PASS',init_seconds=init,anchor_copy_seconds=anchor_sec,end_to_end_per_batch=float(np.median(times[1:])),step_only=float(np.median(steps[1:])),batch_overhead_seconds=float(np.median(np.array(times[1:])-steps[1:])),hash_seconds=float(np.median(hashes[1:])),first_loader_batch_seconds=times[0],start_statistics_seconds=stat,end_statistics_seconds=endstat,synthetic_shape=list(z.shape),snapshot_seconds=snap,rolling=timings,engineering_steps=run.counts['engineering_steps'],no_validation_images=True))
  print((r.PUB/'P0_END_TO_END.json').read_text(),flush=True)
 finally:run.resources(False)
