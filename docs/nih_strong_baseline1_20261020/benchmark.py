"""Discarded engineering models: stable LP/FT input+optimizer throughput, no evaluation."""
import os,json,time,gc,resource
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import Model,Images,rows,seed_all,worker_seed,lr_multiplier,write
def main():
 start=time.time();cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent
 torch.set_num_threads(cfg['CPU_threads']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 data,_=rows(cfg);prior=torch.tensor(np.log(np.array(cfg['train_counts'])/68058),dtype=torch.float32,device='cuda');records=[]
 for arm in cfg['arms'][:2]:
  seed_all(cfg['seed']);full='_full_' in arm;model=Model(cfg,'imagenet').cuda();model.train();model.visual.train(full)
  for p in model.visual.parameters():p.requires_grad_(full)
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'],betas=(.9,.999))
  loader=torch.utils.data.DataLoader(Images(data['train'],'imagenet',True,cfg['packed_cache']),batch_size=128,shuffle=True,num_workers=4,worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(cfg['seed']),pin_memory=True,persistent_workers=True,multiprocessing_context='spawn',prefetch_factor=2)
  iterator=iter(loader);updates=0;exposures=0;timed_updates=0;wait_seconds=0.;compute_seconds=0.;arm_start=time.time();window_start=None
  while True:
   if time.time()>=cfg['hard_deadline_epoch']-120:raise TimeoutError('original deadline during engineering measurement')
   t=time.perf_counter();x,y,_=next(iterator);wait=time.perf_counter()-t;t=time.perf_counter();rate=cfg['lr']*lr_multiplier(updates,15960)
   for group in optimizer.param_groups:group['lr']=rate
   optimizer.zero_grad(set_to_none=True)
   for lo in range(0,len(y),32):
    xx=x[lo:lo+32].cuda(non_blocking=True);yy=y[lo:lo+len(xx)].cuda(non_blocking=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):score=model(xx);loss=F.cross_entropy(score+prior,yy)
    assert torch.isfinite(loss);(loss*(len(xx)/len(y))).backward();float(loss.detach());int((score.argmax(1)==yy).sum())
   grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);float(grad);optimizer.step();torch.cuda.synchronize();compute=time.perf_counter()-t;updates+=1;exposures+=len(y)
   if updates==5:window_start=time.time()
   elif updates>5:
    timed_updates+=1;wait_seconds+=wait;compute_seconds+=compute
    if time.time()-window_start>=60:break
  elapsed=time.time()-window_start;records.append(dict(arm=arm,measurement='engineering_end_to_end_discarded_model',timed_updates=timed_updates,window_start=window_start,window_end=time.time(),window_seconds=elapsed,seconds_per_update=elapsed/timed_updates,images_per_second=128*timed_updates/elapsed,input_wait_seconds=wait_seconds,compute_and_optimizer_seconds=compute_seconds,engineering_optimizer_updates=updates,engineering_train_image_exposures=exposures,arm_wall_seconds=time.time()-arm_start,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
  write(root/'ENGINEERING_THROUGHPUT.json',dict(state='RUNNING',records=records));print(json.dumps(records[-1]),flush=True)
  del iterator,loader,optimizer,model;gc.collect();torch.cuda.empty_cache()
 write(root/'ENGINEERING_THROUGHPUT.json',dict(state='PASS',records=records,scientific_models_or_scores_used=False,engineering_optimizer_updates=sum(r['engineering_optimizer_updates'] for r in records),engineering_train_image_exposures=sum(r['engineering_train_image_exposures'] for r in records),process_wall_seconds=time.time()-start,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))

if __name__=="__main__":main()
