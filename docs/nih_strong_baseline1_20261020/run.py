"""Four paired, sufficiently budgeted static baselines on read-only NIH train/val."""
import time
START=time.time()
import os,json,math,random,signal,resource,gc,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import write,seed_all,rows,Images,Model,metrics,lr_multiplier,worker_seed
CFG=json.loads(Path(os.environ['RUN_CONFIG']).read_text());STOP=False
COST=dict(attempts=2,prior_attempt_optimizer_updates=CFG['prior_attempt_optimizer_updates'],prior_attempt_train_image_exposures=CFG['prior_attempt_train_image_exposures'],optimizer_updates=0,train_image_exposures=0,val_image_exposures=0,arms_completed=0,engineering_image_exposures=CFG['engineering_image_exposures'])


def interrupted(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,interrupted)


def atomic_save(p,value):
 tmp=p.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(p)


def evaluate(model,loader,labels,groups,counts):
 model.eval();allscore=np.empty((len(labels),20),dtype=np.float32)
 with torch.no_grad():
  for x,y,ids in loader:
   with torch.autocast('cuda',dtype=torch.bfloat16):s=model(x.cuda(non_blocking=True))
   assert torch.isfinite(s).all();allscore[ids.numpy()]=s.float().cpu().numpy();COST['val_image_exposures']+=len(x)
 return allscore,metrics(allscore,labels,groups,counts,CFG['legacy_tail'])


def main():
 torch.set_num_threads(CFG['CPU_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);out=Path(CFG['output']);out.mkdir(exist_ok=False);write(out/'config.private.json',CFG);dataset,audit=rows(CFG);write(out/'DATA_AUDIT.json',audit);counts=audit['train_counts'];prior=torch.tensor(np.log(np.asarray(counts)/sum(counts)),dtype=torch.float32,device='cuda');labels=np.array([x[1] for x in dataset['val']]);groups=np.array([x[2] for x in dataset['val']]);history=[];summaries=[];initial_heads={};steps_epoch=math.ceil(len(dataset['train'])/CFG['batch_size']);total=steps_epoch*CFG['epochs'];laststatus=[0.]
 def status(phase,force=False,**kw):
  if force or time.time()-laststatus[0]>=10:
   peaks=dict(RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved());write(out/'resource_peaks.private.json',peaks);write(out/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),started=START,costs=COST,**peaks,**kw));laststatus[0]=time.time()
 for arm in CFG['arms']:
  if STOP:raise TimeoutError('guardian termination before next arm')
  seed_all(CFG['seed']);backbone=arm.split('_')[0];full='_full_' in arm;adir=out/arm;adir.mkdir(exist_ok=False);status('load_model',True,arm=arm);model=Model(CFG,backbone).cuda();head0=model.weight.detach().cpu()
  if backbone in initial_heads:assert torch.equal(initial_heads[backbone],head0)
  else:initial_heads[backbone]=head0.clone()
  for p in model.visual.parameters():p.requires_grad_(full)
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CFG['lr'],weight_decay=CFG['weight_decay'],betas=(.9,.999));generator=torch.Generator().manual_seed(CFG['seed']);trainloader=torch.utils.data.DataLoader(Images(dataset['train'],backbone,True,CFG['packed_cache']),batch_size=CFG['batch_size'],shuffle=True,num_workers=CFG['workers'],worker_init_fn=worker_seed,generator=generator,pin_memory=True,persistent_workers=True,multiprocessing_context='spawn',prefetch_factor=2);valloader=torch.utils.data.DataLoader(Images(dataset['val'],backbone,False,CFG['packed_cache']),batch_size=CFG['micro_batch_size'],shuffle=False,num_workers=0,pin_memory=True);best=-1.;best_epoch=None;update=0;arm_start=time.time();local=[]
  def checkpoint(epoch,complete_epoch,tag='last'):
   atomic_save(adir/(tag+'.private.pt'),dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,complete_epoch=complete_epoch,update=update,costs=dict(COST),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=generator.get_state(),arm=arm,backbone=backbone,started=arm_start,original_start_epoch=CFG['original_start_epoch']))
  for epoch in range(1,CFG['epochs']+1):
   model.train();model.visual.train(full);loss_sum=0.;correct=0;images=0;start=time.time();maxgrad=0.
   for step,(x,y,_) in enumerate(trainloader):
    if STOP:checkpoint(epoch,False);raise TimeoutError('hard deadline/storage termination; partial epoch retained, not complete')
    rate=CFG['lr']*lr_multiplier(update,total)
    for group in optimizer.param_groups:group['lr']=rate
    optimizer.zero_grad(set_to_none=True);batch=len(y)
    for lo in range(0,batch,CFG['micro_batch_size']):
     xx=x[lo:lo+CFG['micro_batch_size']].cuda(non_blocking=True);yy=y[lo:lo+len(xx)].cuda(non_blocking=True)
     with torch.autocast('cuda',dtype=torch.bfloat16):score=model(xx);loss=F.cross_entropy(score+prior,yy)
     assert torch.isfinite(loss);(loss*(len(xx)/batch)).backward();loss_sum+=float(loss.detach())*len(xx);correct+=int((score.argmax(1)==yy).sum());images+=len(xx);COST['train_image_exposures']+=len(xx)
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);maxgrad=max(maxgrad,float(grad));optimizer.step();update+=1;COST['optimizer_updates']+=1;status('train_epoch',arm=arm,epoch=epoch,step=step+1,steps_epoch=steps_epoch,lr=rate,train_loss_so_far=loss_sum/images)
   assert images==68058;checkpoint(epoch,True);score,met=evaluate(model,valloader,labels,groups,counts);np.savez(adir/f'val_epoch_{epoch}.private.npz',scores=score,labels=labels);record=dict(arm=arm,epoch=epoch,updates=update,train_images=images,train_LA_loss=loss_sum/images,train_raw_accuracy=correct/images,max_preclip_gradient_norm=maxgrad,lr=rate,metrics=met,epoch_seconds=time.time()-start);local.append(record);history.append(record);write(out/'LEARNING_CURVES.json',history);write(adir/'history.json',local)
   if met['BA']>best:
    best=met['BA'];best_epoch=epoch;atomic_save(adir/'best.private.pt',dict(model=model.state_dict(),epoch=epoch,arm=arm,backbone=backbone))
   if epoch==10:atomic_save(adir/'epoch10.private.pt',dict(model=model.state_dict(),epoch=epoch,arm=arm,backbone=backbone))
   status('epoch_complete',True,arm=arm,epoch=epoch,best_epoch=best_epoch,best_BA=best);print(json.dumps(record),flush=True)
  assert update==total;COST['arms_completed']+=1;summary=dict(arm=arm,backbone=backbone,full_tuning=full,epochs=CFG['epochs'],updates=update,best_epoch=best_epoch,best=local[best_epoch-1]['metrics'],epoch10=local[9]['metrics'],final=local[-1]['metrics'],final_train_loss=local[-1]['train_LA_loss'],last5_train_loss=[r['train_LA_loss'] for r in local[-5:]],wall_seconds=time.time()-arm_start);summaries.append(summary);write(out/'RESULTS.json',summaries);write(adir/'COMPLETE.json',summary);del optimizer,model,trainloader,valloader;gc.collect();torch.cuda.empty_cache()
 assert COST['arms_completed']==4;status('final_resource_check',True);COST.update(process_wall_seconds=time.time()-START,cumulative_recorded_seconds=CFG['prior_recorded_seconds']+CFG['engineering_seconds']+CFG['prior_attempt_seconds']+json.loads((Path(CFG['packed_cache'])/'PACK_COSTS.private.json').read_text())['process_wall_seconds']+time.time()-START,cumulative_optimizer_updates=CFG['prior_attempt_optimizer_updates']+COST['optimizer_updates'],cumulative_train_image_exposures=CFG['prior_attempt_train_image_exposures']+COST['train_image_exposures'],historical_unrecorded_overhead_unknown=True);write(out/'costs.json',COST);write(out/'FIT_SEAL.json',dict(completed_at=time.time(),arms=4,validation_checkpoint_selection_predeclared=True,test_access=False));write(out/'status.private.json',dict(state='COMPLETE',costs=COST,finished=time.time(),independent_verification='PENDING'));print('COMPLETE',flush=True)


if __name__=='__main__':
 try:main()
 except BaseException:
  out=Path(CFG['output']);out.mkdir(exist_ok=True);peaks=dict(RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0);write(out/'failure.private.json',dict(error=traceback.format_exc(),costs=COST,process_wall_seconds=time.time()-START,**peaks));write(out/'status.private.json',dict(state='FAILED',updated=time.time(),costs=COST,**peaks));raise
