"""Native last-two-block LoRA, complete arrived FP32 token rehearsal."""
import os,json,time,math,random,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import write,guard
from data import train_rows,get_cache,Tokens,OrderSampler
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());rootcfg=json.loads(Path(cfg['root_config']).read_text());root=Path(rootcfg['output']);out=Path(cfg['output']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def status(**kw):write(out/'STATUS.json',dict(time=time.time(),**kw))
def text_for(stage,previous):
 b=torch.load(Path(cfg['train_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);n=sum(len(v) for v in cfg['stages'][:stage+1]);t=F.normalize(b['text'],dim=1) if cfg['dataset']=='NIH' else b['text'];t=t[[b['prompts'].index(cfg['leaf_template'].format(v)) for v in cfg['ordered_names'][:n]]].cuda();assert torch.allclose(t.norm(dim=1),torch.ones(n,device='cuda'),atol=2e-6)
 if previous is not None:assert torch.allclose(t[:len(previous)],previous,atol=2e-6,rtol=0);t=torch.cat([previous,t[len(previous):]])
 return t

def close_loader(dl):
 if getattr(dl,'_iterator',None) is not None:dl._iterator._shutdown_workers()

def main():
 rows,all_y=train_rows(cfg);counts=torch.bincount(all_y,minlength=len(cfg['order']));bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);records=[];steps=0;micro_steps=0
 for seed in [74002,74003,74004]:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);folder=out/str(seed);folder.mkdir(exist_ok=True);assert not list(folder.glob('head*.private.pt')),'PARTIAL_DOMAIN_REQUIRES_AUDIT'
  with torch.random.fork_rng(devices=[0]):model=NativeTail(bundle).cuda()
  params=adapt_native_tail(model);model.eval();frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad};text=None;paths=[];seedsteps=0
  for stage,classes in enumerate(cfg['stages']):
   status(phase='WAITING_TRAIN_CACHE',seed=seed,stage=stage);paths.append(get_cache(rootcfg,cfg,stage=stage));data=Tokens(paths);n=sum(len(v) for v in cfg['stages'][:stage+1]);nc=counts[:n].cuda();assert torch.bincount(data.labels,minlength=n).tolist()==nc.tolist();text=text_for(stage,text);fixedtext=text.clone();N=len(data);sampler=OrderSampler(N);dl=DataLoader(data,batch_size=256,sampler=sampler,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=1);opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10);pg=torch.Generator(device='cuda').manual_seed(seed+10000*stage);curves=[];stage_steps=0;stage_micro=0;gradmax=0.
   @torch.no_grad()
   def observe(epoch,obj):
    guard(cfg);rng=torch.get_rng_state().clone();cu=torch.cuda.get_rng_state_all();sampler.order=list(range(N));cc=torch.zeros(n,dtype=torch.long,device='cuda');nll=torch.zeros(n,dtype=torch.double,device='cuda')
    for tokens,base,yy,indices in dl:
     for tx,by in zip(tokens.split(64),yy.split(64)):
      by=by.cuda();z=20*model(tx.cuda(non_blocking=True))@text.T;assert torch.isfinite(z).all();p=z.argmax(1);cc+=torch.bincount(by[p==by],minlength=n);nll.scatter_add_(0,by,F.cross_entropy(z,by,reduction='none').double())
    a=cc.double()/nc;curves.append(dict(epoch=epoch,objective_mean=obj,optimizer_steps=stage_steps,micro_backwards=stage_micro,full_TRAIN_counts=nc.tolist(),full_TRAIN_correct=cc.tolist(),macro_decision_NLL=float((nll/nc).mean()),macro_BA=float(a.mean()),tail_BA=float(a[nc<250].mean()),LoRA_parameter_norm=float(torch.sqrt(sum(p.square().sum() for p in params))),lr=opt.param_groups[0]['lr']));write(folder/f'curves{stage}.json',curves)
    # DataLoader CPU base seeds may consume RNG; restore observation-only state.
    torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cu)
   observe(0,None)
   for epoch in range(10):
    guard(cfg);sampler.order=torch.randperm(N,device='cuda',generator=pg).cpu().tolist();total=0.
    for bi,(tokens,base,yy,indices) in enumerate(dl):
     if bi%32==0:guard(cfg);status(phase='VISION_TAIL_TRAIN',seed=seed,stage=stage,epoch=epoch,batch=bi,optimizer_steps=seedsteps,micro_backwards=micro_steps,seen=n,all_real_rows=N)
     opt.zero_grad(set_to_none=True);B=len(yy)
     for tx,bx,by in zip(tokens.split(64),base.split(64),yy.split(64)):
      by=by.cuda();x=model(tx.cuda(non_blocking=True));w=N/(n*nc[by].float());loss=(w*(F.cross_entropy(20*x@text.T,by,reduction='none')+.01*(x-bx.cuda(non_blocking=True)).square().sum(1))).sum()/B;assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())*B;stage_micro+=1;micro_steps+=1
     assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);gradmax=max(gradmax,max(float(p.grad.norm()) for p in params));opt.step();steps+=1;seedsteps+=1;stage_steps+=1;assert torch.cuda.max_memory_reserved()<12*2**30
    sched.step();observe(epoch+1,total/N)
   assert stage_steps==10*math.ceil(N/256) and stage_micro==10*math.ceil(N/64) and gradmax>0;assert torch.equal(text,fixedtext);assert all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters() if not p.requires_grad)
   h=dict(delta=delta_state(model),text=text.cpu(),seed=seed,stage=stage,seen=n);torch.save(h,folder/f'head{stage}.private.pt');tx=data[0][0].unsqueeze(0).cuda()
   with torch.no_grad(),torch.random.fork_rng(devices=[0]):
    other=NativeTail(bundle).cuda();adapt_native_tail(other);other.eval();back=torch.load(folder/f'head{stage}.private.pt',map_location='cpu',weights_only=False);load_delta(other,back['delta']);assert torch.equal(model(tx),other(tx));del other,back
   records.append(dict(seed=seed,stage=stage,seen=n,train_counts=nc.tolist(),all_real_rows=N,optimizer_steps=stage_steps,micro_backwards=stage_micro,epochs=10,curve_records=11,LoRA_trainable_parameters=49152,class_specific_parameters=0,all_original_tail_parameters_exact=True,strict_restore_predictions_exact=True,fixed_text_exact=True,class_total_risk_mass=[1.]*n,VAL_reads=0,max_gradient_norm=gradmax));write(out/'TRAIN_RECORDS.json',records);close_loader(dl);del dl,data,opt,sched,tx
  write(folder/'TRAIN_SEAL.json',dict(stages=len(cfg['stages']),optimizer_steps=seedsteps,VAL_reads=0));del model,text,params,frozen;torch.cuda.empty_cache()
 expected=3*sum(10*math.ceil(sum(int(counts[c]) for c in sum(cfg['stages'][:i+1],[]))/256) for i in range(len(cfg['stages'])));assert steps==expected and len(records)==3*len(cfg['stages']);write(out/'ALL_HEADS_SEALED.json',dict(records=len(records),optimizer_steps=steps,micro_backwards=micro_steps,VAL_reads=0));status(phase='WAITING_VAL_CACHE')
 valpath=get_cache(rootcfg,cfg,val=True);vdata=Tokens([valpath]);dl=DataLoader(vdata,batch_size=64,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=1);result=[]
 with torch.no_grad():
  for seed in [74002,74003,74004]:
   for stage in range(len(cfg['stages'])):
    guard(cfg);h=torch.load(out/str(seed)/f'head{stage}.private.pt',map_location='cpu',weights_only=False);model=NativeTail(bundle).cuda();adapt_native_tail(model);model.eval();load_delta(model,h['delta']);text=h['text'].cuda();n=h['seen'];scores=[];baseline=[];ys=[];vids=[]
    for tokens,base,yy,indices in dl:
     z=20*model(tokens.cuda(non_blocking=True))@text.T;f=20*base.cuda(non_blocking=True)@text.T;assert torch.isfinite(z).all() and torch.isfinite(f).all();mask=yy<n;scores.append(z.cpu()[mask]);baseline.append(f.cpu()[mask]);ys.append(yy[mask]);vids.extend(vdata.ids[i] for i in indices[mask].tolist())
    yy=torch.cat(ys);ss=dict(vision_lora=torch.cat(scores),frozen_visual=torch.cat(baseline));p=out/'responses'/str(seed);p.mkdir(parents=True,exist_ok=True);torch.save(dict(scores=ss,labels=yy,ids=vids),p/f'responses{stage}.private.pt');metrics={}
    for arm,z in ss.items():
     pred=z.argmax(1);nn=torch.bincount(yy,minlength=n);cc=torch.bincount(yy[pred==yy],minlength=n);ba=cc.double()/nn;metrics[arm]=dict(correct=cc.tolist(),counts=nn.tolist(),BA=float(ba.mean()),tail_BA=float(ba[counts[:n]<250].mean()),all_finite=True)
    result.append(dict(seed=seed,stage=stage,seen=n,old=sum(len(v) for v in cfg['stages'][:stage]),metrics=metrics));del model,h
 close_loader(dl);write(out/'aggregate.json',dict(dataset=cfg['dataset'],results=result,train_counts=counts.tolist(),optimizer_steps=steps,micro_backwards=micro_steps,trained_stages=len(records),readout_arms=2,no_TEST=True,independent_confirmation=False));status(phase='COMPLETE')
 # Regenerable tokens only after all scientific outputs and domain validation complete.
 removed=0;files=[]
 for path in paths+[valpath]:
  meta=json.loads((path/'META.json').read_text())
  for c in meta['chunks']:
   for name in [c['file'],c['base']]:
    p=path/name;assert p.parent==path and not p.is_symlink()
    if p.exists():removed+=p.stat().st_size;files.append(str(p.relative_to(root)));p.unlink()
 write(out/'REGENERABLE_CACHE_CLEANUP.json',dict(removed_bytes=removed,files=files,all3seeds_TRAIN_and_VAL_complete=True,scientific_artifacts_preserved=True))
if __name__=='__main__':
 try:main()
 except BaseException:(out/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED');raise
