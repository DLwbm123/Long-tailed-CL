"""Birth-frozen native visual class functions; all-history global query risk."""
import os,json,time,math,random,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import write,guard
from data import train_rows,get_cache,Tokens,OrderSampler
from folds import assign_folds
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());rootcfg=json.loads(Path(cfg['root_config']).read_text());root=Path(rootcfg['output']);out=Path(cfg['output']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def status(**kw):write(out/'STATUS.json',dict(time=time.time(),**kw))
def close_loader(dl):
 if getattr(dl,'_iterator',None) is not None:dl._iterator._shutdown_workers()
def make_model(bundle,delta):
 with torch.random.fork_rng(devices=[0]):m=NativeTail(bundle).cuda();adapt_native_tail(m)
 load_delta(m,delta);m.eval();return m

def main():
 rows,all_y=train_rows(cfg);counts=torch.bincount(all_y,minlength=len(cfg['order']));bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);records=[];steps=0;micro_steps=0;expected_total=0
 for seed in [74002,74003,74004]:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);folder=out/str(seed);folder.mkdir(exist_ok=True);assert not list(folder.glob('head*.private.pt')),'PARTIAL_DOMAIN_REQUIRES_AUDIT'
  with torch.random.fork_rng(devices=[0]):model=NativeTail(bundle).cuda()
  params=adapt_native_tail(model);model.eval();frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad};paths=[];seedsteps=0;old_folds={};experts=[]
  for stage,classes in enumerate(cfg['stages']):
   status(phase='WAITING_TRAIN_CACHE',seed=seed,stage=stage);paths.extend(get_cache(rootcfg,cfg,stage=stage));data=Tokens(paths);n=sum(len(v) for v in cfg['stages'][:stage+1]);old=n-len(classes);nc=counts[:n].cuda();N=len(data);assert torch.bincount(data.labels,minlength=n).tolist()==nc.tolist()
   folds=assign_folds(data.labels.tolist(),[rows[i]['identity_component'] for i in data.ids]);mapping=dict(zip(data.ids,folds));assert all(mapping[i]==f for i,f in old_folds.items());old_folds=mapping
   indices=[[i for i,f in enumerate(folds) if f==b] for b in [0,1]];new_indices=[[i for i in ii if int(data.labels[i])>=old] for ii in indices];fold_counts=[torch.bincount(data.labels[ii],minlength=n).cuda() for ii in indices];assert all(bool((c>0).all()) for c in fold_counts)
   sampler=OrderSampler(N);dl=DataLoader(data,batch_size=256,sampler=sampler,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=1);old_logits=torch.empty(N,old)
   # Immutable birth experts score newly arrived rows only after their arrival.
   with torch.no_grad():
    for e in experts:
     frozen_model=make_model(bundle,e['delta']);sampler.order=list(range(N));pp=e['fold_prototypes'];assert e['end']<=old
     for tokens,base,yy,ix in dl:
      for tx,jj in zip(tokens.split(64),ix.split(64)):
       zz=frozen_model(tx.cuda(non_blocking=True));ff=torch.tensor([folds[i] for i in jj.tolist()],device='cuda');pr=pp.to('cuda')[1-ff];scores=20*torch.einsum('bd,bcd->bc',zz,pr);old_logits[jj,e['start']:e['end']]=scores.cpu()
     del frozen_model
   assert torch.isfinite(old_logits).all();old_snapshots=[{k:(v.clone() if torch.is_tensor(v) else {kk:vv.clone() for kk,vv in v.items()} if k=='delta' else v) for k,v in e.items()} for e in experts]
   opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10);pg=torch.Generator(device='cuda').manual_seed(seed+10000*stage);curves=[];stage_steps=0;stage_micro=0;gradmax=0.;training_query_rows=0;support_refreshes=0;support_training_rows=0
   @torch.no_grad()
   def encode_new(ii):
    guard(cfg);sampler.order=ii;sums=torch.zeros(len(classes),512,device='cuda',dtype=torch.double);cc=torch.zeros(len(classes),device='cuda',dtype=torch.long)
    for tokens,base,yy,ix in dl:
     for tx,by in zip(tokens.split(64),yy.split(64)):
      by=by.cuda()-old;assert bool((by>=0).all());zz=model(tx.cuda(non_blocking=True));sums.index_add_(0,by,zz.double());cc+=torch.bincount(by,minlength=len(classes))
    assert bool((cc>0).all());return F.normalize((sums/cc[:,None]).float(),dim=1).detach(),cc
   @torch.no_grad()
   def observe(epoch,obj):
    rng=torch.get_rng_state().clone();cu=torch.cuda.get_rng_state_all();cc=torch.zeros(n,device='cuda',dtype=torch.long);den=torch.zeros_like(cc);nll=torch.zeros(n,device='cuda',dtype=torch.double)
    for qfold in [0,1]:
     proto,sc=encode_new(new_indices[1-qfold]);assert torch.equal(sc,fold_counts[1-qfold][old:]);sampler.order=indices[qfold]
     for tokens,base,yy,ix in dl:
      for tx,by,jj in zip(tokens.split(64),yy.split(64),ix.split(64)):
       by=by.cuda();z=torch.cat([old_logits[jj].cuda(),20*model(tx.cuda(non_blocking=True))@proto.T],dim=1);assert torch.isfinite(z).all();pr=z.argmax(1);cc+=torch.bincount(by[pr==by],minlength=n);den+=torch.bincount(by,minlength=n);nll.scatter_add_(0,by,F.cross_entropy(z,by,reduction='none').double())
    assert torch.equal(den,nc);a=cc.double()/nc;curves.append(dict(epoch=epoch,objective_mean=obj,optimizer_steps=stage_steps,micro_backwards=stage_micro,query_counts=nc.tolist(),crossfit_query_correct=cc.tolist(),crossfit_query_macro_NLL=float((nll/nc).mean()),crossfit_query_macro_BA=float(a.mean()),crossfit_query_tail_BA=float(a[nc<250].mean()),LoRA_parameter_norm=float(torch.sqrt(sum(p.square().sum() for p in params))),lr=opt.param_groups[0]['lr'],is_independent_holdout=False));write(folder/f'curves{stage}.json',curves);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cu)
   observe(0,None)
   for epoch in range(10):
    total=0.
    for qfold in [0,1]:
     guard(cfg);opt.zero_grad(set_to_none=True);proto,sc=encode_new(new_indices[1-qfold]);assert not proto.requires_grad and torch.equal(sc,fold_counts[1-qfold][old:]);support_refreshes+=1;support_training_rows+=int(sc.sum());ii=indices[qfold];sampler.order=[ii[i] for i in torch.randperm(len(ii),device='cuda',generator=pg).cpu().tolist()]
     for bi,(tokens,base,yy,ix) in enumerate(dl):
      if bi%32==0:guard(cfg);status(phase='BIRTH_FUNCTION_TRAIN',seed=seed,stage=stage,epoch=epoch,query_fold=qfold,batch=bi,optimizer_steps=seedsteps,total_optimizer_steps=steps,micro_backwards=micro_steps,seen=n,all_real_rows=N,frozen_birth_experts=len(experts))
      opt.zero_grad(set_to_none=True);B=len(yy)
      for tx,bx,by,jj in zip(tokens.split(64),base.split(64),yy.split(64),ix.split(64)):
       by=by.cuda();zz=model(tx.cuda(non_blocking=True));logits=torch.cat([old_logits[jj].cuda(),20*zz@proto.T],dim=1);w=N/(n*nc[by].float());loss=(w*(F.cross_entropy(logits,by,reduction='none')+.01*(zz-bx.cuda(non_blocking=True)).square().sum(1))).sum()/B;assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())*B;stage_micro+=1;micro_steps+=1;training_query_rows+=len(by)
      assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);gradmax=max(gradmax,max(float(p.grad.norm()) for p in params));opt.step();steps+=1;seedsteps+=1;stage_steps+=1;assert torch.cuda.max_memory_reserved()<12*2**30
    sched.step();observe(epoch+1,total/N)
   expected=10*sum(math.ceil(len(ii)/256) for ii in indices);expected_micro=10*sum(math.ceil(len(ii)/64) for ii in indices);expected_total+=expected;assert stage_steps==expected and stage_micro==expected_micro and training_query_rows==10*N and support_training_rows==10*int(nc[old:].sum()) and support_refreshes==20 and gradmax>0;assert all(torch.equal(p.detach().cpu(),frozen[k]) for k,p in model.named_parameters() if not p.requires_grad)
   proto,pc=encode_new([i for i in range(N) if int(data.labels[i])>=old]);fold_proto=torch.stack([encode_new(ii)[0].cpu() for ii in new_indices]);experts.append(dict(start=old,end=n,delta=delta_state(model),prototypes=proto.cpu(),fold_prototypes=fold_proto))
   for e,b in zip(experts[:-1],old_snapshots):
    assert torch.equal(e['prototypes'],b['prototypes']) and torch.equal(e['fold_prototypes'],b['fold_prototypes']) and all(torch.equal(e['delta'][k],v) for k,v in b['delta'].items())
   sums=torch.zeros(n,512,dtype=torch.double);sums.index_add_(0,data.labels,data.base.double());raw_proto=F.normalize((sums/counts[:n,None]).float(),dim=1);h=dict(experts=experts,raw_prototypes=raw_proto,train_counts=nc.cpu(),seed=seed,stage=stage,seen=n);torch.save(h,folder/f'head{stage}.private.pt');tx=data[0][0].unsqueeze(0).cuda()
   with torch.no_grad():
    other=make_model(bundle,experts[-1]['delta']);assert torch.equal(model(tx),other(tx));del other
   records.append(dict(seed=seed,stage=stage,seen=n,train_counts=nc.tolist(),all_real_rows=N,fold_rows=[len(ii) for ii in indices],fold_class_counts=[x.tolist() for x in fold_counts],optimizer_steps=stage_steps,micro_backwards=stage_micro,epochs=10,curve_records=11,LoRA_active_parameters=49152,retained_LoRA_parameters=49152*len(experts),class_specific_parameters=0,all_original_tail_parameters_exact=True,strict_restore_predictions_exact=True,old_fold_assignment_exact=True,metadata_groups_disjoint=True,support_detached=True,old_birth_functions_exact=True,new_class_support_reencoded_for_head=True,VAL_reads=0,training_support_refreshes=support_refreshes,training_support_rows=support_training_rows,training_query_rows=training_query_rows,max_gradient_norm=gradmax));write(out/'TRAIN_RECORDS.json',records);close_loader(dl);del dl,data,opt,sched,tx,proto,old_logits
  write(folder/'TRAIN_SEAL.json',dict(stages=len(cfg['stages']),optimizer_steps=seedsteps,VAL_reads=0));del model,params,frozen,experts;torch.cuda.empty_cache()
 assert steps==expected_total==cfg['expected_steps'];write(out/'ALL_HEADS_SEALED.json',dict(records=len(records),optimizer_steps=steps,micro_backwards=micro_steps,VAL_reads=0));status(phase='WAITING_VAL_CACHE');valpaths=get_cache(rootcfg,cfg,val=True);vdata=Tokens(valpaths);dl=DataLoader(vdata,batch_size=64,shuffle=False,num_workers=2,pin_memory=True);result=[]
 with torch.no_grad():
  for seed in [74002,74003,74004]:
   for stage in range(len(cfg['stages'])):
    guard(cfg);h=torch.load(out/str(seed)/f'head{stage}.private.pt',map_location='cpu',weights_only=False);models=[make_model(bundle,e['delta']) for e in h['experts']];n=h['seen'];scores=[];baseline=[];ys=[];vids=[]
    for tokens,base,yy,ix in dl:
     tokens=tokens.cuda(non_blocking=True);z=torch.cat([20*m(tokens)@e['prototypes'].cuda().T for m,e in zip(models,h['experts'])],dim=1);ff=20*base.cuda()@h['raw_prototypes'].cuda().T;mask=yy<n;scores.append(z.cpu()[mask]);baseline.append(ff.cpu()[mask]);ys.append(yy[mask]);vids.extend(vdata.ids[i] for i in ix[mask].tolist())
    yy=torch.cat(ys);ss=dict(birth_functions=torch.cat(scores),frozen_support=torch.cat(baseline))
    if stage>0:
     prior=torch.load(out/'responses'/str(seed)/f'responses{stage-1}.private.pt',map_location='cpu',weights_only=False);old_n=prior['scores']['birth_functions'].shape[1];mask=yy<old_n;assert [i for i,y in zip(vids,yy.tolist()) if y<old_n]==prior['ids'];assert torch.equal(ss['birth_functions'][mask,:old_n],prior['scores']['birth_functions']),'OLD_BIRTH_SCORE_CHANGED'
    p=out/'responses'/str(seed);p.mkdir(parents=True,exist_ok=True);torch.save(dict(scores=ss,labels=yy,ids=vids),p/f'responses{stage}.private.pt');metrics={}
    for arm,z in ss.items():
     assert torch.isfinite(z).all();pr=z.argmax(1);nn=torch.bincount(yy,minlength=n);cc=torch.bincount(yy[pr==yy],minlength=n);ba=cc.double()/nn;metrics[arm]=dict(correct=cc.tolist(),counts=nn.tolist(),BA=float(ba.mean()),tail_BA=float(ba[counts[:n]<250].mean()),all_finite=True)
    result.append(dict(seed=seed,stage=stage,seen=n,old=sum(len(v) for v in cfg['stages'][:stage]),metrics=metrics));del models,h
 close_loader(dl);write(out/'aggregate.json',dict(dataset=cfg['dataset'],results=result,train_counts=counts.tolist(),optimizer_steps=steps,micro_backwards=micro_steps,trained_stages=len(records),readout_arms=2,no_TEST=True,independent_confirmation=False));status(phase='COMPLETE');removed=0;files=[]
 for path in paths+valpaths:
  meta=json.loads((path/'META.json').read_text())
  for ch in meta['chunks']:
   for name in [ch['file'],ch['base']]:
    p=path/name;assert p.parent==path and not p.is_symlink()
    if p.exists():removed+=p.stat().st_size;files.append(str(p.relative_to(root)));p.unlink()
 write(out/'REGENERABLE_CACHE_CLEANUP.json',dict(removed_bytes=removed,files=files,all3seeds_TRAIN_and_VAL_complete=True,scientific_artifacts_preserved=True))
if __name__=='__main__':
 try:main()
 except BaseException:(out/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED');raise
