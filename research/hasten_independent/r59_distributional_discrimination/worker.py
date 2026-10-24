"""Full leave-one-out, globally balanced class-distribution discrimination."""
import os,json,csv,time,math,traceback,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
CFG=json.loads(Path(os.environ['RUN_CONFIG']).read_text());OUT=Path(CFG['output']);OUT.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def write(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def status(**v):write(OUT/'STATUS.json',dict(time=time.time(),**v))
def guard():
 assert time.time()<json.loads(Path(CFG['clock']).read_text())['deadline_unix']-35,'ORIGINAL_DEADLINE'
 assert torch.cuda.max_memory_reserved()<8*2**30,'GPU_BUDGET'
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
class Metric(nn.Module):
 def __init__(self):super().__init__();self.a=nn.Parameter(torch.zeros(512))
 def weights(self):
  p=F.softmax(self.a,dim=0).clamp_min(1e-8);return 512*p/p.sum()
 def forward(self,x):return F.normalize(x*self.weights().sqrt(),dim=1)
 def regularizer(self):return .01*self.a.square().mean()

def distribution_logits(q,keys,counts,query_ids=None,query_y=None):
 s=20*q@keys.T
 if query_ids is not None:s[torch.arange(len(q),device=s.device),query_ids]=-torch.inf
 scores=[];start=0
 for c,n in enumerate(counts.tolist()):
  den=n if query_ids is None else n-(query_y==c).to(s.dtype)
  scores.append(torch.logsumexp(s[:,start:start+n],dim=1)-torch.as_tensor(den,device=s.device).log());start+=n
 assert start==len(keys)
 return torch.stack(scores,dim=1)

def metrics(z,y,n,counts):
 assert torch.isfinite(z).all();pred=z.argmax(1);nnn=torch.bincount(y,minlength=n);cc=torch.bincount(y[pred==y],minlength=n);v=cc.double()/nnn;tail=[i for i in range(n) if counts[i]<250]
 return dict(correct=cc.tolist(),counts=nnn.tolist(),BA=float(v.mean()),tail_BA=float(v[tail].mean()),all_finite=True)

def main():
 ds=CFG['dataset'];stages=CFG['stages'];order=CFG['order'];C=len(order)
 def rows(split):
  with open(Path(CFG['manifest'])/(split+'.csv')) as f:r=list(csv.DictReader(f))
  assert all(v['split']==split for v in r);return r
 tr=rows('train');labels=torch.tensor([order.index(int(v['original_label'])) for v in tr]);counts=torch.bincount(labels,minlength=C);assert counts.tolist()==CFG['train_counts_ordered'] and int(counts.min())>=2
 def arrived(stage):
  guard();ids=[i for i,v in enumerate(labels.tolist()) if v in stages[stage]];dst=OUT/'memory'/f'block{stage}.private.pt'
  if dst.exists():a=torch.load(dst,map_location='cpu',weights_only=False)
  else:
   raw=torch.load(Path(CFG['train_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);assert raw['ids']==ids;x=F.normalize(raw['x'],dim=1) if ds=='NIH' else raw['x'];a=dict(ids=ids,x=x,y=labels[ids]);dst.parent.mkdir(exist_ok=True);torch.save(a,dst)
  assert a['ids']==ids and torch.equal(a['y'],labels[ids]);assert a['x'].shape==(len(ids),512) and torch.isfinite(a['x']).all() and torch.allclose(a['x'].norm(dim=1),torch.ones(len(ids)),atol=2e-6)
  return a['x'],a['y']
 records=[];newsteps=0;expected=3*sum(10*math.ceil(sum(int(counts[c]) for c in sum(stages[:i+1],[]))/256) for i in range(len(stages)))
 for ss in [74002,74003,74004]:
  seed(ss);folder=OUT/str(ss);folder.mkdir(exist_ok=True);assert not (folder/'TRAIN_SEAL.json').exists(),'DO_NOT_RETRAIN_COMPLETE_SEED';model=Metric().cuda();x=None;y=None;seedsteps=0
  for stage,classes in enumerate(stages):
   xx,yy=arrived(stage);x=xx.cuda() if x is None else torch.cat([x,xx.cuda()]);y=yy.cuda() if y is None else torch.cat([y,yy.cuda()]);idx=y.argsort(stable=True);x=x[idx];y=y[idx];n=sum(len(v) for v in stages[:stage+1]);N=len(x);nc=counts[:n].cuda();assert torch.equal(torch.bincount(y,minlength=n),nc)
   opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10);curves=[];gradmax=0.;stage_steps=0
   @torch.no_grad()
   def observe(epoch,obj):
    guard();rng=torch.get_rng_state().clone();cu=torch.cuda.get_rng_state_all();keys=model(x);correct=torch.zeros(n,dtype=torch.long,device='cuda');sums=torch.zeros(n,dtype=torch.double,device='cuda')
    for ids in torch.arange(N,device='cuda').split(256):
     z=distribution_logits(keys[ids],keys,nc,ids,y[ids]);assert torch.isfinite(z).all();p=z.argmax(1);correct+=torch.bincount(y[ids][p==y[ids]],minlength=n);sums.scatter_add_(0,y[ids],F.cross_entropy(z,y[ids],reduction='none').double())
    a=correct.double()/nc;tail=nc<250;w=model.weights();curves.append(dict(epoch=epoch,objective_mean=obj,optimizer_steps=stage_steps,full_TRAIN_counts=nc.tolist(),full_TRAIN_LOO_correct=correct.tolist(),macro_LOO_NLL=float((sums/nc).mean()),macro_LOO_BA=float(a.mean()),tail_LOO_BA=float(a[tail].mean()),metric_weight_min=float(w.min()),metric_weight_max=float(w.max()),metric_effective_dimensions=float(w.sum().square()/w.square().sum()),lr=opt.param_groups[0]['lr']));write(folder/f'curves{stage}.json',curves);assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cu,torch.cuda.get_rng_state_all()))
   observe(0,None)
   for epoch in range(10):
    guard();perm=torch.randperm(N,device='cuda');total=0.
    for bi,ids in enumerate(perm.split(256)):
     if bi%32==0:guard();status(phase='FULL_REAL_LOO_TRAIN',seed=ss,stage=stage,epoch=epoch,seen=n,history_rows=N-len(xx),current_rows=len(xx),optimizer_steps=seedsteps,batch=bi)
     opt.zero_grad(set_to_none=True);keys=model(x);z=distribution_logits(keys[ids],keys,nc,ids,y[ids]);w=N/(n*nc[y[ids]].float());loss=(w*F.cross_entropy(z,y[ids],reduction='none')).mean()+model.regularizer();assert torch.isfinite(loss);loss.backward();assert model.a.grad is not None and torch.isfinite(model.a.grad).all();gradmax=max(gradmax,float(model.a.grad.norm()));opt.step();assert torch.isfinite(model.a).all();stage_steps+=1;seedsteps+=1;newsteps+=1;total+=float(loss.detach())*len(ids)
    sched.step();observe(epoch+1,total/N)
   assert stage_steps==10*math.ceil(N/256) and gradmax>0;state=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},seen=n,stage=stage,seed=ss);torch.save(state,folder/f'head{stage}.private.pt')
   with torch.no_grad(),torch.random.fork_rng(devices=[0]):
    restored=torch.load(folder/f'head{stage}.private.pt',map_location='cuda',weights_only=False);other=Metric().cuda();other.load_state_dict(restored['state'],strict=True);assert torch.equal(model(x[:16]),other(x[:16]));del other,restored
   records.append(dict(seed=ss,stage=stage,seen=n,train_counts=nc.tolist(),history_rows=N-len(xx),current_rows=len(xx),all_seen_real_rows=N,class_total_risk_mass=[1.]*n,optimizer_steps=stage_steps,epochs=10,curves=len(curves),max_gradient_norm=gradmax,learned_shared_parameters=512,class_specific_learned_parameters=0,all_query_self_entries_excluded=True,own_class_support_denominator=nc.sub(1).tolist(),no_support_gradient_detach=True,strict_restore_predictions_exact=True,VAL_reads=0));write(OUT/'TRAIN_RECORDS.json',records);del opt,sched
  write(folder/'TRAIN_SEAL.json',dict(stages=len(stages),optimizer_steps=seedsteps,VAL_reads=0));del model,x,y;torch.cuda.empty_cache()
 assert newsteps==expected;write(OUT/'ALL_HEADS_SEALED.json',dict(records=len(records),optimizer_steps=newsteps,VAL_reads=0))
 vr=rows('val');vids=list(range(len(tr),len(tr)+len(vr)));vy=torch.tensor([order.index(int(v['original_label'])) for v in vr]);v=torch.load(CFG['validation_cache'],map_location='cpu',weights_only=False);assert v['ids']==vids;vx=v['x'];assert torch.isfinite(vx).all() and torch.allclose(vx.norm(dim=1),torch.ones(len(vx)),atol=2e-6);write(OUT/'VAL_CACHE_RECEIPT.json',dict(exact_ids=True,unit_cache_no_second_normalization=True,read_after_all_heads_sealed=True,rows=len(vx)))
 results=[]
 with torch.no_grad():
  for ss in [74002,74003,74004]:
   folder=OUT/str(ss);x=None;y=None
   for stage in range(len(stages)):
    guard();xx,yy=arrived(stage);x=xx.cuda() if x is None else torch.cat([x,xx.cuda()]);y=yy.cuda() if y is None else torch.cat([y,yy.cuda()]);idx=y.argsort(stable=True);x=x[idx];y=y[idx];h=torch.load(folder/f'head{stage}.private.pt',map_location='cuda',weights_only=False);model=Metric().cuda();model.load_state_dict(h['state'],strict=True);n=h['seen'];nc=counts[:n].cuda();keys=model(x);rawkeys=F.normalize(x,dim=1);means=torch.stack([x[y==c].mean(0) for c in range(n)]);mask=vy<n;yyv=vy[mask];pieces={k:[] for k in ['distribution_shared','distribution_fixed','centroid_shared']}
    for xb in vx[mask].split(256):
     q=model(xb.cuda());pieces['distribution_shared'].append(distribution_logits(q,keys,nc).cpu());pieces['distribution_fixed'].append(distribution_logits(xb.cuda(),rawkeys,nc).cpu());pieces['centroid_shared'].append((20*q@model(means).T).cpu())
    scores={k:torch.cat(z) for k,z in pieces.items()};torch.save(dict(scores=scores,labels=yyv,ids=[vids[i] for i in torch.where(mask)[0].tolist()]),folder/f'responses{stage}.private.pt');results.append(dict(seed=ss,stage=stage,seen=n,old=sum(len(v) for v in stages[:stage]),metrics={k:metrics(z,yyv,n,counts) for k,z in scores.items()}));del model,h,keys,rawkeys
 write(OUT/'aggregate.json',dict(dataset=ds,results=results,train_counts=counts.tolist(),optimizer_steps=newsteps,trained_stages=len(records),readout_arms=3,no_TEST=True,independent_confirmation=False));status(phase='COMPLETE')
if __name__=='__main__':
 try:main()
 except BaseException:(OUT/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED');raise
