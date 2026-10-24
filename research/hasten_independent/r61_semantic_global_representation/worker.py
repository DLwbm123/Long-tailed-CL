"""Complete matched representation/semantic/objective study, no readout selection."""
import os,json,csv,time,math,traceback,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
CFG=json.loads(Path(os.environ['RUN_CONFIG']).read_text());OUT=Path(CFG['output']);OUT.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ARMS=['semantic_global','paired_control','permuted_semantic']
def write(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def status(**v):write(OUT/'STATUS.json',dict(time=time.time(),**v))
def guard():
 assert time.time()<json.loads(Path(CFG['clock']).read_text())['deadline_unix']-35,'ORIGINAL_DEADLINE'
 assert torch.cuda.max_memory_reserved()<8*2**30,'GPU_BUDGET'
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Linear(512,64);self.b=nn.Linear(64,512);nn.init.zeros_(self.b.weight);nn.init.zeros_(self.b.bias)
 def forward(self,x):return F.normalize(x+self.b(F.gelu(self.a(x))),dim=1)
def loss_values(z,x,t,y,nc,N,arm):
 w=N/(len(nc)*nc[y].float());scores=20*z@t.T
 if arm=='paired_control':
  paired=scores[:,y];target=torch.arange(len(y),device=y.device);risk=.5*(F.cross_entropy(paired,target,reduction='none')+F.cross_entropy(paired.T,target,reduction='none'))
 else:risk=F.cross_entropy(scores,y,reduction='none')
 return (w*(risk+.01*(z-x).square().sum(1))).mean()
def anchor_permutation(stages,n,arm):
 p=list(range(n))
 if arm=='permuted_semantic':
  for cs in stages:
   if max(cs)>=n:break
   for j,c in enumerate(cs):p[c]=cs[(j+1)%len(cs)]
 return p

def metrics(z,y,n,counts):
 assert torch.isfinite(z).all();pred=z.argmax(1);nnn=torch.bincount(y,minlength=n);cc=torch.bincount(y[pred==y],minlength=n);v=cc.double()/nnn;tail=[i for i in range(n) if counts[i]<250]
 return dict(correct=cc.tolist(),counts=nnn.tolist(),BA=float(v.mean()),tail_BA=float(v[tail].mean()),all_finite=True)

def main():
 ds=CFG['dataset'];stages=CFG['stages'];order=CFG['order'];C=len(order)
 def rows(split):
  with open(Path(CFG['manifest'])/(split+'.csv')) as f:r=list(csv.DictReader(f))
  assert all(v['split']==split for v in r);return r
 tr=rows('train');labels=torch.tensor([order.index(int(v['original_label'])) for v in tr]);counts=torch.bincount(labels,minlength=C);assert counts.tolist()==CFG['train_counts_ordered']
 def arrived(stage):
  guard();ids=[i for i,v in enumerate(labels.tolist()) if v in stages[stage]];dst=OUT/'memory'/f'block{stage}.private.pt'
  if dst.exists():a=torch.load(dst,map_location='cpu',weights_only=False)
  else:
   b=torch.load(Path(CFG['train_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);assert b['ids']==ids;n=sum(len(v) for v in stages[:stage+1]);prompts=[CFG['leaf_template'].format(v) for v in CFG['ordered_names'][:n]];bank=F.normalize(b['text'],dim=1) if ds=='NIH' else b['text'];text=bank[[b['prompts'].index(v) for v in prompts]];x=F.normalize(b['x'],dim=1) if ds=='NIH' else b['x'];a=dict(ids=ids,x=x,y=labels[ids],text=text);dst.parent.mkdir(exist_ok=True);torch.save(a,dst)
  assert a['ids']==ids and torch.equal(a['y'],labels[ids]);assert torch.isfinite(a['x']).all() and torch.allclose(a['x'].norm(dim=1),torch.ones(len(ids)),atol=2e-6);assert torch.allclose(a['text'].norm(dim=1),torch.ones(len(a['text'])),atol=2e-6);return a
 records=[];newsteps=0;expected=9*sum(10*math.ceil(sum(int(counts[c]) for c in sum(stages[:i+1],[]))/256) for i in range(len(stages)))
 for ss in [74002,74003,74004]:
  initial=None
  for arm in ARMS:
   seed(ss);folder=OUT/arm/str(ss);folder.mkdir(exist_ok=True,parents=True);assert not (folder/'TRAIN_SEAL.json').exists(),'DO_NOT_RETRAIN_COMPLETE_SEED';model=Encoder().cuda()
   if initial is None:initial={k:v.clone() for k,v in model.state_dict().items()}
   else:assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
   x=None;y=None;text=None;seedsteps=0
   for stage,classes in enumerate(stages):
    b=arrived(stage);xx=b['x'].cuda();yy=b['y'].cuda();x=xx if x is None else torch.cat([x,xx]);y=yy if y is None else torch.cat([y,yy]);n=sum(len(v) for v in stages[:stage+1]);N=len(x);nc=counts[:n].cuda();assert torch.equal(torch.bincount(y,minlength=n),nc);old=0 if text is None else len(text);newtext=b['text'].cuda()
    assert text is None or torch.allclose(newtext[:old],text,atol=2e-6,rtol=0);text=newtext if text is None else torch.cat([text,newtext[old:]]);t=text[anchor_permutation(stages,n,arm)];fixedtext=text.clone();fixedt=t.clone();assert not text.requires_grad;opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10);curves=[];gradmax=0.;stage_steps=0
    @torch.no_grad()
    def observe(epoch,obj):
     rng=torch.get_rng_state().clone();cu=torch.cuda.get_rng_state_all();cc=torch.zeros(n,dtype=torch.long,device='cuda');sums=torch.zeros(n,dtype=torch.double,device='cuda')
     for ids in torch.arange(N,device='cuda').split(2048):
      zz=20*model(x[ids])@t.T;assert torch.isfinite(zz).all();p=zz.argmax(1);cc+=torch.bincount(y[ids][p==y[ids]],minlength=n);sums.scatter_add_(0,y[ids],F.cross_entropy(zz,y[ids],reduction='none').double())
     a=cc.double()/nc;curves.append(dict(epoch=epoch,objective_mean=obj,optimizer_steps=stage_steps,full_TRAIN_counts=nc.tolist(),full_TRAIN_correct=cc.tolist(),macro_decision_NLL=float((sums/nc).mean()),macro_BA=float(a.mean()),tail_BA=float(a[nc<250].mean()),lr=opt.param_groups[0]['lr']));write(folder/f'curves{stage}.json',curves);assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cu,torch.cuda.get_rng_state_all()))
    observe(0,None);pg=torch.Generator(device='cuda').manual_seed(ss+10000*stage)
    for epoch in range(10):
     guard();perm=torch.randperm(N,device='cuda',generator=pg);total=0.;status(phase='REPRESENTATION_TRAIN',arm=arm,seed=ss,stage=stage,epoch=epoch,seen=n,all_real_rows=N,optimizer_steps=seedsteps)
     for ids in perm.split(256):
      opt.zero_grad(set_to_none=True);z=model(x[ids]);loss=loss_values(z,x[ids],t,y[ids],nc,N,arm);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());gradmax=max(gradmax,max(float(p.grad.norm()) for p in model.parameters()));opt.step();stage_steps+=1;seedsteps+=1;newsteps+=1;total+=float(loss.detach())*len(ids)
     sched.step();observe(epoch+1,total/N)
    assert stage_steps==10*math.ceil(N/256) and gradmax>0;assert torch.equal(text,fixedtext) and torch.equal(t,fixedt);torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},text=text.cpu(),assigned_text=t.cpu(),seen=n,stage=stage,seed=ss,arm=arm),folder/f'head{stage}.private.pt')
    with torch.no_grad(),torch.random.fork_rng(devices=[0]):
     h=torch.load(folder/f'head{stage}.private.pt',map_location='cuda',weights_only=False);other=Encoder().cuda();other.load_state_dict(h['state'],strict=True);assert torch.equal(20*model(x[:16])@t.T,20*other(x[:16])@h['assigned_text'].T);del other,h
    records.append(dict(arm=arm,seed=ss,stage=stage,seen=n,train_counts=nc.tolist(),all_seen_real_rows=N,class_total_risk_mass=[1.]*n,optimizer_steps=stage_steps,epochs=10,curves=11,max_gradient_norm=gradmax,shared_parameters=sum(v.numel() for v in model.parameters()),class_specific_learned_parameters=0,initial_core_paired_exact=True,same_prespecified_batch_generator=True,fixed_anchors_exact=True,old_semantic_prefix_preserved=True,strict_restore_predictions_exact=True,VAL_reads=0));write(OUT/'TRAIN_RECORDS.json',records);del opt,sched
   write(folder/'TRAIN_SEAL.json',dict(stages=len(stages),optimizer_steps=seedsteps,VAL_reads=0));del model,x,y,text;torch.cuda.empty_cache()
 assert newsteps==expected and len(records)==9*len(stages);write(OUT/'ALL_HEADS_SEALED.json',dict(records=len(records),optimizer_steps=newsteps,VAL_reads=0))
 vr=rows('val');vids=list(range(len(tr),len(tr)+len(vr)));vy=torch.tensor([order.index(int(v['original_label'])) for v in vr]);v=torch.load(CFG['validation_cache'],map_location='cpu',weights_only=False);assert v['ids']==vids;vx=v['x'];assert torch.isfinite(vx).all() and torch.allclose(vx.norm(dim=1),torch.ones(len(vx)),atol=2e-6);write(OUT/'VAL_CACHE_RECEIPT.json',dict(exact_ids=True,unit_cache_no_second_normalization=True,read_after_all9_TRAIN_heads_sealed=True,rows=len(vx)))
 results=[]
 with torch.no_grad():
  for ss in [74002,74003,74004]:
   for stage in range(len(stages)):
    guard();n=sum(len(v) for v in stages[:stage+1]);mask=vy<n;yy=vy[mask];scores={}
    for arm in ARMS:
     h=torch.load(OUT/arm/str(ss)/f'head{stage}.private.pt',map_location='cuda',weights_only=False);model=Encoder().cuda();model.load_state_dict(h['state'],strict=True);scores[arm]=torch.cat([(20*model(b.cuda())@h['assigned_text'].T).cpu() for b in vx[mask].split(2048)])
     if arm=='semantic_global':scores['frozen_semantic']=20*vx[mask]@h['text'].cpu().T
     del model,h
    f=OUT/'responses'/str(ss);f.mkdir(exist_ok=True,parents=True);torch.save(dict(scores=scores,labels=yy,ids=[vids[i] for i in torch.where(mask)[0].tolist()]),f/f'responses{stage}.private.pt');results.append(dict(seed=ss,stage=stage,seen=n,old=sum(len(v) for v in stages[:stage]),metrics={k:metrics(z,yy,n,counts) for k,z in scores.items()}))
 write(OUT/'aggregate.json',dict(dataset=ds,results=results,train_counts=counts.tolist(),optimizer_steps=newsteps,trained_stages=len(records),readout_arms=4,no_TEST=True,independent_confirmation=False));status(phase='COMPLETE')
if __name__=='__main__':
 try:main()
 except BaseException:(OUT/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED');raise
