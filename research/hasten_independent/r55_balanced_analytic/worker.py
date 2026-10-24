"""Complete frozen-coordinate long-tail mainline; true TRAIN sufficient statistics."""
import os,json,csv,time,math,traceback,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
CFG=json.loads(Path(os.environ['RUN_CONFIG']).read_text());OUT=Path(CFG['output']);OUT.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
D=2048

def write(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def status(**v):write(OUT/'STATUS.json',dict(time=time.time(),**v))
def guard():
 origin=json.loads(Path(CFG['clock']).read_text());assert time.time()<origin['deadline_unix']-35,'ORIGINAL_DEADLINE';assert torch.cuda.max_memory_reserved()<8*2**30,'GPU_BUDGET'
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
class Map(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Linear(512,64);self.b=nn.Linear(64,512);nn.init.zeros_(self.b.weight);nn.init.zeros_(self.b.bias)
 def forward(self,x):
  d=self.b(F.gelu(self.a(x)));return F.normalize(x+d,dim=1),d

def features(model,x,R):
 u,_=model(x);z=F.relu(u@R)*math.sqrt(2/D);return torch.cat([z,torch.ones(len(z),1,device=z.device,dtype=z.dtype)],1).double()

def metrics(z,y,n,counts):
 p=z.argmax(1);nnn=torch.bincount(y,minlength=n);cc=torch.bincount(y[p==y],minlength=n);tail=[c for c in range(n) if counts[c]<250];acc=cc.double()/nnn
 return dict(correct=cc.tolist(),counts=nnn.tolist(),BA=float(acc.mean()),tail_BA=float(acc[tail].mean()),all_finite=bool(torch.isfinite(z).all()))

def main():
 ds=CFG['dataset'];order=CFG['order'];stages=CFG['stages'];C=len(order)
 def rows(split):
  with open(Path(CFG['manifest'])/(split+'.csv')) as f:a=list(csv.DictReader(f))
  assert all(v['split']==split for v in a);return a
 tr=rows('train');y=torch.tensor([order.index(int(v['original_label'])) for v in tr]);counts=torch.bincount(y,minlength=C);assert (counts>0).all();assert counts.tolist()==CFG['train_counts_ordered']
 def cache(stage):
  guard();a=torch.load(Path(CFG['train_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);ids=[i for i,v in enumerate(y.tolist()) if v in stages[stage]];assert a['ids']==ids
  x=a['x'];x=F.normalize(x,dim=1) if ds=='NIH' else x;assert x.shape==(len(ids),512);assert torch.isfinite(x).all() and torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6)
  write(OUT/f'cache_stage{stage}_receipt.json',dict(exact_ids=True,stage=stage,rows=len(ids),no_VAL=True,no_future_features=True));return x.cuda(),y[ids].cuda()
 allheads=[];newsteps=0
 for ss in [74002,74003,74004]:
  seed(ss);folder=OUT/str(ss);folder.mkdir(exist_ok=True);assert not (folder/'TRAIN_SEAL.json').exists(),'DO_NOT_RETRAIN_COMPLETE_SEED'
  x0,y0=cache(0);n0=len(stages[0]);N0=len(x0);nc=torch.bincount(y0,minlength=n0);model=Map().cuda();head=nn.Parameter(torch.stack([x0[y0==c].mean(0) for c in range(n0)]));assert torch.equal(model(x0[:4])[0],F.normalize(x0[:4],dim=1));opt=torch.optim.AdamW(list(model.parameters())+[head],lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20)
  probe=torch.cat([torch.where(y0==c)[0][:64] for c in range(n0)]);curves=[]
  @torch.no_grad()
  def observe(epoch,obj):
   rng=torch.get_rng_state().clone();cu=[v.clone() for v in torch.cuda.get_rng_state_all()];u,_=model(x0[probe]);logits=20*u@F.normalize(head,dim=1).T;pred=logits.argmax(1);nll=F.cross_entropy(logits,y0[probe],reduction='none');cc=torch.bincount(y0[probe][pred==y0[probe]],minlength=n0);n=torch.bincount(y0[probe],minlength=n0)
   z=dict(epoch=epoch,objective_mean=obj,optimizer_steps=newsteps,probe_counts=n.tolist(),probe_correct=cc.tolist(),macro_NLL=float(torch.stack([nll[y0[probe]==c].mean() for c in range(n0)]).mean()),lr=opt.param_groups[0]['lr']);assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cu,torch.cuda.get_rng_state_all()));curves.append(z);write(folder/'BASE_CURVES.json',curves)
  observe(0,None)
  for epoch in range(20):
   guard();status(phase='BASE_TRAIN',seed=ss,epoch=epoch);perm=torch.randperm(N0,device='cuda');total=0.
   for ids in perm.split(64):
    opt.zero_grad(set_to_none=True);u,d=model(x0[ids]);logits=20*u@F.normalize(head,dim=1).T;w=N0/(n0*nc[y0[ids]].float());loss=(w*F.cross_entropy(logits,y0[ids],reduction='none')).mean()+.01*d.square().sum(1).mean();assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in list(model.parameters())+[head]);opt.step();newsteps+=1;total+=float(loss.detach())*len(ids)
   sched.step();observe(epoch+1,total/N0)
  assert newsteps==(ss-74001)*20*math.ceil(N0/64);model.eval();model.requires_grad_(False);frozen={k:v.clone() for k,v in model.state_dict().items()};gen=torch.Generator(device='cuda').manual_seed(ss+550000);R=torch.randn(512,D,device='cuda',generator=gen);fixedR=R.clone();torch.save(dict(state={k:v.cpu() for k,v in frozen.items()},R=R.cpu(),seed=ss),folder/'MAP.private.pt')
  del opt,head,probe,perm;g=torch.zeros(D+1,D+1,dtype=torch.double,device='cuda');rawg=torch.zeros_like(g);q=torch.zeros(D+1,0,dtype=torch.double,device='cuda');rawq=q.clone();N=0
  for stage,classes in enumerate(stages):
   guard();status(phase='STATISTICS',seed=ss,stage=stage);x,yy=(x0,y0) if stage==0 else cache(stage);old=q.shape[1];n=old+len(classes);assert classes==list(range(old,n));qold=q.clone();q=F.pad(q,(0,len(classes)));rawq=F.pad(rawq,(0,len(classes)));mass=[];rng=torch.get_rng_state().clone();cu=[v.clone() for v in torch.cuda.get_rng_state_all()]
   with torch.no_grad():
    for c in classes:
     xc=x[yy==c];assert len(xc)==counts[c];mass.append(float(len(xc)/int(counts[c])))
     for xb in xc.split(512):
      z=features(model,xb,R);g.addmm_(z.T,z,alpha=1/int(counts[c]));rawg.addmm_(z.T,z);q[:,c]+=z.sum(0)/int(counts[c]);rawq[:,c]+=z.sum(0)
    N+=len(x);assert torch.equal(q[:,:old],qold);assert mass==[1.]*len(classes);assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cu,torch.cuda.get_rng_state_all()));assert torch.equal(R,fixedR) and all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items())
    weights={};audit={}
    for name,A,B in [('balanced',g,q),('sample_weighted',rawg*(n/N),rawq*(n/N))]:
     lam=.01*A.trace()/(D+1);system=A+lam*torch.eye(D+1,device='cuda',dtype=torch.double);L=torch.linalg.cholesky(system);W=torch.cholesky_solve(B,L);err=float((system@W-B).norm()/B.norm().clamp_min(1e-15));assert err<1e-6 and torch.isfinite(W).all();weights[name]=W.cpu();audit[name]=dict(lambda_value=float(lam),normal_equation_relative_residual=err,total_sample_mass=n,criterion='global_equal_class' if name=='balanced' else 'global_equal_sample')
    state=dict(weights=weights,seen=n,old=old,seed=ss,stage=stage,audit=audit);torch.save(state,folder/f'head{stage}.private.pt');torch.save(dict(balanced_G=g.cpu(),raw_G=rawg.cpu(),Q=q.cpu(),raw_Q=rawq.cpu(),rows=N),folder/f'statistics{stage}.private.pt')
    restored=torch.load(folder/f'head{stage}.private.pt',map_location='cuda',weights_only=False);check=features(model,x[:min(16,len(x))],R)
    assert all(torch.equal(check@weights[k].cuda(),check@restored['weights'][k]) for k in weights)
    mapback=torch.load(folder/'MAP.private.pt',map_location='cuda',weights_only=False)
    with torch.random.fork_rng(devices=[0]):
     m2=Map().cuda();m2.load_state_dict(mapback['state'],strict=True);m2.eval();assert torch.equal(features(model,x[:4],R),features(m2,x[:4],mapback['R']));del m2,mapback
   allheads.append(dict(seed=ss,stage=stage,seen=n,train_counts=counts[:n].tolist(),audit=audit,old_Q_exact=True,frozen_map_exact=True,frozen_R_exact=True,strict_restore_predictions_exact=True,class_mass=mass,VAL_reads=0));write(OUT/'TRAIN_RECORDS.json',allheads)
   if stage>0:del x,yy
  write(folder/'TRAIN_SEAL.json',dict(stages=len(stages),optimizer_steps=20*math.ceil(N0/64),VAL_reads=0));del model,R,fixedR,frozen,g,rawg,q,rawq,x0,y0;torch.cuda.empty_cache()
 assert len(allheads)==3*len(stages);assert newsteps==3*20*math.ceil(N0/64);write(OUT/'ALL_HEADS_SEALED.json',dict(records=len(allheads),VAL_reads=0,optimizer_steps=newsteps))
 # No validation feature is opened until all three seeds TRAIN heads are sealed.
 vr=rows('val');ids=list(range(len(tr),len(tr)+len(vr)));vy=torch.tensor([order.index(int(v['original_label'])) for v in vr]);v=torch.load(CFG['validation_cache'],map_location='cpu',weights_only=False);assert v['ids']==ids;vx=v['x'];assert torch.isfinite(vx).all() and torch.allclose(vx.norm(dim=1),torch.ones(len(vx)),atol=2e-6);write(OUT/'VAL_CACHE_RECEIPT.json',dict(exact_ids=True,unit_cache_no_second_normalization=True,read_after_all_heads_sealed=True,rows=len(ids)))
 results=[]
 for ss in [74002,74003,74004]:
  folder=OUT/str(ss);s=torch.load(folder/'MAP.private.pt',map_location='cuda',weights_only=False);model=Map().cuda();model.load_state_dict(s['state'],strict=True);model.eval();R=s['R'];base=[]
  with torch.no_grad():
   for xb in vx.split(512):base.append(features(model,xb.cuda(),R).cpu())
   Z=torch.cat(base)
   for stage in range(len(stages)):
    guard();h=torch.load(folder/f'head{stage}.private.pt',map_location='cpu',weights_only=False);n=h['seen'];mask=vy<n;z=Z[mask];yy=vy[mask];scores={k:z@W for k,W in h['weights'].items()};torch.save(dict(scores=scores,labels=yy,ids=[ids[i] for i in torch.where(mask)[0].tolist()]),folder/f'responses{stage}.private.pt');results.append(dict(seed=ss,stage=stage,seen=n,old=h['old'],metrics={k:metrics(a,yy,n,counts) for k,a in scores.items()}))
  del model,R,s,Z,base;torch.cuda.empty_cache()
 write(OUT/'aggregate.json',dict(dataset=ds,results=results,train_counts=counts.tolist(),optimizer_steps=newsteps,analytic_stages=len(allheads),readout_arms=2,no_TEST=True,independent_confirmation=False));status(phase='COMPLETE')

if __name__=='__main__':
 try:main()
 except BaseException:
  (OUT/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED');raise
