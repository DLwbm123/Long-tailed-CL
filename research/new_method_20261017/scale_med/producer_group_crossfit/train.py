import gc,json,time
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint as recompute
from binding import native,checkpoint
from train_base import setup,raw,finite,opt
from geometry import transform
from sampling import schedule,write
from data import pixels
from space import check_space

def run(cfg):
 root,out=Path(cfg['root']),Path(cfg['output']);plan=torch.load(root/'prep/PLAN.private.pt',weights_only=False)
 allpixels=pixels(cfg);m,params,bkeys,ckeys,heads,pointwise=setup(cfg)
 initial={k:params[k].detach().clone() for k in bkeys};nh=[p.detach().clone() for p in heads]
 receipts=[]
 for fold in ([0] if cfg['mode']=='PREFLIGHT' else [0,1]):
  support=plan['fold']==fold; supids=torch.where(support)[0];query=set(torch.where(~support)[0].tolist())
  y=plan['y'][support];groups=[plan['groups'][i] for i in supids];x=allpixels[support]
  orders=schedule(y,groups);assert not query & set(supids[orders.flatten()].tolist())
  t=torch.load(root/'prep'/f'{fold}.teacher.private.pt',weights_only=False)
  mp={k:tuple(vv.cuda() for vv in v) if isinstance(v,tuple) else v.cuda() for k,v in t['mapper'].items()}
  w0=t['W'].cuda();counts=dict(logical_forward_images=0,forward_calls=0)
  def reset():
   with torch.no_grad():
    for k,v in initial.items():params[k].copy_(v)
   for k in bkeys:params[k].requires_grad_(True)
   m.zero_grad(set_to_none=True)
  def encode(xx):
   counts['logical_forward_images']+=len(xx);counts['forward_calls']+=1
   o=raw(m,pointwise,xx)
   return torch.nn.functional.normalize(torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1)
  def loss(ids,w,ck=True,full=False):
   zz=[]
   for part in ids.split(len(ids) if full else 8):
    zz.append(recompute(encode,x[part].cuda(),use_reentrant=False) if ck else encode(x[part].cuda()))
   return torch.nn.functional.cross_entropy(transform(torch.cat(zz),mp)@w[:-1]+w[-1],y[ids].cuda())
  reset();ps=[params[k] for k in bkeys];assert sum(v.numel() for v in ps)==102912
  assert sorted(k for k,v in params.items() if v.requires_grad)==bkeys
  w=w0.detach().clone().requires_grad_(True)
  if cfg['mode']=='PREFLIGHT':
   ids=orders[0]
   with torch.no_grad():
    g=encode(x[ids].cuda()).cpu();micro=torch.cat([encode(x[z].cuda()).cpu() for z in ids.split(8)])
   err=float((g-plan['cached_g'][supids[ids]]).abs().max());assert err<=1e-10
   fresh=torch.autograd.grad(loss(ids,w,False,True),ps+[w]);check=torch.autograd.grad(loss(ids,w),ps+[w])
   assert all(torch.isfinite(v).all() for v in fresh+check) and all(torch.allclose(a,b,atol=1e-5,rtol=1e-3) for a,b in zip(fresh,check))
   assert fresh[-1].norm()>0 and sum(float(v.square().sum()) for v in fresh[:-1])>0
   optimizer=opt(ps+[w]);optimizer.zero_grad(set_to_none=True);v=loss(ids,w);v.backward();finite(ps+[w]);optimizer.step();finite(ps+[w]);assert not torch.equal(w,w0)
   assert all(torch.equal(v,z) for v,z in zip(heads,nh))
   over=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys);assert all(torch.equal(native(over)[k],params[k].detach().cpu()) for k in bkeys)
   reset();assert all(torch.equal(params[k],v) for k,v in initial.items())
   # Exact full-width persistence qualification, discarded probe values only.
   n=1536;fullw=torch.cat([w.detach().cpu(),torch.zeros(n,2,dtype=torch.float64)],1)
   artifact=dict(A=torch.zeros(n*(n+1)//2,dtype=torch.float64),B=torch.zeros(n,6,dtype=torch.float64),W=fullw,mapper=t['mapper'],PERM=__import__('geometry').PERM,bias_overrides=over,head_origin='joint_CE',classes=4)
   torch.save(artifact,out/'capacity.private.pt');size=(out/'capacity.private.pt').stat().st_size;assert size<=19058800
   back=torch.load(out/'capacity.private.pt',weights_only=False);assert torch.equal(back['W'],fullw)
   write(out/'aggregate.json',dict(pass_all=True,G_replay_error=err,micro_feature_roundoff=float((g-micro).abs().max()),gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(fresh,check)),native_heads_unchanged=True,bias_restore_and_roundtrip_exact=True,probe_discarded=True,query_absent_from_orders=True,capacity_template_bytes=size,forward_counts=counts))
   return
  assert json.loads((root/'preflight/aggregate.json').read_text())['pass_all']
  optimizer=opt(ps+[w]);losses=[];start=time.monotonic()
  torch.save(dict(support_global_ids=supids,orders=orders),out/f'{fold}.sampling.private.pt')
  for step,ids in enumerate(orders):
   optimizer.zero_grad(set_to_none=True);v=loss(ids,w);assert torch.isfinite(v);v.backward();finite(ps+[w]);optimizer.step();finite(ps+[w]);losses.append(float(v.detach()))
   if step%10==0:
    check_space(cfg['window_root']);write(out/'status.json',dict(state='RUNNING',fold=fold,step=step+1,total_steps=800,CE=losses[-1],elapsed_seconds=time.monotonic()-start,forward_counts=counts))
   if (step+1)%100==0:
    check_space(cfg['window_root'],reserve=24*2**20)
    torch.save(dict(step=step+1,bias_overrides=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys),head=w.detach().cpu(),optimizer=optimizer.state_dict()),out/f'{fold}.resume.private.pt')
  assert all(torch.equal(v,z) for v,z in zip(heads,nh))
  over=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys)
  torch.save(dict(bias_overrides=over,head=w.detach().cpu(),mapper=t['mapper'],source_commit=cfg['source_commit']),out/f'{fold}.encoder.private.pt')
  receipts.append(dict(fold=fold,steps=800,supervised_slots=25600,mean_first100=sum(losses[:100])/100,mean_last100=sum(losses[-100:])/100,support_rows=int(support.sum()),query_rows=int((~support).sum()),query_absent_from_updates=True,forward_counts=dict(counts)))
  write(out/'partial.json',dict(producers=receipts));del w,optimizer,x;gc.collect()
 write(out/'ENCODER_SEAL.json',dict(folds=[0,1],steps_each=800,query_excluded_entire_producer=True,development_used=False,future_used=False,test_used=False,source_commit=cfg['source_commit']))
 write(out/'status.json',dict(state='COMPLETE',phase='ENCODERS_SEALED'))
