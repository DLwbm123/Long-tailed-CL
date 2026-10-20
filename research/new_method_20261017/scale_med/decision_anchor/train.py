import json,time,gc
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint as recompute
from train_base import setup,raw,finite,opt
from binding import native,checkpoint
from solver import transform
from protection import build,penalty,device_state,synthetic
from space import check_space
ARMS=['E','L','R','D']
def write(p,a):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(a,indent=2,allow_nan=False));t.replace(p)
def schedule(y,groups):
 gen=torch.Generator().manual_seed(74250);members={c:{g:torch.tensor([i for i,v in enumerate(groups) if v==g and int(y[i])==c]) for g in sorted({groups[i] for i in torch.where(y==c)[0].tolist()})} for c in range(4)};orders=[];assert all(len(v)>=8 for v in members.values())
 for step in range(800):
  ids=[]
  for c in range(4):
   keys=list(members[c]);chosen=torch.randperm(len(keys),generator=gen)[:8]
   for j in chosen:
    val=members[c][keys[int(j)]];ids.append(int(val[torch.randint(len(val),(1,),generator=gen)]))
  assert len({groups[i] for i in ids})==32;orders.append(torch.tensor(ids))
 return torch.stack(orders)
def load_training(cfg):
 obj=torch.load(Path(cfg['reference'])/'inputs/0.private.pt',weights_only=False);tr=~obj['valid'];y=obj['y'][tr];groups=[obj['groups'][i] for i in torch.where(tr)[0].tolist()];idx=json.loads((Path(cfg['matched'])/'pixels_0/INDEX.json').read_text());pixels=[]
 for cell in idx['chunks']:
  x=torch.load(Path(cfg['matched'])/'pixels_0'/cell['file'],weights_only=False);pixels.append(x[tr[cell['first']:cell['first']+len(x)]])
 assert len(y)==2555;return torch.cat(pixels),y,groups,obj['g'][tr].double()
def run(cfg):
 out=Path(cfg['output']);toy=synthetic() if cfg['mode']=='PREFLIGHT' else {};pixels,y,groups,cached=load_training(cfg);orders=schedule(y,groups);torch.save(orders,out/'sampling.private.pt');s=build(cached,y,groups);torch.save(s,out/'protection.private.pt');reference=transform(cached,s['mapper']);s=device_state(s,'cuda');m,params,bkeys,ckeys,heads,pw=setup(cfg);initial={k:params[k].detach().clone() for k in bkeys};ih=[p.detach().clone() for p in heads];counts={'logical_forward_images':0,'forward_calls':0}
 def reset():
  with torch.no_grad():
   for k,v in initial.items():params[k].copy_(v)
   for p,v in zip(heads,ih):p.copy_(v)
  for k in bkeys:params[k].requires_grad_(True)
  for p in heads:p.requires_grad_(True)
  m.zero_grad(set_to_none=True)
 def encode(x):
  counts['logical_forward_images']+=len(x);counts['forward_calls']+=1;o=raw(m,pw,x);g=torch.nn.functional.normalize(torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1);return g,o['logits'][:,:4],o['logits_few'][:,:4]
 def task(ids,arm,ck=True,full=False):
  vals=[[],[],[]]
  for part in ids.split(len(ids) if full else 8):
   z=recompute(encode,pixels[part].cuda(),use_reentrant=False) if ck else encode(pixels[part].cuda())
   for a,v in zip(vals,z):a.append(v)
  g,a,b=[torch.cat(v) for v in vals];ce=(torch.nn.functional.cross_entropy(a,y[ids].cuda())+torch.nn.functional.cross_entropy(b,y[ids].cuda()))/2;delta=transform(g,s['mapper'])-reference[ids].cuda();reg=penalty(delta,s,arm);return ce+2*reg,ce,reg,delta.square().sum(1).mean()
 reset();ps=[params[k] for k in bkeys]+heads
 if cfg['mode']=='PREFLIGHT':
  ids=orders[0]
  with torch.no_grad():
   g=encode(pixels[ids].cuda())[0].cpu();micro=torch.cat([encode(pixels[a].cuda())[0].cpu() for a in ids.split(8)])
  err=float((g-cached[ids]).abs().max());assert err<=1e-10
  # A fixed parameter perturbation exposes nonzero selective restoring gradients.
  gen=torch.Generator(device='cuda').manual_seed(74253)
  with torch.no_grad():
   for k in bkeys:params[k].add_(.001*torch.randn(params[k].shape,device='cuda',generator=gen))
  probe=[]
  for arm in ['L','R','D']:
   loss,ce,reg,drift=task(ids,arm,False,True);fresh=torch.autograd.grad(loss,ps);loss2,_,reg2,_=task(ids,arm,True);cp=torch.autograd.grad(loss2,ps);ge=max(float((a-b).abs().max()) for a,b in zip(fresh,cp));assert all(torch.allclose(a,b,atol=1e-5,rtol=1e-3) for a,b in zip(fresh,cp));rr=task(ids,arm,False,True)[2];gr=torch.autograd.grad(rr,[params[k] for k in bkeys]);length=sum(float(v.square().sum()) for v in gr)**.5;assert length>0 and all(torch.isfinite(v).all() for v in gr);probe.append(dict(arm=arm,regularizer=float(reg.detach()),gradient_L2=length,checkpoint_gradient_max_error=ge))
  for arm in ARMS:
   reset();optimizer=opt(ps);optimizer.zero_grad();l=task(ids,arm)[0];l.backward();finite(ps);optimizer.step();finite(ps)
   with torch.no_grad():
    for p,v in zip(heads,ih):assert torch.equal(p[4:],v[4:])
   over=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys);assert all(torch.equal(native(over)[k],params[k].detach().cpu()) for k in bkeys)
  reset();assert all(torch.equal(params[k],v) for k,v in initial.items());assert all(torch.equal(p,v) for p,v in zip(heads,ih))
  from core_depth import PERM
  template=dict(A=torch.ones(1536*1537//2,dtype=torch.float64),B=torch.zeros(1536,6,dtype=torch.float64),W=torch.zeros(1536,6,dtype=torch.float64),mapper={k:(v.cpu() if torch.is_tensor(v) else tuple(t.cpu() for t in v)) for k,v in s['mapper'].items()},PERM=PERM,bias_overrides=over,alpha=.01,source_commit=cfg['source_commit']);torch.save(template,out/'capacity.private.pt');capacity=(out/'capacity.private.pt').stat().st_size;assert capacity<=19058800;back=torch.load(out/'capacity.private.pt',weights_only=False);assert all(torch.equal(v,back['bias_overrides'][k]) for k,v in over.items())
  write(out/'aggregate.json',dict(pass_all=True,**toy,actual_protection_gradients=probe,zero_G_replay_error=err,micro_feature_roundoff=float((g-micro).abs().max()),actual_four_steps_finite=True,bias_roundtrip_restore_exact=True,head_rows_outside4_unchanged=True,capacity_template_bytes=capacity,training_rows_only=True,teacher_rank=s['r'],support_dimension=s['m'],random_overlap=s['random_overlap_singular_values'],discarded_probe_parameters=True,forward_counts=counts));return
 assert json.loads(Path(cfg['preflight']).read_text())['pass_all'];receipts=[]
 for arm in ARMS:
  reset();optimizer=opt(ps);counts.update(logical_forward_images=0,forward_calls=0);logs=[];start=time.monotonic()
  for step,ids in enumerate(orders):
   optimizer.zero_grad(set_to_none=True);l,ce,reg,drift=task(ids,arm);assert torch.isfinite(l);l.backward();finite(ps);optimizer.step();finite(ps);logs.append([float(ce.detach()),float(reg.detach()),float(drift.detach())])
   if step%10==0:check_space(cfg['window_root']);write(out/'status.json',dict(state='RUNNING',arm=arm,step=step+1,total_steps=800,elapsed_seconds=time.monotonic()-start,last_CE=logs[-1][0],last_protection=logs[-1][1],forward_counts=counts))
   if (step+1)%100==0:
    check_space(cfg['window_root'],reserve=24*2**20);torch.save(dict(arm=arm,step=step+1,bias_overrides=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys),heads=[p.detach().cpu().clone() for p in heads],optimizer=optimizer.state_dict()),out/(arm+'.resume.private.pt'))
  over=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys);torch.save(dict(bias_overrides=over,source_commit=cfg['source_commit']),out/(arm+'.encoder.private.pt'));receipts.append(dict(arm=arm,steps=800,supervised_slots=25600,mean_first100=[sum(v[j] for v in logs[:100])/100 for j in range(3)],mean_last100=[sum(v[j] for v in logs[-100:])/100 for j in range(3)],loss_component_order=['CE','protection','total_drift'],forward_counts=dict(counts)));write(out/'partial.json',dict(arms=receipts))
 write(out/'ENCODER_SEAL.json',dict(arms=ARMS,steps_each=800,future_frozen=True,initial_training_only=True,test_used=False,bias_parameters=102912,teacher_rank=s['r'],support_dimension=s['m'],random_overlap=s['random_overlap_singular_values'],source_commit=cfg['source_commit']));write(out/'aggregate.json',dict(decision='FOUR_FIXED_TRAJECTORIES_COMPLETE',arms=receipts,method_success=False));write(out/'status.json',dict(state='COMPLETE',phase='ENCODERS_SEALED'))
