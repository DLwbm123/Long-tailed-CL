import os,json,time
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint as recompute
from train_base import setup,raw,finite,opt
from binding import native,checkpoint
from solver import objective,synthetic
from space import check_space

def write(p,a):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(a,indent=2,allow_nan=False));t.replace(p)
def schedule(y,groups):
 gen=torch.Generator().manual_seed(74220);members={c:{g:torch.tensor([i for i,v in enumerate(groups) if v==g and int(y[i])==c]) for g in sorted({groups[i] for i in torch.where(y==c)[0].tolist()})} for c in range(4)};assert all(len(v)>=8 for v in members.values());orders=[]
 for step in range(800):
  sides=[[],[]]
  for c in range(4):
   keys=list(members[c]);chosen=torch.randperm(len(keys),generator=gen)[:8]
   for j,k in enumerate(chosen):
    ids=members[c][keys[int(k)]];sides[j//4].append(int(ids[torch.randint(len(ids),(1,),generator=gen)]))
  order=torch.tensor(sides[0]+sides[1]);assert len({groups[i] for i in order.tolist()})==32;orders.append(order)
 return torch.stack(orders),members

def load_training(cfg):
 obj=torch.load(Path(cfg['reference'])/'inputs/0.private.pt',weights_only=False);mask=~obj['valid'];y=obj['y'][mask];groups=[obj['groups'][i] for i in torch.where(mask)[0].tolist()];index=json.loads((Path(cfg['matched'])/'pixels_0/INDEX.json').read_text());values=[]
 for cell in index['chunks']:
  x=torch.load(Path(cfg['matched'])/'pixels_0'/cell['file'],weights_only=False);values.append(x[mask[cell['first']:cell['first']+len(x)]])
 assert len(y)==2555;return torch.cat(values),y,groups,obj['g'][mask]

def run(cfg):
 out=Path(cfg['output']);pixels,y,groups,cached=load_training(cfg);orders,members=schedule(y,groups);torch.save(orders,out/'sampling.private.pt');m,params,bkeys,ckeys,heads,pointwise=setup(cfg);initial_bias={k:params[k].detach().clone() for k in bkeys};initial_heads=[p.detach().clone() for p in heads];counter={'logical_forward_images':0,'forward_calls':0}
 def encode(x):
  counter['logical_forward_images']+=len(x);counter['forward_calls']+=1;o=raw(m,pointwise,x);g=torch.nn.functional.normalize(torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1);return g,o['logits'][:,:4],o['logits_few'][:,:4]
 def task(ids,arm,ck=True):
  vals=[[],[],[]]
  for part in ids.split(8):
   x=pixels[part].cuda();v=recompute(encode,x,use_reentrant=False) if ck else encode(x)
   for i,z in enumerate(v):vals[i].append(z)
  g,a,b=[torch.cat(z) for z in vals];yy=y[ids].cuda()
  return (torch.nn.functional.cross_entropy(a,yy)+torch.nn.functional.cross_entropy(b,yy))/2 if arm=='E' else objective(g,yy,arm)
 def reset(arm):
  with torch.no_grad():
   for k,v in initial_bias.items():params[k].copy_(v)
   for p,v in zip(heads,initial_heads):p.copy_(v)
  for k in bkeys:params[k].requires_grad_(True)
  for p in heads:p.requires_grad_(arm=='E')
  m.zero_grad(set_to_none=True)
 if cfg['mode']=='PREFLIGHT':
  s=synthetic();ids=orders[0];reset('O');ps=[params[k] for k in bkeys]
  with torch.no_grad():g=torch.cat([encode(pixels[z].cuda())[0].cpu() for z in ids.split(8)])
  replay=float((g-cached[ids]).abs().max());assert replay<=1e-10
  loss=task(ids,'O',False);fresh=torch.autograd.grad(loss,ps);loss2=task(ids,'O',True);cp=torch.autograd.grad(loss2,ps);err=max(float((a-b).abs().max()) for a,b in zip(fresh,cp));assert all(torch.allclose(a,b,atol=1e-5,rtol=1e-3) for a,b in zip(fresh,cp))
  length=sum(float(g.square().sum()) for g in fresh)**.5;assert length>0;direction=[g/length for g in fresh];epsilon=.001
  def shifted(sign):
   with torch.no_grad():
    for k,d in zip(bkeys,direction):params[k].copy_(initial_bias[k]+sign*epsilon*d)
    return float(task(ids,'O',False))
  fd=(shifted(1)-shifted(-1))/(2*epsilon);assert abs(fd-length)<=.05*abs(length)+.001
  reset('O');optimizer=opt(ps);optimizer.zero_grad();task(ids,'O',True).backward();finite(ps);optimizer.step();finite(ps);overrides=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys);assert all(torch.equal(native(overrides)[k],params[k].detach().cpu()) for k in bkeys)
  # Test I and E actual paths too, then discard all trained probe parameters.
  for arm in ['I','E']:
   reset(arm);pp=ps+(heads if arm=='E' else []);optimizer=opt(pp);optimizer.zero_grad();l=task(ids,arm,True);assert torch.isfinite(l);l.backward();finite(pp);optimizer.step();finite(pp)
  # Match actual persistence format at full original width, without fitting a probe classifier.
  from solver import mapper
  from core_depth import PERM
  g=torch.cat([encode(pixels[z].cuda())[0].detach().cpu() for z in ids.split(8)]);q=torch.ones(32,dtype=torch.float64)/8;t=mapper(g,q);d=1536;template=dict(A=torch.ones(d*(d+1)//2,dtype=torch.float64),B=torch.zeros(d,6,dtype=torch.float64),W=torch.zeros(d,6,dtype=torch.float64),mapper=t,PERM=PERM,bias_overrides=overrides,bias_representation='absolute_overrides',alpha=.01,classes=6,source_commit=cfg['source_commit']);torch.save(template,out/'capacity.private.pt');capacity=(out/'capacity.private.pt').stat().st_size;assert capacity<=19058800;back=torch.load(out/'capacity.private.pt',weights_only=False);assert all(torch.equal(back['bias_overrides'][k],v) for k,v in overrides.items())
  write(out/'aggregate.json',dict(pass_all=True,capacity_template_bytes=capacity,**s,zero_G_replay_error=replay,micro_checkpoint_gradient_error=err,bias_direction_derivative=length,bias_center_difference=fd,bias_roundtrip_exact=True,actual_E_I_O_step_finite=True,training_query_not_validation=True,sampling_steps=800,unique_groups_per_task=32,discarded_probe_models=True,forward_counts=counter));return
 assert json.loads(Path(cfg['preflight']).read_text())['pass_all'];arm_receipts=[]
 for arm in ['E','I','O']:
  reset(arm);ps=[params[k] for k in bkeys]+(heads if arm=='E' else []);optimizer=opt(ps);counter.update(logical_forward_images=0,forward_calls=0);start=time.monotonic();losses=[]
  for step,ids in enumerate(orders):
   optimizer.zero_grad(set_to_none=True);l=task(ids,arm,True);assert torch.isfinite(l);l.backward();finite(ps);optimizer.step();finite(ps);losses.append(float(l.detach()))
   if step%10==0:check_space(cfg['window_root']);write(out/'status.json',dict(state='RUNNING',arm=arm,step=step+1,total_steps=800,elapsed_seconds=time.monotonic()-start,forward_counts=counter))
   if (step+1)%100==0:
    check_space(cfg['window_root'],reserve=16*2**20);torch.save(dict(arm=arm,step=step+1,bias_overrides=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys),heads=[p.detach().cpu().clone() for p in heads] if arm=='E' else [],optimizer=optimizer.state_dict()),out/(arm+'.resume.private.pt'))
  overrides=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys);assert all(torch.equal(native(overrides)[k],params[k].detach().cpu()) for k in bkeys);torch.save(dict(bias_overrides=overrides,source_commit=cfg['source_commit']),out/(arm+'.encoder.private.pt'));exposure={}
  for c in range(4):
   selected=orders.flatten();selected=selected[y[selected]==c];from collections import Counter
   freq=Counter(groups[i] for i in selected.tolist());frequencies=[freq[g] for g in members[c]];exposure[str(c)]=dict(slots=len(selected),known_groups=len(members[c]),group_slot_min=min(frequencies),group_slot_max=max(frequencies))
  arm_receipts.append(dict(arm=arm,steps=800,supervised_slots=25600,loss_first100=sum(losses[:100])/100,loss_last100=sum(losses[-100:])/100,forward_counts=dict(counter),class_exposure=exposure));write(out/'partial.json',dict(arms=arm_receipts))
 write(out/'ENCODER_SEAL.json',dict(arms=['E','I','O'],steps_each=800,supervised_slots_each=25600,bias_parameters=102912,bias_bytes=411648,initial_training_only=True,future_frozen=True,test_used=False,source_commit=cfg['source_commit']));write(out/'aggregate.json',dict(decision='THREE_FIXED_TRAJECTORIES_COMPLETE',arms=arm_receipts,method_success=False));write(out/'status.json',dict(state='COMPLETE',phase='ENCODERS_SEALED'))
