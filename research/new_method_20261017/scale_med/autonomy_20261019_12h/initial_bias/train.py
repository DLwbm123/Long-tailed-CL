import os,sys,json,time
from pathlib import Path
import torch
from binding import native,checkpoint
from space import check_space

def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def setup(cfg):
 bc=json.loads(Path(cfg['benchmark_config']).read_text());s=Path(bc['legacy_source']);sys.path.insert(0,str(s/'third_party/APART'));sys.path.insert(0,str(s/'tools'))
 from run_multilabel import ApartFeatures
 from run_medical_v2 import seed_all
 from ct3p_core import pointwise
 from safetensors.torch import load_file
 lock=json.loads((s/'docs/real_query_node1_20261016/PROTOCOL_LOCK.json').read_text());seed_all(lock['seed']);m=ApartFeatures(str(s),bc['weight'],23,'cuda',lock['seed']).eval().requires_grad_(False);p=dict(m.net.backbone.named_parameters());w=load_file(bc['weight']);keys=sorted(k for k in w if k.endswith('.bias') and not k.startswith('head.'));bound=native({k:w[k] for k in keys});assert all(torch.equal(p[k].detach().cpu(),v) for k,v in bound.items());heads=list(m.net.backbone.head.parameters())+list(m.net.backbone.head_few.parameters())
 return m,p,sorted(bound),keys,heads,pointwise

def install_guard(cfg,stage=0):
 bc=json.loads(Path(cfg['benchmark_config']).read_text());ref=Path(cfg['reference']).resolve();matched=Path(cfg['matched']).resolve();testroot=Path(bc['output']).resolve();raw=[Path(bc['datasets']['ISIC'][z]).resolve() for z in ['images','development_images']]
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();assert not any(p.is_relative_to(z) for z in raw),'raw image forbidden'
  if p.is_relative_to(ref/'inputs'):assert stage<3 and p==ref/'inputs'/f'{stage}.private.pt','future or retired support'
  if p.is_relative_to(matched):assert p.is_relative_to(matched/f'pixels_{stage}'),'future, retired or test pixels'
  if p.is_relative_to(testroot):assert stage==3,'test forbidden'
 sys.addaudithook(guard)

def raw(m,pointwise,x):
 with pointwise(m.net,m.learner):return m.net(x,train=False,weight=None)
def loss(o,y,q,n):
 v=(torch.nn.functional.cross_entropy(o['logits'][:,:4],y,reduction='none')+torch.nn.functional.cross_entropy(o['logits_few'][:,:4],y,reduction='none'))/2
 return (n/4)*(q*v).mean()
def opt(ps):return torch.optim.AdamW(ps,lr=.003,weight_decay=0,betas=(.9,.999),eps=1e-8)
def finite(ps):assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in ps)
def main(cfg):
 out=Path(cfg['output']);start=time.monotonic();mode=cfg['mode'];stage=int(os.environ.get('FEATURE_STAGE','0'));install_guard(cfg,stage);m,params,bkeys,ckeys,heads,pointwise=setup(cfg)
 if mode=='FEATURE':
  state=torch.load(Path(cfg['training_output'])/'encoder.private.pt',weights_only=False);mapped=native(state['bias_overrides']);assert sorted(mapped)==bkeys
  with torch.no_grad():
   for k in bkeys:params[k].copy_(mapped[k].cuda())
  index=json.loads((Path(cfg['matched'])/f'pixels_{stage}/INDEX.json').read_text());values=[]
  with torch.no_grad():
   for cell in index['chunks']:
    x=torch.load(Path(cfg['matched'])/f'pixels_{stage}'/cell['file'],weights_only=False)
    for xx in x.split(32):values.append(torch.nn.functional.normalize(m(xx.cuda()).cpu().double(),dim=1))
  v=torch.cat(values);assert torch.isfinite(v).all();check_space(cfg['window_root'],reserve=v.numel()*v.element_size()*3);torch.save(v,out/'g.private.pt');write(out/'aggregate.json',dict(rows=len(v),finite=True,source_commit=cfg['source_commit']));return
 obj=torch.load(Path(cfg['reference'])/'inputs/0.private.pt',weights_only=False);tr=~obj['valid'];ix=torch.where(tr)[0];yy=obj['y'][tr];gg=[obj['groups'][i] for i in ix.tolist()]
 from core_attribution import weights
 q=weights(yy,gg).float();assert len(ix)==2555 and abs(float(q.sum())-4)<1e-6
 # Read only current stage pixels, select training rows before any objective/feature cache.
 idx=json.loads((Path(cfg['matched'])/'pixels_0/INDEX.json').read_text());chunks=[]
 for cell in idx['chunks']:
  f=cell['first'];keep=tr[f:f+cell['rows']];x=torch.load(Path(cfg['matched'])/'pixels_0'/cell['file'],weights_only=False);chunks.append(x[keep])
 pixels=torch.cat(chunks);del chunks,obj;assert len(pixels)==2555
 for p in heads:p.requires_grad_(True)
 initial_heads=[p.detach().clone() for p in heads];initial_bias={k:params[k].detach().clone() for k in bkeys}
 if mode=='TRAIN_PREFLIGHT':
  pick=torch.cat([torch.where(yy==c)[0][:2] for c in range(4)]);x=pixels[pick].cuda();y=yy[pick].cuda();w=q[pick].cuda();o=raw(m,pointwise,x);z=[o[k].detach() for k in ['pre_logits','pre_logits_few']];baseline=torch.nn.functional.normalize(torch.cat(z,1),dim=1);fresh=loss(o,y,w,2555);gr=torch.autograd.grad(fresh,heads)
  co=dict(logits=m.net.backbone.head(z[0]),logits_few=m.net.backbone.head_few(z[1]));cached=loss(co,y,w,2555);cg=torch.autograd.grad(cached,heads);err=max(float((a-b).abs().max()) for a,b in zip(gr,cg));assert err<=1e-7 and float(abs(fresh-cached))<=1e-7
  optimizer=opt(heads);optimizer.zero_grad();loss(raw(m,pointwise,x),y,w,2555).backward();optimizer.step();after=raw(m,pointwise,x);g=torch.nn.functional.normalize(torch.cat([after['pre_logits'],after['pre_logits_few']],1),dim=1);assert torch.equal(g,baseline)
  with torch.no_grad():
   for p,v in zip(heads,initial_heads):p.copy_(v)
  for k in bkeys:params[k].requires_grad_(True)
  ps=[params[k] for k in bkeys]+heads;optimizer=opt(ps);optimizer.zero_grad();l=loss(raw(m,pointwise,x),y,w,2555);l.backward();finite(ps);assert any(float(params[k].grad.norm())>0 for k in bkeys);optimizer.step();finite(ps)
  overrides=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys);assert all(torch.equal(native(overrides)[k],params[k].detach().cpu()) for k in bkeys)
  # q-weighted uniform batch mean equals full objective in expectation; check exhaustive singleton average.
  v=torch.arange(1,9,dtype=torch.float32);toyq=torch.ones(8)/2;assert torch.allclose(torch.stack([(8/4)*(toyq[i:i+1]*v[i:i+1]).mean() for i in range(8)]).mean(),(toyq*v).sum()/4)
  torch.save(dict(bias_overrides=overrides),out/'preflight_encoder.private.pt')
  write(out/'aggregate.json',dict(pass_all=True,head_only_features_exact=True,head_cached_gradient_error=err,actual_bias_step_finite=True,canonical_override_roundtrip=True,weighted_loss_rule=True,training_rows=8,discarded_preflight_model=True,not_formal_training=True));return
 assert mode=='TRAIN';assert json.loads(Path(cfg['preflight_training']).read_text())['pass_all'];logs=[];n=len(yy)
 # HeadOnly exact cache: these two pre_logits do not depend on the temporary heads.
 vals=[[],[]]
 with torch.no_grad():
  for x in pixels.split(32):
   o=raw(m,pointwise,x.cuda());vals[0].append(o['pre_logits'].cpu());vals[1].append(o['pre_logits_few'].cpu())
 z=[torch.cat(v) for v in vals];del vals
 def run(kind):
  ps=heads if kind=='HeadOnly' else [params[k] for k in bkeys]+heads
  for k in bkeys:params[k].requires_grad_(kind=='Bias')
  optimizer=opt(ps);gen=torch.Generator().manual_seed(74162)
  for epoch in range(10):
   order=torch.randperm(n,generator=gen);seen=0;total=0.
   for step,ids in enumerate(order.split(8)):
    optimizer.zero_grad(set_to_none=True);y=yy[ids].cuda();w=q[ids].cuda()
    if kind=='HeadOnly':o=dict(logits=m.net.backbone.head(z[0][ids].cuda()),logits_few=m.net.backbone.head_few(z[1][ids].cuda()))
    else:o=raw(m,pointwise,pixels[ids].cuda())
    l=loss(o,y,w,n);assert torch.isfinite(l);l.backward();finite(ps);optimizer.step();finite(ps);seen+=len(ids);total+=float(l.detach())*len(ids)
    if step%32==0:
     check_space(cfg['window_root']);write(out/'status.json',dict(state='RUNNING',phase=kind,epoch=epoch+1,processed=seen,total=n,elapsed_seconds=time.monotonic()-start))
   assert seen==2555;logs.append(dict(phase=kind,epoch=epoch+1,training_exposures=seen,shuffled_loss_mean=total/n));write(out/'partial.json',dict(epochs=logs,source_commit=cfg['source_commit']))
   check_space(cfg['window_root'],reserve=32*2**20);state=dict(bias_overrides=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys),heads=[p.detach().cpu().clone() for p in heads],optimizer=optimizer.state_dict(),generator=gen.get_state(),phase=kind,epoch=epoch+1,source_commit=cfg['source_commit']);temp=out/'resume.private.pt.tmp';torch.save(state,temp);temp.replace(out/'resume.private.pt')
 run('HeadOnly')
 assert all(torch.equal(params[k],initial_bias[k]) for k in bkeys)
 with torch.no_grad():
  for p,v in zip(heads,initial_heads):p.copy_(v)
 run('Bias');assert len(logs)==20 and all(v['training_exposures']==2555 for v in logs)
 overrides=checkpoint({k:params[k].detach().cpu() for k in bkeys},ckeys);assert all(torch.equal(native(overrides)[k],params[k].detach().cpu()) for k in bkeys)
 torch.save(dict(bias_overrides=overrides,source_commit=cfg['source_commit'],initial_training_only=True),out/'encoder.private.pt');write(out/'ENCODER_SEAL.json',dict(epochs_per_control=10,bias_training_exposures=25550,headonly_training_exposures=25550,bias_parameters=102912,bias_bytes=411648,future_frozen=True,source_commit=cfg['source_commit'],test_used=False));write(out/'aggregate.json',dict(decision='FIXED_TRAINING_COMPLETED_NOT_GLOBAL_OPTIMUM',epochs=logs,source_commit=cfg['source_commit'],method_success=False));write(out/'status.json',dict(state='COMPLETE',phase='ENCODER_SEALED'))
