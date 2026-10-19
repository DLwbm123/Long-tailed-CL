"""No classifier fitting: gradient, key binding, zero restore and byte certificates."""
import os,sys,json,time,io,resource
from pathlib import Path
import torch,numpy as np
from space import check_space
IX=torch.tensor([0,1,16,17,32,33,48,49]);CAP=19058800
def write(p,x):
 p.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__))))
def main(cfg):
 begin=time.monotonic();out=Path(cfg['output']);root=Path(cfg['root']);bc=json.loads(Path(cfg['benchmark_config']).read_text());legacy=Path(bc['legacy_source']);replay=Path(cfg['prior_replay']).resolve();matched=Path(cfg['matched']).resolve();testroot=Path(bc['output']).resolve();ref=Path(cfg['reference']).resolve()
 raw=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']]
 allowed={replay/'original/pixels.private.pt',replay/'original/outputs.private.pt',matched/'transforms.private.pt'}
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  assert not any(p.is_relative_to(z) for z in raw),'raw image forbidden'
  assert not p.is_relative_to(testroot),'test forbidden'
  if p.is_relative_to(replay) or p.is_relative_to(matched):assert p in allowed,'nontraining tensor forbidden'
  if p.is_relative_to(ref):raise AssertionError('support bundle not needed')
 sys.addaudithook(guard)
 pixels=torch.load(replay/'original/pixels.private.pt',weights_only=False)[IX];assert pixels.shape==(8,3,224,224)
 check_space(cfg['window_root'],reserve=256*2**20)
 if cfg['mode']=='ORIGINAL':
  sys.path.insert(0,str(legacy/'third_party/APART'));sys.path.insert(0,str(legacy/'tools'))
  from run_multilabel import ApartFeatures
  from run_medical_v2 import seed_all
  from safetensors.torch import load_file
  lock=json.loads((legacy/'docs/real_query_node1_20261016/PROTOCOL_LOCK.json').read_text());seed_all(lock['seed']);encoder=ApartFeatures(str(legacy),bc['weight'],23,'cuda',lock['seed']).eval().requires_grad_(False)
  locked=load_file(bc['weight']);keys=sorted(k for k in locked if k.endswith('.bias') and not k.startswith('head.'));params=dict(encoder.net.backbone.named_parameters());assert keys and all(k in params and params[k].shape==locked[k].shape for k in keys)
  assert all(k.startswith(('blocks.','patch_embed.','norm.')) for k in keys);del locked
  for k in keys:params[k].requires_grad_(True)
  assert {n for n,p in encoder.net.backbone.named_parameters() if p.requires_grad}==set(keys)
  base={k:params[k].detach().clone() for k in keys};x=pixels.cuda()
  def value():return torch.nn.functional.normalize(encoder(x).double(),dim=1)
  with torch.no_grad():baseline=value();repeat=value()
  cached=torch.load(replay/'original/outputs.private.pt',weights_only=False)['cached'][IX].cuda();cacheerr=float((baseline-cached).abs().max());repeaterr=float((baseline-repeat).abs().max());assert cacheerr<=1e-5 and repeaterr<=1e-7
  d=(torch.randint(0,2,(1536,),generator=torch.Generator().manual_seed(74160))*2-1).to('cuda',torch.float64);g=value();objective=(g*d).sum()/len(g);objective.backward()
  grads={k:params[k].grad.detach().clone() for k in keys};assert all(torch.isfinite(v).all() for v in grads.values());norm=torch.sqrt(sum(v.double().square().sum() for v in grads.values()));assert norm>0
  early=any(k.startswith(tuple('blocks.'+str(i)+'.' for i in range(6))) and float(v.norm())>0 for k,v in grads.items());late=any(k.startswith(tuple('blocks.'+str(i)+'.' for i in range(6,12))) and float(v.norm())>0 for k,v in grads.items());assert early and late
  vals=[]
  with torch.no_grad():
   for sign in [-1,1]:
    for k in keys:params[k].copy_(base[k]+sign*.001*(grads[k]/norm).to(base[k].dtype))
    vals.append(float((value()*d).sum()/len(x)))
   for k in keys:params[k].copy_(base[k])
   restoreerr=float((value()-baseline).abs().max())
  fd=(vals[1]-vals[0])/.002;analytic=float(norm);fderr=abs(fd-analytic);assert fderr<=1e-4+.01*abs(analytic) and restoreerr<=1e-7
  gen=torch.Generator().manual_seed(74161);delta={k:(torch.randint(0,2,params[k].shape,generator=gen)*2-1).float()*.0001 for k in keys}
  with torch.no_grad():
   for k in keys:params[k].copy_(base[k]+delta[k].cuda())
   changed=value();change=float((changed-baseline).abs().max());assert torch.isfinite(changed).all() and change>1e-8
   for k in keys:params[k].copy_(base[k]);assert torch.equal(params[k],base[k])
  poolparams=sum(p.numel() for pool in [encoder.net.backbone.pool.pool,encoder.net.backbone.pool_few.pool] for p in pool.parameters());torch.save(dict(delta=delta,parameter_names=keys,probe_only=True),out/'bias_probe.private.pt')
  result=dict(pass_all=True,decision='ORIGINAL_GRADIENT_BINDING_PASS',mode=cfg['mode'],rows=8,classification_fits=0,optimizer_updates=0,classification_scores=0,validation_rows=0,test_rows=0,bias_tensors=len(keys),bias_parameters=sum(params[k].numel() for k in keys),bias_FP32_bytes=sum(params[k].numel()*4 for k in keys),full_pool_parameters=poolparams,full_pool_FP32_bytes=poolparams*4,cache_max_error=cacheerr,repeat_max_error=repeaterr,restore_max_error=restoreerr,synthetic_delta_g_max_change=change,gradient_L2=analytic,early_and_late_gradients=True,finite_difference=fd,finite_difference_abs_error=fderr,parameter_manifest=[dict(name=k,shape=list(params[k].shape)) for k in keys],method_success=False)
 else:
  assert cfg['mode']=='MODERN';first=json.loads((root/'original/aggregate.json').read_text());assert first['pass_all'] and json.loads((root/'original/EXIT.json').read_text())['exit_code']==0
  import timm
  from safetensors.torch import load_file
  torch.manual_seed(74002);m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);m.load_state_dict({k:v for k,v in load_file(bc['weight']).items() if not k.startswith('head.')},strict=True);m.eval().requires_grad_(False).cuda();params=dict(m.named_parameters());probe=torch.load(root/'original/bias_probe.private.pt',weights_only=False);delta=probe['delta'];keys=probe['parameter_names'];assert set(keys)==set(delta) and all(k in params and params[k].shape==delta[k].shape for k in keys);base={k:params[k].detach().clone() for k in keys}
  def h():
   z=pixels.cuda();z=m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(z))))
   for b in m.blocks[:6]:z=b(z)
   return z[:,0].float()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
   baseline=h()
   for k in keys:params[k].copy_(base[k]+delta[k].cuda())
   changed=h();change=float((changed-baseline).abs().max());assert torch.isfinite(changed).all() and change>1e-8
   for k in keys:params[k].copy_(base[k]);assert torch.equal(params[k],base[k])
   restored=h();restoreerr=float((restored-baseline).abs().max());assert torch.equal(restored,baseline)
  ts=torch.load(matched/'transforms.private.pt',weights_only=False);PERM=torch.randperm(1536,generator=torch.Generator().manual_seed(74010));sizes={};payload={}
  for arm in ['G','P','C']:
   t=ts[arm];n=t['d']+(0 if arm=='G' else 128)+1
   artifact=dict(A_upper=torch.zeros(n*(n+1)//2,dtype=torch.float64),B=torch.zeros(n,6,dtype=torch.float64),W=torch.zeros(n,6,dtype=torch.float64),mapper=t,PERM=PERM,encoder_bias_delta={k:torch.zeros_like(v) for k,v in delta.items()},alpha=.01,classes=6,metadata=dict(cap=CAP,capacity_template_not_fitted=True,bias_rule='locked_pretrained_bias_keys',source_commit=cfg['source_commit']))
   fn=out/(arm+'.capacity_template.private.pt');torch.save(artifact,fn);sizes[arm]=fn.stat().st_size;assert sizes[arm]<=CAP
   loaded=torch.load(fn,weights_only=False);assert torch.equal(artifact['A_upper'],loaded['A_upper']) and all(torch.equal(artifact['encoder_bias_delta'][k],loaded['encoder_bias_delta'][k]) for k in keys);payload[arm]=sum(v.numel()*v.element_size() for v in artifact['encoder_bias_delta'].values())
  result=dict(pass_all=True,decision='ENGINEERING_FEASIBLE_NOT_TRAINED',mode=cfg['mode'],same_delta_key_shape_binding=True,bias_parameters=first['bias_parameters'],bias_FP32_bytes=first['bias_FP32_bytes'],synthetic_delta_h_max_change=change,zero_restore_h_max_error=restoreerr,capacity_template_actual_bytes=sizes,delta_payload_in_each_template=payload,cap=CAP,template_not_fitted_artifact=True,classification_fits=0,optimizer_updates=0,classification_scores=0,validation_rows=0,test_rows=0,rows=8,method_success=False)
 result.update(source_commit=cfg['source_commit'],worker_body_wall_seconds=time.monotonic()-begin,GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(out/'aggregate.json',result)
