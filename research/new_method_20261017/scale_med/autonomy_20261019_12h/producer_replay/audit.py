"""Fixed engineering replay, original APART + bare ViT, two runtimes, same pixels."""
import json,os,sys,time,resource
from pathlib import Path
import torch,numpy as np

def normal(x):return torch.nn.functional.normalize(x,dim=1)
def describe(a,b):
 assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all();d=(a-b);l2=d.norm(dim=1)
 return dict(rows=len(a),max_abs=float(d.abs().max()),mean_L2=float(l2.mean()),max_L2=float(l2.max()))
def main(cfg):
 out=Path(cfg['output']);root=Path(cfg['root']);begin=time.monotonic();torch.set_num_threads(2);mode=cfg['mode']
 def write(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
 if mode=='AGGREGATE':
  for d in ['original','modern']:assert json.loads((root/d/'EXIT.json').read_text())['exit_code']==0
  a=torch.load(root/'original/outputs.private.pt',map_location='cpu',weights_only=False);b=torch.load(root/'modern/outputs.private.pt',map_location='cpu',weights_only=False);u=normal(a['cached'][:,:768]);ap=normal(a['apart'][:,:768]);old=normal(a['bare']);new=normal(b['bare']);comparisons=dict(original_APART_vs_cached=describe(a['apart'],a['cached']),original_bare_vs_cached_direction=describe(old,u),modern_bare_vs_cached_direction=describe(new,u),modern_vs_original_bare_direction=describe(new,old),original_APART_vs_original_bare_direction=describe(ap,old));cachepass=comparisons['original_APART_vs_cached']['max_abs']<=1e-5
  verdict='UNRESOLVED_HISTORICAL_CACHE_REPLAY' if not cachepass else ('HISTORICAL_CACHE_REPLAY_PASS_RUNTIME_OR_PATH_DIFFERENCE' if comparisons['modern_vs_original_bare_direction']['max_L2']>1e-4 or comparisons['original_APART_vs_original_bare_direction']['max_L2']>1e-4 else 'HISTORICAL_CACHE_REPLAY_PASS_PRIOR_DIFFERENCE_STILL_UNRESOLVED')
  result=dict(source_commit=cfg['source_commit'],decision=verdict,comparison=0,classification_fits=0,classification_scores=0,test_images=0,validation_images=0,comparisons=comparisons,versions={d:json.loads((root/d/'metadata.json').read_text()) for d in ['original','modern']},same_decoded_pixels=True,original_cache_tolerance=1e-5,prior_norm_audit_gate_unchanged=True,feature_forward_image_instances=192,method_success=False)
  write('aggregate.json',result);return
 bc=json.loads(Path(cfg['benchmark_config']).read_text());ref=Path(cfg['reference']).resolve();image_root=Path(bc['datasets']['ISIC']['development_images']).resolve();testroot=Path(bc['output']).resolve();allowed=set()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(ref):assert p==ref/'inputs/0.private.pt','other source input forbidden'
  if p.is_relative_to(image_root):assert p in allowed,'image outside fixed training set'
  assert not p.is_relative_to(testroot),'test forbidden'
 sys.addaudithook(guard)
 if mode=='ORIGINAL':
  obj=torch.load(ref/'inputs/0.private.pt',map_location='cpu',weights_only=False);ix=torch.cat([torch.where((~obj['valid'])&(obj['y']==c))[0][:16] for c in range(4)]);assert len(ix)==64 and not obj['valid'][ix].any();rows=[obj['rows'][i] for i in ix.tolist()];cached=obj['g'][ix].double();allowed.update((image_root/r['relative_path']).resolve() for r in rows);del obj
  legacy=Path(bc['legacy_source']);sys.path.insert(0,str(legacy/'third_party/APART'));sys.path.insert(0,str(legacy/'tools'))
  from run_multilabel import ApartFeatures
  from run_prototype_single import Images
  from run_medical_v2 import transform,seed_all
  lock=json.loads((legacy/'docs/real_query_node1_20261016/PROTOCOL_LOCK.json').read_text());seed_all(lock['seed']);encoder=ApartFeatures(str(legacy),bc['weight'],23,'cuda',lock['seed']).eval().requires_grad_(False)
  from torch.utils.data import DataLoader
  pixels=torch.cat([x for x,_ in DataLoader(Images(rows,str(image_root),transform(False),lock['seed']),batch_size=32,num_workers=0,shuffle=False)]);assert pixels.shape==(64,3,224,224);torch.save(pixels,out/'pixels.private.pt');values=[]
  with torch.no_grad():
   for x in pixels.split(32):values.append(normal(encoder(x.cuda()).cpu().double()))
  apart=torch.cat(values);assert apart.shape==(64,1536);del encoder;torch.cuda.empty_cache()
 else:
  assert mode=='MODERN';pixels=torch.load(root/'original/pixels.private.pt',map_location='cpu',weights_only=False)
 import timm,PIL,torchvision
 from safetensors.torch import load_file
 model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);model.load_state_dict({k:v for k,v in load_file(bc['weight']).items() if not k.startswith('head.')},strict=True);model.eval().requires_grad_(False).cuda();values=[]
 with torch.no_grad(),torch.autocast('cuda',enabled=False):
  for x in pixels.split(32):values.append(model.forward_features(x.cuda())[:,0].cpu().double())
 bare=torch.cat(values);data=dict(bare=bare)
 if mode=='ORIGINAL':data.update(cached=cached,apart=apart)
 torch.save(data,out/'outputs.private.pt');write('metadata.json',dict(source_commit=cfg['source_commit'],python=sys.version.split()[0],torch=torch.__version__,torchvision=torchvision.__version__,timm=timm.__version__,Pillow=PIL.__version__,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,batch_size=32,rows=64,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),worker_body_wall_seconds=time.monotonic()-begin,worker_CPU_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),training_only=True))
