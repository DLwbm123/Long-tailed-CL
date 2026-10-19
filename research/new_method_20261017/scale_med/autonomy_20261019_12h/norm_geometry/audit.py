"""Same-weight final-CLS direction/radius audit on 64 fixed training rows."""
import json,os,sys,time,resource
from pathlib import Path
import numpy as np
import torch

def normalize(x):return torch.nn.functional.normalize(x,dim=1)
def summary(x):
 x=np.asarray(x,dtype=float);return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std()),CV=float(x.std()/abs(x.mean())) if x.mean()!=0 else None)
def main(cfg):
 start=time.monotonic();torch.set_num_threads(4);out=Path(cfg['output']);ref=Path(cfg['reference']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());src=json.loads(Path(bc['transfer_config']).read_text());root=Path(bc['datasets']['ISIC']['development_images']).resolve();testroot=Path(bc['output']).resolve();current=ref/'inputs/0.private.pt';allowed=set()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(ref):assert p==current,'other reference inputs forbidden'
  if p.is_relative_to(root):assert p in allowed,'image outside fixed training audit'
  assert not p.is_relative_to(testroot),'historical test forbidden'
 sys.addaudithook(guard)
 obj=torch.load(current,map_location='cpu',weights_only=False);tr=~obj['valid'];ix=torch.cat([torch.where(tr&(obj['y']==c))[0][:16] for c in range(4)]);assert len(ix)==64 and not obj['valid'][ix].any();rows=[obj['rows'][i] for i in ix.tolist()];allowed.update((root/r['relative_path']).resolve() for r in rows);g=obj['g'][ix].double();y=obj['y'][ix];assert torch.equal(g[:,:768],g[:,768:]);u=normalize(g[:,:768]);del obj
 for p in [ref/'inputs/1.private.pt',testroot/'ISIC.test.features.private.pt']:
  try:guard('open',(str(p),'rb'))
  except AssertionError:pass
  else:raise AssertionError('audit isolation')
 import timm
 from safetensors.torch import load_file
 from run import Images
 from torch.utils.data import DataLoader
 from space import check_space
 check_space(cfg['window_root'],reserve=2**20);torch.manual_seed(74159)
 model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);state={k:v for k,v in load_file(src['weight']).items() if not k.startswith('head.')};model.load_state_dict(state,strict=True);del state;model.eval().requires_grad_(False).cuda()
 # Synthetic row normalization identity; no labels or selection.
 z=torch.tensor([[3.,4.],[5.,12.]],dtype=torch.float64);assert (normalize(z)*z.norm(dim=1,keepdim=True)-z).abs().max()<1e-14
 all32=[];all16=[]
 for x,_ in DataLoader(Images(rows,dict(images=str(root))),batch_size=16,num_workers=0,shuffle=False):
  with torch.inference_mode(),torch.autocast('cuda',enabled=False):h=model.forward_features(x.cuda())[:,0].double().cpu()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):hh=model.forward_features(x.cuda())[:,0].double().cpu()
  assert torch.isfinite(h).all() and torch.isfinite(hh).all();all32.append(h);all16.append(hh);(out/'status.json').write_text(json.dumps(dict(state='RUNNING',phase='fixed_training_structure',rows=sum(len(a) for a in all32))))
 h=torch.cat(all32);h16=torch.cat(all16);v=normalize(h);v16=normalize(h16);d=(v-u).norm(dim=1);d16=(v16-u).norm(dim=1);rad=h.norm(dim=1);cos=(u*v).sum(1);ray=(u*rad[:,None]-h).norm(dim=1)/h.norm(dim=1);close=bool((d<=1e-4).all())
 result=dict(source_commit=cfg['source_commit'],fixed_training_images=64,each_class_images=16,validation_images=0,test_images=0,classification_fits=0,classification_scores=0,parameter_selection=False,FP32_vs_cached_global_unit_L2=summary(d.numpy()),FP16_vs_cached_global_unit_L2=summary(d16.numpy()),FP32_cosine=summary(cos.numpy()),relative_ray_reconstruction_error=summary(ray.numpy()),raw_CLS_radius=summary(rad.numpy()),radius_by_class={str(c):summary(rad[y==c].numpy()) for c in range(4)},raw_CLS_FP16_relative_error=summary(((h16-h).norm(dim=1)/h.norm(dim=1)).numpy()),fixed_direction_tolerance=1e-4,direction_close_all64=close,decision='DIRECTION_PARITY_NORM_MECHANISM_ELIGIBLE_FOR_DESIGN' if close else 'UNRESOLVED_DIRECTION_DIFFERENCE_NO_NORM_ATTRIBUTION',synthetic_identity_pass=True,access_guard_probes_pass=True,no_new_feature_cache=True,no_risk_or_method_success_claim=True,feature_forward_images=128,worker_body_wall_seconds=time.monotonic()-start,worker_CPU_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
 (out/'aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False));(out/'status.json').write_text(json.dumps(dict(state='COMPLETE',decision=result['decision'])))
