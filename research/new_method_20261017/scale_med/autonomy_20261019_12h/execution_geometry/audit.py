"""Fixed pixel / batch / autograd-context / AMP-schedule engineering isolation."""
import json,os,sys,time,resource
from pathlib import Path
import torch

def unit(x):return torch.nn.functional.normalize(x,dim=1)
def compare(a,b):
 assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all();d=(a-b).abs();l=(unit(a)-unit(b)).norm(dim=1)
 return dict(max_abs=float(d.max()),mean_unit_L2=float(l.mean()),max_unit_L2=float(l.max()),rows_above_1e_minus4=int((l>1e-4).sum()))
def main(cfg):
 out=Path(cfg['output']);start=time.monotonic();torch.set_num_threads(4);base=Path(cfg['prior_replay']);ref=Path(cfg['reference']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());root=Path(bc['datasets']['ISIC']['development_images']).resolve();testroot=Path(bc['output']).resolve();allowed=set()
 def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(ref):assert p==ref/'inputs/0.private.pt','future/reference payload blocked'
  if p.is_relative_to(root):assert p in allowed,'outside fixed64 images'
  assert not p.is_relative_to(testroot),'test forbidden'
 sys.addaudithook(guard)
 z=torch.tensor([[3.,4.],[5.,12.]],dtype=torch.float64);assert compare(z,z)['max_unit_L2']==0 and compare(z,z.flip(1))['max_unit_L2']>1e-4
 obj=torch.load(ref/'inputs/0.private.pt',map_location='cpu',weights_only=False);ix=torch.cat([torch.where((~obj['valid'])&(obj['y']==c))[0][:16] for c in range(4)]);assert len(ix)==64 and not obj['valid'][ix].any();rows=[obj['rows'][i] for i in ix.tolist()];g=obj['g'][ix].double();allowed.update((root/r['relative_path']).resolve() for r in rows);del obj
 for p in [ref/'inputs/1.private.pt',testroot/'ISIC.test.features.private.pt']:
  try:guard('open',(str(p),'rb'))
  except AssertionError:pass
  else:raise AssertionError('isolation probe failed')
 from run import Images
 from torch.utils.data import DataLoader
 locked=torch.load(base/'original/pixels.private.pt',map_location='cpu',weights_only=False);fresh=torch.cat([x for x,_ in DataLoader(Images(rows,dict(images=str(root))),batch_size=16,num_workers=0,shuffle=False)]);assert fresh.shape==locked.shape==(64,3,224,224);pxdiff=(fresh-locked).abs();pixel=dict(exact=torch.equal(fresh,locked),max_abs=float(pxdiff.max()),different_rows=int((pxdiff.flatten(1).max(1).values>0).sum()))
 # Decode disparity is kept; GPU comparisons all consume exactly the locked tensor.
 del fresh,pxdiff
 import timm
 from safetensors.torch import load_file
 torch.manual_seed(74159);model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);model.load_state_dict({k:v for k,v in load_file(bc['weight']).items() if not k.startswith('head.')},strict=True);model.eval().requires_grad_(False).cuda();outputs={}
 schedules=[('NO_GRAD32',32,False,False),('INFERENCE32',32,True,False),('INFERENCE16',16,True,False),('INTERLEAVED16',16,True,True)]
 for name,batch,infer,amp in schedules:
  vals=[];half=[]
  for x in locked.split(batch):
   with (torch.inference_mode() if infer else torch.no_grad()),torch.autocast('cuda',enabled=False):h=model.forward_features(x.cuda())[:,0].double().cpu()
   vals.append(h)
   if amp:
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):hh=model.forward_features(x.cuda())[:,0].double().cpu()
    half.append(hh)
  outputs[name]=torch.cat(vals)
  if amp:outputs[name+'_FP16']=torch.cat(half)
  torch.save(outputs,out/'outputs.private.pt');write('status.json',dict(state='RUNNING',completed_schedules=list(outputs)))
 prior=torch.load(base/'modern/outputs.private.pt',map_location='cpu',weights_only=False)['bare'];u=unit(g[:,:768]);comp={name+'_vs_cache':compare(unit(v),u) for name,v in outputs.items()};comp.update(no_grad32_vs_prior32=compare(outputs['NO_GRAD32'],prior),context32=compare(outputs['INFERENCE32'],outputs['NO_GRAD32']),batch16_vs32=compare(outputs['INFERENCE16'],outputs['INFERENCE32']),AMP_schedule_FP32=compare(outputs['INTERLEAVED16'],outputs['INFERENCE16']))
 decisions=[]
 if not pixel['exact']:decisions.append('DECODE_TENSOR_DIFFERENCE')
 for label in ['context32','batch16_vs32','AMP_schedule_FP32']:
  if comp[label]['max_unit_L2']>1e-4:decisions.append(label.upper()+'_DIFFERENCE')
 result=dict(source_commit=cfg['source_commit'],decision=';'.join(decisions) if decisions else 'PRIOR_MISMATCH_NOT_REPRODUCED_UNRESOLVED',pixels=pixel,comparisons=comp,training_images=64,validation_images=0,test_images=0,classification_fits=0,classification_scores=0,forward_image_instances=320,schedules=[s[0] for s in schedules],tolerance=1e-4,no_tolerance_selection=True,prior_norm_audit_gate_unchanged=True,synthetic_comparison_pass=True,access_guard_probes_pass=True,method_success=False,torch_version=torch.__version__,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),body_wall_seconds=time.monotonic()-start,body_CPU_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]))
 write('aggregate.json',result);write('status.json',dict(state='COMPLETE',decision=result['decision']))
