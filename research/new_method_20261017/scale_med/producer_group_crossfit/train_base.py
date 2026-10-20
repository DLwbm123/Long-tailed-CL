import os,sys,json,time
from pathlib import Path
import torch
from binding import native,checkpoint
from space import check_space

def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def setup(cfg):
 import subprocess
 free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits","--id=0"],text=True).strip());assert free>=8192,"GPU0_FREE_BELOW_FROZEN_LAUNCH_MARGIN"
 receipt=Path(cfg["output"])/"GPU_AVAILABILITY.json";items=json.loads(receipt.read_text()) if receipt.exists() else [];items.append(dict(GPU=0,free_MiB=free,minimum_MiB=8192));receipt.write_text(json.dumps(items))
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
