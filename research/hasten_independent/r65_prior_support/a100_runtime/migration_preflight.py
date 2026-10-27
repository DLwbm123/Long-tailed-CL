"""A100 migration check using the already-authorized original two TRAIN rows, zero optimizer."""
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import write,guard
from data import get_cache,Tokens,store
from vision import NativeTail,adapt_native_tail,load_delta

cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output'])
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
guard(cfg);started=time.time();dc=cfg['domains']['NIH']
paths=get_cache(cfg,dc,stage=0);data=Tokens(paths)
meta=json.loads((root/'cache/0/META.json').read_text());indices=[data.ids.index(i) for i in meta['ids']]
assert len(indices)==2 and meta['split']=='train'
engine=store();engine.prepare(data,indices);tokens=engine.batch(data,indices,True)
assert torch.equal(tokens,engine.batch(data,indices,False))
cached_pixel=engine.pixel(data,indices[0])
from PIL import Image
image=Path(cfg['image_roots']['NIH'])/data.relative_paths[indices[0]]
with Image.open(image) as im:original_pixel=engine.preprocess(im.convert('RGB'))
assert torch.equal(cached_pixel,original_pixel)
prior=[]
for c in meta['chunks']:
    a=np.memmap(root/'cache/0'/c['file'],dtype=np.float32,mode='r',shape=(c['rows'],197,768))
    prior.append(torch.from_numpy(np.array(a,copy=True)));a._mmap.close()
old=torch.cat(prior);prefix_error=float((tokens.cpu()-old).abs().max());assert prefix_error<=2e-5,prefix_error
bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False)
with torch.random.fork_rng(devices=[0]),torch.no_grad():
    model=NativeTail(bundle).cuda().eval();base=model(tokens.cuda())
    native_error=float((base.cpu()-data.base[indices]).abs().max());assert native_error<=2e-5
    params=adapt_native_tail(model);head=torch.load(root/'NIH/74003/head0.private.pt',map_location='cpu',weights_only=False);load_delta(model,head['delta'])
    other=NativeTail(bundle).cuda().eval();adapt_native_tail(other);load_delta(other,head['delta'])
    assert torch.equal(model(tokens.cuda()),other(tokens.cuda()))
    assert all(torch.equal(p.cpu(),head['delta'][n]) for n,p in model.named_parameters() if p.requires_grad)
z=model(tokens.cuda());prototype=F.normalize(data.base[indices].cuda(),dim=1).detach()
loss=F.cross_entropy(20*z@prototype.T,torch.tensor([0,1],device='cuda'));loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
assert max(float(p.grad.norm()) for p in params)>0
assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
assert torch.cuda.max_memory_reserved(0)<12*2**30 and torch.cuda.max_memory_reserved(1)<12*2**30
write(root/'A100_MIGRATION_PREFLIGHT.json',dict(passed=True,optimizer_steps=0,TRAIN_rows=2,no_VAL_TEST=True,
      source_host='my-gpu',physical_GPUs=[0,1],backend='bounded lossless FP32 RAM cache',
      prefix_max_abs=prefix_error,native_embedding_max_abs=native_error,
      lossless_preprocessed_pixels_exact=True,sealed_delta_restore_exact=True,finite_native_backward=True,original_parameters_frozen=True,original_deadline_unchanged=True,
      partial_optimizer_resume=False,software_and_real_native_checks=True,
      torch_version=torch.__version__,wall_seconds=time.time()-started))
print('A100 native migration preflight passed, zero optimization',flush=True)
