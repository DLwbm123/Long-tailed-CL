"""Actual-data qualification: 32 images per arm, backward but zero optimizer steps."""
import time
START=time.time()
import os,json,gc,resource
from pathlib import Path
import torch
from common import Model,Images,rows,seed_all,write,lr_multiplier
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());torch.set_num_threads(cfg['CPU_threads']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;data,audit=rows(cfg);root=Path(cfg['output']).parent;probe=root/'probe.private.tmp'
with probe.open('xb') as f:f.write(b'probe-ok\n')
assert probe.read_bytes()==b'probe-ok\n';probe.unlink();records=[];heads={};prior=torch.tensor(__import__('numpy').log(__import__('numpy').array(cfg['train_counts'])/68058),dtype=torch.float32,device='cuda')
for arm in cfg['arms']:
 seed_all(cfg['seed']);backbone=arm.split('_')[0];full='_full_' in arm;ds=Images(data['train'],backbone,True);xx=torch.stack([ds[i][0] for i in range(cfg['micro_batch_size'])]).cuda();yy=torch.tensor([ds[i][1] for i in range(cfg['micro_batch_size'])],device='cuda');model=Model(cfg,backbone).cuda();initial=model.weight.detach().cpu()
 if backbone in heads:assert torch.equal(initial,heads[backbone])
 else:heads[backbone]=initial.clone()
 for p in model.visual.parameters():p.requires_grad_(full)
 model.visual.train(full)
 with torch.autocast('cuda',dtype=torch.bfloat16):score=model(xx);loss=torch.nn.functional.cross_entropy(score+prior,yy)
 loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);assert score.shape==(cfg['micro_batch_size'],20) and torch.isfinite(loss) and model.weight.grad is not None;assert all(p.grad is None for p in model.visual.parameters()) if not full else any(p.grad is not None for p in model.visual.parameters());records.append(dict(arm=arm,loss=float(loss.detach()),gradient_norm=float(grad),GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)));del model,xx,yy,ds;gc.collect();torch.cuda.empty_cache()
assert 0<lr_multiplier(0,15960)<1 and abs(lr_multiplier(15959,15960)-.1)<1e-12
result=dict(state='PASS',arms=records,engineering_seconds=time.time()-START,engineering_image_exposures=cfg['micro_batch_size']*4,optimizer_updates=0,data_audit=audit,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved());write(root/'engineering_check.private.json',result);print(json.dumps(result),flush=True)
