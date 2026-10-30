import os,json,time,torch
import torch.nn.functional as F
from pathlib import Path
from common import write,guard
from data import get_cache,Tokens,store
from vision import NativeTail,adapt_native_tail,load_delta
from pooling import pool,channels
from visibility import capture
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);t=time.time();guard(cfg)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
data=Tokens(get_cache(cfg,cfg['domains']['HK'],stage=0));x=store().batch(data,[0,1],True)
bundle=torch.load(r/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);m=NativeTail(bundle).cuda().eval()
with torch.no_grad():base_error=float((m(x.cuda()).cpu()-data.base[[0,1]]).abs().max())
assert base_error<=2e-5;adapt_native_tail(m)
src=torch.load(Path(cfg['source_run'])/'HK/uniform_controller/74002/head0.private.pt',map_location='cpu',weights_only=False);load_delta(m,src['delta'])
m.requires_grad_(False)
with torch.no_grad():
 cls,patch=pool(m,x.cuda());assert torch.equal(cls,m(x.cuda()))
 h=m.norm(m.blocks(x.cuda()));reference=F.normalize(torch.stack([F.normalize(m.proj(h[:,j]),dim=-1) for j in range(1,197)],dim=1).mean(1),dim=1)
 error=float((patch-reference).abs().max());assert error<=2e-5
 for z in channels(cls,patch).values():assert torch.allclose(z.norm(dim=1),torch.ones(2,device='cuda'),atol=2e-6)
assert all(p.grad is None and not p.requires_grad for p in m.parameters())
assert all(torch.equal(dict(m.named_parameters())[k].detach().cpu(),v) for k,v in src['delta'].items())
assert all(torch.cuda.max_memory_reserved(i)<12*2**30 for i in [0])
capture(r,'NATIVE')
write(r/'NATIVE_CHECK.json',dict(passed=True,real_TRAIN_pixel_rows=2,base_CLS_max_abs=base_error,
 source_CLS_original_forward_exact=True,patch_formula_max_abs=error,source_restore_exact=True,
 all_visual_parameters_frozen=True,backward_calls=0,optimizer_steps=0,VAL_TEST_reads=0,wall_seconds=time.time()-t))
