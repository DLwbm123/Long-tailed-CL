import json,os,time
from pathlib import Path
import torch
from common import write,guard
from data import get_cache,Tokens,store
from vision import NativeTail,adapt_native_tail,load_delta
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);t=time.time();guard(cfg)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
data=Tokens(get_cache(cfg,cfg['domains']['HK'],stage=0));x=store().batch(data,[0,1],True)
bundle=torch.load(r/'cache/0/tail.private.pt',map_location='cpu',weights_only=False)
m=NativeTail(bundle).cuda().eval()
with torch.no_grad():err=float((m(x.cuda()).cpu()-data.base[[0,1]]).abs().max())
assert err<=2e-5;adapt_native_tail(m)
state=torch.load(Path(cfg['source_run'])/'HK/uniform_controller/74002/head0.private.pt',map_location='cpu',weights_only=False)
load_delta(m,state['delta'])
with torch.no_grad():first=m(x.cuda()).cpu();load_delta(m,state['delta']);second=m(x.cuda()).cpu()
assert torch.equal(first,second);assert all(p.grad is None for p in m.parameters())
assert all(int(v['step'])==24 for v in state['optimizer_state']['state'].values())
assert all(torch.cuda.max_memory_reserved(i)<12*2**30 for i in [0,1])
write(r/'NATIVE_CHECK.json',dict(passed=True,real_TRAIN_rows=2,native_embedding_max_abs=err,
       strict_source_restore=True,backward_calls=0,visual_optimizer_steps=0,head_fits=0,VAL_reads=0,
       wall_seconds=time.time()-t))
