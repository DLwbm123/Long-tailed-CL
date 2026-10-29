"""Two true arrived TRAIN pixels, synthetic labels only for native gradient engineering."""
import os,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from common import write,guard
from data import get_cache,Tokens,store
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
from decision import Controller,grpo_loss
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);guard(cfg);t=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
dc=cfg['domains']['HK'];data=Tokens(get_cache(cfg,dc,stage=0));engine=store();tx=engine.batch(data,[0,1],True);torch.set_num_threads(4)
bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);m=NativeTail(bundle).cuda().eval()
with torch.no_grad():error=float((m(tx.cuda()).cpu()-data.base[[0,1]]).abs().max());assert error<=2e-5
params=adapt_native_tail(m);frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad};z=m(tx.cuda());proto=data.base[[0,1]].cuda();loss=F.cross_entropy(20*z@proto.T,torch.tensor([0,1],device='cuda'))+.01*(z-proto).square().sum(1).mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
assert all(p.grad is None and torch.equal(p.detach().cpu(),frozen[n]) for n,p in m.named_parameters() if not p.requires_grad)
other=NativeTail(bundle).cuda().eval();adapt_native_tail(other);load_delta(other,delta_state(m))
with torch.no_grad():assert torch.equal(m(tx.cuda()),other(tx.cuda()))
c=Controller();ss=[torch.zeros(2,8) for _ in range(16)];mm=[torch.ones(3,dtype=torch.bool) for _ in range(16)];loss,stats=grpo_loss(c,ss,mm,[i%3 for i in range(16)],torch.full((8,2),-torch.tensor(3.).log()),torch.arange(8).float()-3.5);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters())
assert all(torch.cuda.max_memory_reserved(i)<12*2**30 for i in [0,1])
write(root/'NATIVE_CHECK.json',dict(passed=True,real_TRAIN_rows=2,synthetic_loss_fixture=True,native_embedding_max_abs=error,LoRA_parameters=49152,original_parameters_frozen=True,strict_restore_exact=True,finite_visual_controller_backward=True,optimizer_steps=0,VAL_reads=0,wall_seconds=time.time()-t));print('Native engineering passed; no optimizer/VAL',flush=True)
