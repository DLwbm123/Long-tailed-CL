"""Two actual arrived HK TRAIN pixels, zero optimization, no efficacy screen."""
from pathlib import Path
import os,json,time
import torch
import torch.nn.functional as F
from common import write,guard
from data import get_cache,Tokens,store
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
from decision import ARMS,objective,tailness,policy_logits
from check_decision import check
from support import make_support,support_scores
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;guard(cfg);started=time.time()
dc=cfg['domains']['HK'];data=Tokens(get_cache(cfg,dc,stage=0));ids=[0,1];engine=store();engine.prepare(data,ids);tokens=engine.batch(data,ids,True);assert tokens.shape==(2,197,768) and tokens.dtype==torch.float32
bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);m=NativeTail(bundle).cuda().eval()
with torch.no_grad():base=m(tokens.cuda());error=float((base.cpu()-data.base[ids]).abs().max());assert error<=2e-5
params=adapt_native_tail(m);frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad};proto=make_support(F.normalize(data.base[ids].cuda(),dim=1).detach(),torch.tensor([0,1],device='cuda'),2);labels=torch.tensor([0,1],device='cuda');counts=torch.tensor([2.,8.],device='cuda');weights=tailness(counts,torch.tensor([0,7],device='cuda'))
for arm in ARMS:
 for p in params:p.grad=None
 z=m(tokens.cuda());u=policy_logits(support_scores(z,proto,arm),counts,arm);risk,stats=objective(u,u.detach(),u.detach(),labels,weights,arm,torch.Generator(device='cuda').manual_seed(67));risk.mean().backward();assert torch.isfinite(risk).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
 assert all(p.grad is None and torch.equal(p.detach().cpu(),frozen[n]) for n,p in m.named_parameters() if not p.requires_grad)
with torch.no_grad(),torch.random.fork_rng(devices=[0]):
 other=NativeTail(bundle).cuda().eval();adapt_native_tail(other);load_delta(other,delta_state(m));assert torch.equal(m(tokens.cuda()),other(tokens.cuda()))
assert sum(p.numel() for p in params)==49152 and all(torch.cuda.max_memory_reserved(i)<12*2**30 for i in [0,1])
write(root/'GRPO_PREFLIGHT.json',dict(passed=True,optimizer_steps=0,real_HK_TRAIN_rows=2,no_VAL_TEST=True,synthetic_two_class_loss_fixture_on_real_TRAIN_pixels=True,native_embedding_max_abs=error,original_parameters_frozen=True,strictrestore_exact=True,finite_native_backward_all_four_arms=True,LoRA_parameters=49152,software=json.loads((root/"SOFTWARE_CHECK.json").read_text()),physical_GPUs=[0,1],wall_seconds=time.time()-started))
print('HK native GRPO preflight passed; zero optimization',flush=True)
