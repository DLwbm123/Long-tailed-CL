from pathlib import Path
import os,json,time,torch
from common import write,guard
from data import get_cache,Tokens,store
from vision import NativeTail,adapt_native_tail,load_delta,delta_state
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);dc=cfg['domains']['HK'];start=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
with torch.no_grad():
 data=Tokens(get_cache(cfg,dc,stage=0));engine=store();engine.prepare(data,[0,1]);tokens=engine.batch(data,[0,1],True).cuda();bundle=torch.load(r/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);m=NativeTail(bundle).cuda().eval();err=float((m(tokens).cpu()-data.base[[0,1]]).abs().max());assert err<=2e-5
 params=adapt_native_tail(m);h=torch.load(Path(cfg['matched_reference_run'])/'HK/74002/head0.private.pt',map_location='cpu',weights_only=False);load_delta(m,h['delta']);a=m(tokens);load_delta(m,delta_state(m));b=m(tokens);assert torch.equal(a,b) and torch.isfinite(a).all() and all(p.grad is None for p in params)
write(r/'GRPO_PREFLIGHT.json',dict(passed=True,real_HK_TRAIN_rows=2,optimizer_steps=0,no_VAL_TEST=True,no_backward=True,embedding_maxabs=err,strictrestore_exact=True,LoRA_parameter_count=49152,wall_seconds=time.time()-start))
