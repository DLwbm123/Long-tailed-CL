import os,json,time,copy
from pathlib import Path
import torch,numpy as np
from core import Model
from verify import numpy_forward
start=time.time();cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent
torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
x=torch.randn(8,1536,device='cuda');model=Model(1536,128,x[:3]).cuda()
_,score=model(x,25.);loss=torch.nn.functional.cross_entropy(score,torch.arange(8,device='cuda')%3);loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters())
torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']).step()
state={k:v.detach().cpu().double().numpy() for k,v in model.state_dict().items()}
with torch.no_grad():exact=copy.deepcopy(model).cpu().double()(x.cpu().double(),25.)[1].numpy()
error=float(abs(exact-numpy_forward(x.cpu().numpy(),state,25.)[1]).max());assert error<=cfg['score_absolute_tolerance']
r=dict(state='PASS',synthetic_GPU_FP32_updates=1,NIH_encodes=0,validation_encodes=0,score_error=error,seconds=time.time()-start,GPU_reserved_bytes=torch.cuda.max_memory_reserved(),original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch'])
(root/'CUDA_PREFLIGHT.private.json').write_text(json.dumps(r,indent=2))
