"""Two analytic radius cases, pinned entailment function, no training or data."""
import os,json,sys,time
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(2)
sys.path.insert(0,cfg['official']);from utils.ort import compute_stable_entailment_loss
records=[]
for radius in [.1,.4]:
 for dtype in [torch.float32,torch.float64]:
  p=torch.tensor([[radius,0.,0.]],dtype=dtype,requires_grad=True);c=torch.tensor([[.6,.2,0.]],dtype=dtype,requires_grad=True);curv=torch.tensor(1.,dtype=dtype)
  loss=compute_stable_entailment_loss(p,c,curv);g=torch.autograd.grad(loss,[p,c]);finite=all(bool(torch.isfinite(x).all()) for x in g)
  records.append(dict(radius=radius,dtype=str(dtype),loss=float(loss),finite_loss=bool(torch.isfinite(loss)),finite_gradients=finite,gradient_norm=None if not finite else float(torch.cat([x.flatten() for x in g]).norm())))
assert all(r['finite_loss'] for r in records)
supported=not records[0]['finite_gradients'] and records[1]['finite_gradients'] and records[2]['finite_gradients'] and records[3]['finite_gradients']
result=dict(expected_pattern_supported=supported,records=records,float32_upper_clamp_equals_one=float(torch.tensor(1-1e-8,dtype=torch.float32))==1.,float64_upper_clamp_equals_one=float(torch.tensor(1-1e-8,dtype=torch.float64))==1.,hypothesis='FP32 aperture clamp endpoint rounds to 1; test whether this yields nonfinite full loss gradients',r25_exact_failure_root_cause_proven=False,synthetic_only=True,optimizer_steps=0,dataset_reads=0,val_reads=0,no_test=True)
(out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False));(out/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE',time=time.time())))
