"""Synthetic engineering assertions, zero optimizer updates; no scientific screening."""
import os,json,runpy,time
from pathlib import Path
import torch
ns=runpy.run_path('/tmp/q1083/worker.py',run_name='engineering_fixture');M=ns['Metric'];F=ns['F'];cfg=ns['CFG'];root=Path(cfg['output'])
torch.manual_seed(123);torch.cuda.manual_seed_all(123);m=M().cuda();x=F.normalize(torch.randn(19,512,device='cuda'),dim=1);means=F.normalize(torch.randn(3,512,device='cuda'),dim=1);y=torch.tensor([0]*11+[1]*5+[2]*3,device='cuda');cnt=torch.bincount(y);w=len(y)/(3*cnt[y].float())
assert torch.allclose(m(x),x,atol=2e-7);assert all(torch.allclose(w[y==c].sum()/len(y),torch.tensor(1/3,device='cuda')) for c in range(3));loss=(w*F.cross_entropy(m.logits(x,means),y,reduction='none')).mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());assert m.V.grad.norm()>0
with torch.no_grad():
 m.V.copy_(torch.randn_like(m.V));m.project();assert torch.allclose(m.correction_norm(),(m.U@m.V.T).norm(),atol=2e-6);assert m.correction_norm()<=.250001
 diff=(m.U@m.V.T)@x.T;assert diff.norm(dim=0).max()<=.250001
 m2=M().cuda();m2.load_state_dict(m.state_dict(),strict=True);assert torch.equal(m.logits(x,means),m2.logits(x,means));assert sum(p.numel() for p in m.parameters())==16384
(root/'PREFLIGHT.json').write_text(json.dumps(dict(passed=True,synthetic_only=True,optimizer_steps=0,class_mass_equal=True,finite_shared_gradient=True,low_rank=16,learned_parameters=16384,per_class_parameters=0,correction_bound=.25,strict_restore=True,time=time.time()),indent=2))
