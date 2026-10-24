"""Synthetic engineering validation only; no real data or scientific optimizer."""
import os,json,runpy,math
from pathlib import Path
import torch
import torch.nn.functional as F
ns=runpy.run_path(str(Path(__file__).with_name('worker.py')),run_name='definitions');Map=ns['Map'];torch.manual_seed(7)
x=F.normalize(torch.randn(17,512,device='cuda'),dim=1);y=torch.tensor([0]*9+[1]*5+[2]*3,device='cuda');nc=torch.bincount(y);m=Map().cuda();u,d=m(x);assert torch.equal(u,F.normalize(x,dim=1)) and float(d.abs().max())==0
head=torch.nn.Parameter(torch.stack([x[y==c].mean(0) for c in range(3)]));loss=(17/(3*nc[y].float())*F.cross_entropy(20*u@F.normalize(head,dim=1).T,y,reduction='none')).mean()+.01*d.square().sum(1).mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in list(m.parameters())+[head]);assert m.b.weight.grad.norm()>0
z=torch.cat([F.relu(x@torch.randn(512,16,device='cuda'))*math.sqrt(2/16),torch.ones(17,1,device='cuda')],1).double();w=1/nc[y].double();G=z.T@(w[:,None]*z);Q=z.T@(w[:,None]*F.one_hot(y,3));g=torch.zeros_like(G);q=torch.zeros_like(Q)
for c in range(3):
 a=z[y==c];g+=a.T@a/int(nc[c]);q[:,c]=a.mean(0)
assert torch.allclose(g,G,atol=1e-12) and torch.allclose(q,Q,atol=1e-12);mass=[float(w[y==c].sum()) for c in range(3)];assert max(abs(a-1) for a in mass)<1e-12
A=G+.01*G.trace()/len(G)*torch.eye(len(G),device='cuda');W=torch.linalg.solve(A,Q);err=float((A@W-Q).norm()/Q.norm());assert err<1e-10
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());Path(cfg['output'],'PREFLIGHT.json').write_text(json.dumps(dict(passed=True,synthetic_only=True,real_optimizer_steps=0,VAL_reads=0,TEST_reads=0,global_class_mass=mass,split_batch_equivalent=True,ridge_residual=err,finite_actual_architecture_backward=True)))
print('SYNTHETIC_PREFLIGHT_PASSED')
