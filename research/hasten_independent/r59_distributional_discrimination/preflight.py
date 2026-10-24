"""Synthetic full objective/backward/LOO/restore checks; no optimizer updates."""
import runpy,torch,json,time
from pathlib import Path
s=runpy.run_path('/tmp/q1085/worker.py',run_name='engineering_fixture');M=s['Metric'];F=s['F'];f=s['distribution_logits'];cfg=s['CFG'];root=Path(cfg['output']);torch.manual_seed(123)
x=F.normalize(torch.randn(12,512,device='cuda'),dim=1);y=torch.tensor([0]*3+[1]*4+[2]*5,device='cuda');nc=torch.bincount(y);ids=torch.tensor([0,4,8],device='cuda');m=M().cuda();k=m(x);z=f(k[ids],k,nc,ids,y[ids]);assert torch.isfinite(z).all();assert torch.allclose(m.weights(),torch.ones(512,device='cuda')) and torch.allclose(k,x,atol=2e-7)
raw=20*k[ids]@k.T
for i,j in enumerate(ids.tolist()):
 for c in range(3):
  use=torch.where(y==c)[0];use=use[use!=j];v=torch.logsumexp(raw[i,use],0)-torch.tensor(float(len(use)),device='cuda').log();assert torch.allclose(v,z[i,c],atol=2e-6)
loss=F.cross_entropy(z,y[ids])+m.regularizer();loss.backward();assert m.a.grad is not None and torch.isfinite(m.a.grad).all() and m.a.grad.norm()>0;assert abs(float(m.a.grad.sum()))<1e-5
w=len(y)/(3*nc[y].float());assert all(torch.allclose(w[y==c].sum()/len(y),torch.tensor(1/3,device='cuda')) for c in range(3));m2=M().cuda();m2.load_state_dict(m.state_dict(),strict=True);assert torch.equal(f(m(x[ids]),m(x),nc,ids,y[ids]),f(m2(x[ids]),m2(x),nc,ids,y[ids]));assert sum(p.numel() for p in m.parameters())==512
# Production-size synthetic backward, no real TRAIN/VAL and no optimizer.
del x,k,z,raw,loss,m2
big=F.normalize(torch.randn(68058,512,device='cuda'),dim=1);by=torch.arange(68058,device='cuda')%20;perm=by.argsort(stable=True);big=big[perm];by=by[perm];bc=torch.bincount(by);qi=torch.arange(256,device='cuda');m.zero_grad(set_to_none=True);kk=m(big);lz=f(kk[qi],kk,bc,qi,by[qi]);F.cross_entropy(lz,by[qi]).backward();assert torch.isfinite(m.a.grad).all() and torch.cuda.max_memory_reserved()<8*2**30
(root/'PREFLIGHT.json').write_text(json.dumps(dict(passed=True,synthetic_only=True,optimizer_steps=0,manual_LOO_logits_exact=True,self_excluded=True,correct_class_denominators=True,finite_shared_gradient=True,shared_parameters=512,no_perclass_parameters=True,strict_restore=True,time=time.time()),indent=2))
