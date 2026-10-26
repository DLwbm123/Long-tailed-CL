"""CPU counterexample for the frozen upstream implementation; no training."""
import os,sys,json
from pathlib import Path
import torch
official=Path(os.environ['HASTEN_OFFICIAL_ROOT'])
sys.path.insert(0,str(official))
from utils.inc_net import LearnableLorentzExp
torch.set_num_threads(2);torch.manual_seed(32)
d=16;Q,_=torch.linalg.qr(torch.randn(d,d,dtype=torch.float64))
S=torch.tensor([40.,30.,20.,10.]+[.01]*12,dtype=torch.float64)
C=Q@torch.diag(S)@Q.T
m=LearnableLorentzExp(dim=d,hidden_dim=d).double();m.accumulated_cov=C;m.compute_null_space(.95)
B=m.null_space_basis;P=B@B.T
_,s,V=torch.svd(C)
rank=min(int(torch.nonzero(torch.cumsum(s*s,0)/torch.sum(s*s)>=.95)[0])+1,d-10)
protected=V[:,:rank];reference=V[:,rank:];referenceP=reference@reference.T
G=torch.randn(d,d,dtype=torch.float64);X=protected.T
out=dict(rank=rank,official_overlap=float(torch.linalg.norm(protected.T@B)),reference_overlap=float(torch.linalg.norm(protected.T@reference)),official_old_input_change=float(torch.linalg.norm(X@(P@G).T)),correct_basis_left_multiply_change=float(torch.linalg.norm(X@(referenceP@G).T)),reference_right_multiply_change=float(torch.linalg.norm(X@(G@referenceP).T)))
assert out['reference_overlap']<1e-10 and out['reference_right_multiply_change']<1e-10
assert out['official_overlap']>.1 and out['official_old_input_change']>.1
print(json.dumps(out,indent=2))
