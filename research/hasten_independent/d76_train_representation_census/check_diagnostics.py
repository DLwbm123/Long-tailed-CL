import json,torch
import torch.nn.functional as F
from diagnostics import summarize,h_snapshot

def check():
 z=F.normalize(torch.tensor([[1.,.2],[.9,.3],[.1,1.],[.2,.9]],dtype=torch.double),dim=1);y=torch.tensor([0,0,1,1]);f=torch.tensor([0,1,0,1]);r,m=summarize(z,y,f,2)
 for c in range(2):
  x=z[y==c];assert abs(r[c]['unbiased_within_trace']-float(x.var(dim=0,correction=1).sum()))<1e-12;assert abs(r[c]['mean_distinct_pair_cosine']-float(x[0]@x[1]))<1e-12
  assert r[c]['raw']['correct']==2 and r[c]['fold_counts']==[1,1]
 perm=torch.tensor([3,0,2,1]);v,n=summarize(z[perm],y[perm],f[perm],2);assert torch.allclose(m['sums'],n['sums'],atol=1e-12)
 h=h_snapshot(torch.tensor([0.,1.],dtype=torch.double));assert h[0]==0 and torch.isfinite(h).all() and abs(float(h[1])-8*(7/8)*(1/8)/((7/8*1/8)**.5+1e-8))<1e-12
 return dict(passed=True,images=0,optimizer_steps=0,VAL_TEST_reads=0,within_trace_matches_direct_variance=True,pair_cosine_identity=True,permutation_invariant_moments=True,opposite_fold_counts=True,finite_probabilities=True,h_endpoint_safe=True)
if __name__=='__main__':print(json.dumps(check(),indent=2))
