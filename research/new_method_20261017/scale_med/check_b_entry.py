import torch
from run_b_entry import entry_scores
from run_b_consensus import retention
torch.set_default_dtype(torch.float64)
b=torch.tensor([[0.,2.,0.],[2.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
d=torch.tensor([[3.,0.,0.],[0.,4.,1.],[1.9,2.,0.],[0.,2.,0.]])
v=torch.tensor([[0.,2.,0.],[0.,1.,5.],[1.9,0.,2.],[3.,0.,0.]])
s,gate=entry_scores(b,d,v)
assert gate.tolist()==[True,False,False,True] and s.argmax(1).tolist()==[1,2,0,1]
# An eligible equal mean can return to B; no forced choice of a non-B class.
y=torch.tensor([1,2,0,1]);r=retention(b,{'density':d,'distance':v},s,y,1)
assert r['all']['harm_bound']['observed']==0
g=torch.Generator().manual_seed(11)
for _ in range(30):
    b,d,v=[torch.randn(21,5,generator=g) for _ in range(3)]
    s,gate=entry_scores(b,d,v);changed=s.argmax(1)!=b.argmax(1)
    assert not (changed&gate).any()
print('PASS: either-branch veto, equal averaging, eligible B return, harm subset')
