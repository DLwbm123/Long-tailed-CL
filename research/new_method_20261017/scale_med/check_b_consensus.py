import torch
from run_b_consensus import consensus,retention
torch.set_default_dtype(torch.float64)
b=torch.tensor([[0.,2.],[2.,0.],[2.,0.],[2.,0.]])
l=torch.tensor([[4.,0.],[0.,6.],[0.,6.],[0.,4.]])
d=torch.tensor([[0.,4.],[0.,6.],[0.,-1.],[0.,4.]])
t=consensus(b,l,d,[1.,1.]);assert t.argmax(1).tolist()==[1,1,0,0]
# Includes opposing signs, true rescue/harm, exact ties and B at a nonzero index.
y=torch.tensor([1,1,0,0]);r=retention(b,{'density':b+.5*l,'distance':b+.5*d},t,y,1)
assert r['all']['density']['retained']==1 and r['all']['harm_bound']['observed']==0
g=torch.Generator().manual_seed(3)
for _ in range(30):
    b,l,d=[torch.randn(19,5,generator=g) for _ in range(3)]
    t=consensus(b,l,d,[.7,1.3]);a=b.argmax(1);changed=t.argmax(1)!=a
    for z in [l,d]:assert not (changed&((b/.7+.5*z/1.3).argmax(1)==a)).any()
print('PASS: ties retain B, shared evidence, rescue accounting, label-free harm-subset invariant')
