import numpy as np
import torch
from run_fair_benchmark import bootstrap
from run_b_consensus import consensus

b=torch.tensor([[2.,0.],[0.,2.],[2.,0.],[0.,2.]],dtype=torch.float64);y=torch.tensor([0,0,1,1])
out={'B':b,'same':b.clone(),'perfect':torch.eye(2,dtype=torch.float64)[y]}
r=bootstrap(out,y,['a','a','b','c'],1,100,74003)
assert r['arms']['same']['all']['delta_BA_pp_CI95']==[0.,0.]
assert r['arms']['perfect']['all']['delta_BA_pp_CI95'][0]>=0
assert r['arms']['B']['old']['group_equal_BA']==.5
# The frozen implementation must preserve nonzero-index B winners on a zero tie.
t=consensus(torch.tensor([[0.,2.]]),torch.tensor([[4.,0.]]),torch.tensor([[4.,0.]]),[1.,1.])
assert t.argmax(1).item()==1
print('PASS: paired group resampling, image/group distinction, invariant arm interval and frozen tie rule')
