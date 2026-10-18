import torch
from run_b_local import diagnostics
from run_v2 import fuse

b=torch.tensor([[2.,0.,-1.],[0.,2.,-1.],[2.,0.,-1.],[2.,0.,-1.]],dtype=torch.float64)
l=torch.tensor([[0.,6.,-1.],[6.,0.,-1.],[0.,1.,-1.],[0.,1.,-1.]],dtype=torch.float64)
y=torch.tensor([0,0,0,2]);f=fuse(b,l,[1.,1.],.5)
cs,summary,rows=diagnostics(b,l,f,y)
assert cs[0]['rescue']==1 and cs[0]['harm']==1
assert summary['branch_disagreement']['count']==4 and summary['fusion_changed']['count']==2
assert rows[0]['B_true_margin']==2 and rows[3]['B_true_margin']==-3
assert rows[0]['B_B_vs_local']==2 and rows[0]['local_B_vs_local']==-6
assert summary['rescue']['margins']['fusion_true_margin']['median']==1
assert sum(c['both_wrong'] for c in cs)==1
print('PASS: paired class counts, disagreement sets, signed margins, empty classes, fusion')
