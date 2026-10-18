import torch
from run_transfer import aggregate,paired,query_allowed

s=torch.tensor([[3.,1.],[1.,3.],[3.,1.],[1.,3.]],dtype=torch.float64)
y=torch.tensor([0,0,1,1]);m=aggregate(s,y,1)
assert m['summary']['all']['BA']==.5 and m['summary']['old']['BA']==.5 and m['summary']['new']['BA']==.5
assert m['confusion']==[[1,1],[1,1]]
t=s.clone();t[1]=torch.tensor([3.,1.]);t[3]=torch.tensor([3.,1.]);p=paired(s,t,y,1)
assert p['all']['rescue']==1 and p['all']['harm']==1 and p['old']['rescue']==1 and p['new']['harm']==1
assert all(query_allowed(False,r) for r in ['S_old','S_new'])
assert not query_allowed(False,'Qfit') and not query_allowed(False,'Qgate')
assert query_allowed(True,'Qfit') and query_allowed(True,'Qgate')
assert not query_allowed(True,'E') and not query_allowed(True,'test')
try:aggregate(torch.full_like(s,float('nan')),y,1)
except AssertionError:pass
else:raise AssertionError('Nonfinite metrics accepted')
print('PASS: macro/group metrics, confusion, paired rescue/harm, model-seal query boundary, E/test exclusion')
