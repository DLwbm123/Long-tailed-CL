import ast
from pathlib import Path
import torch
from teacher import teacher_prob
ns=dict(torch=torch,teacher_prob=teacher_prob)
tree=ast.parse((Path(__file__).parent/'worker.py').read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize'];exec(compile(tree,'<summary-selfcheck>','exec'),ns)
x=torch.eye(3,dtype=torch.double);w=torch.cat([torch.eye(3,dtype=torch.double),torch.zeros(1,3,dtype=torch.double)])
rows=ns['summarize'](x,torch.tensor([0,2,1]),w,3,'synthetic')
assert sum(r['count'] for r in rows)==3 and sum(r['eligible'] for r in rows)==1
rows=ns['summarize'](x,torch.arange(3),torch.zeros_like(w),3,'uniform')
assert sum(r['eligible'] for r in rows)==0
print('consistency mask, uniform rejection and full denominators passed')
