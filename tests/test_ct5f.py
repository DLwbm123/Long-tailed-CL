"""Boundary and paired metric check; runs on CPU without constructing a learner."""
import ast
from pathlib import Path
import numpy as np
from report_ct5f import recall_groups,stage_order
from ct2d_math import metrics
# Load only the small stage guard from the GPU runner: this check does not initialize it.
tree=ast.parse((Path(__file__).parents[1]/'tools/run_ct5f.py').read_text());ns={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bounds'],type_ignores=[]),'bounds','exec'),ns)
hk=[{'seen':c} for c in list(range(2,21,2))+[23]]
assert ns['bounds'](hk,11)==(20,23) and ns['bounds'](hk,1)==(0,2)
assert ns['bounds']([{'seen':c} for c in (2,4,6,8)],4)==(6,8)
# Three final current classes: using seen-2 must fail this expectation.
rec=np.array([[100.]*20+[0.,50.,100.]])
g=recall_groups(rec,20,np.arange(23),[20,21,22]);assert g['old'][0]==100 and g['current'][0]==50 and g['tail'][0]==50
assert abs(g['HM'][0]-200/3)<1e-10
order=np.array([3,0,2,1]);y=np.arange(4);scores=np.eye(4);m,pc,_=metrics(scores,y,order,2,{'tail':[1,3]}, {},np.array(['a','b','c','d']))
assert m['balanced_accuracy']==100 and [x['original_label'] for x in pc]==[3,0,2,1]
print('PASS: full-stage boundaries, three-class current metrics, remapped labels')

p=dict(raw=np.eye(2),y=np.array([0,1]),order=np.array([3,0,2,1]),original=np.array([3,0]))
assert np.array_equal(stage_order(p,2),[3,0])
try:stage_order(dict(p,original=np.array([0,3])),2)
except AssertionError:pass
else:raise AssertionError('incorrect label mapping accepted')
print('PASS: full arrival order compatibility and label guard')
