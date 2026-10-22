"""Small synthetic checks of split leakage and temperature/metric semantics."""
import torch
from design import make_plan, validate, EXPECTED, CLASSES, score, select_temperature, TEMPERATURES, flows
rows=[dict(label=c,identity_component=f'c{c}_g{i}',relative_path=f'{c}/{i}') for c,n in enumerate(EXPECTED) for i in range(n)]
p=make_plan(rows);validate(p)
assert p==make_plan(rows)
bad=make_plan(rows);bad['folds'][0]['inner'][0]['train'].append(bad['folds'][0]['test'][0])
try:validate(bad)
except AssertionError:pass
else:raise AssertionError('Leakage not detected')
y=torch.tensor(CLASSES);gs=list(range(12));n=torch.zeros(12,13);a=torch.nn.functional.one_hot(y,13).float()*.9+.1/13
temp, losses=select_temperature(n,a,y,gs);assert temp==1. and len(losses)==7
for t in TEMPERATURES:
    mix=.5*(a+(n/t).softmax(1));s=score(mix,y,gs);assert s['BA']==1 and s['recall'][1] is None
    assert torch.equal(mix,.5*(a+n.softmax(1)))
f=flows(n.softmax(1),a,a,y)
assert sum(r['harmed_vs_adapter'] for r in f)==0
print('PASS: group leakage, singleton exclusion, complete OOF coverage, temperature tie, balanced score, flow semantics')
