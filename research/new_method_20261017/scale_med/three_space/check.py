"""Small synthetic API and correctness checks; no benchmark inputs."""
from verify import selfcheck
from pyscipopt import Model
from scipy.optimize import linprog
selfcheck()
for feasible in [True,False]:
    m=Model();m.hideOutput();w=m.addVar(lb=None,ub=None);a=m.addVar(vtype='B');b=m.addVar(vtype='B')
    m.addCons(a==1);m.addConsIndicator(w<=0,binvar=a)
    m.addCons(b==int(not feasible));m.addConsIndicator(-w<=-1,binvar=b);m.optimize()
    assert str(m.getStatus())==('optimal' if feasible else 'infeasible')
r=linprog([0.],A_ub=[[1.],[-1.]],b_ub=[0.,-1.],bounds=(None,None),method='highs');assert r.status==2
print('SELFCHECK_PASS; SCIP',m.getMajorVersion(),m.getMinorVersion(),m.getTechVersion())
