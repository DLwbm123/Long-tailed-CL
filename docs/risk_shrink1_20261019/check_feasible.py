import json,os,time
from pathlib import Path
import numpy as np
from feasible_solver import initial_from_direction,constrained
start=time.time();q=np.ones((1,1));b=np.array([-.8]);qs=np.ones((1,1,1));bs=np.array([[-.2]])
x=initial_from_direction(q,b,qs,bs,np.ones(1),np.ones(1));assert 0<x[0]<.4
a,s=constrained(q,b,qs,bs,np.ones(1),set(),x)
assert s['success'] and abs(a[0]-.4)<1e-6 and s['max_normalized_violation']<=1e-7
try:constrained(q,b,qs,bs,np.ones(1),set(),np.ones(1))
except AssertionError:pass
else:raise AssertionError('Infeasible initialization accepted')
r=dict(status='PASS',engineering_solver_calls=1,diagnostic_LP_calls=0,known_optimum_verified=True,infeasible_start_rejected=True,wall_seconds=time.time()-start)
print(json.dumps(r))
if os.environ.get('CHECK_OUTPUT'):Path(os.environ['CHECK_OUTPUT']).write_text(json.dumps(r,indent=2)+'\n')
