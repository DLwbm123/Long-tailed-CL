"""Small engineering qualification; never reads experimental validation or OOF data."""
import json,time
import numpy as np
from conic import constrained,factor
start=time.time()
a,audit=constrained(np.array([[1.]]),np.array([-.8]),np.array([[[1.]]]),np.array([[-.2]]),np.ones(1),set())
assert audit['success'] and abs(a[0]-.4)<1e-6 and audit['max_normalized_violation']<=1e-7
z,za=constrained(np.eye(2),np.array([-.8,-.8]),np.array([np.eye(2)]),np.array([[-.2,-.2]]),np.ones(1),{0})
assert za['success'] and z[0]==0 and abs(z[1]-.4)<1e-6
rng=np.random.default_rng(74002)
for k in [1,3,14]:
 t=rng.normal(size=(k,k));q=t.T@t/100;b=rng.normal(size=k)/10;l,_=factor(q)
 for _ in range(10):
  x=rng.random(k);v=-2*b@x;s=np.r_[1+v,2*l@x,1-v]
  assert abs((s[0]**2-np.dot(s[1:],s[1:]))+4*(x@q@x+2*b@x))<1e-10
print(json.dumps(dict(status='PASS',engineering_solver_calls=2,wall_seconds=time.time()-start,known_solution=a.tolist(),fixed_zero_solution=z.tolist(),quadratic_SOC_identity_verified=True)))
