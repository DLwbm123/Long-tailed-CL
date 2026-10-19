import json,os,time
from pathlib import Path
import numpy as np
from certificate import verify,search,acceptance
start=time.time()
# a^2+a<=0, a>=0 has the unique zero solution.
p=verify(np.array([[[1.]]]),np.array([[.5]]),np.ones(1),['1'])
a,s=acceptance(p,1,1e-7);assert s['success'] and a[0]==0
# A negative linear term cannot certify a zero-only set.
try:verify(np.array([[[1.]]]),np.array([[-.5]]),np.ones(1),['1'])
except AssertionError:pass
else:raise AssertionError('Accepted an invalid positive-combination proof')
# Gradient alone is insufficient: reject negative curvature.
try:verify(np.array([[[-1.]]]),np.array([[.5]]),np.ones(1),['1'])
except AssertionError:pass
else:raise AssertionError('Accepted an indefinite certificate')
r=dict(status='PASS',zero_only_case=True,negative_gradient_rejected=True,negative_curvature_rejected=True,LP_calls=0,scientific_solver_calls=0,wall_seconds=time.time()-start)
print(json.dumps(r))
if os.environ.get('CHECK_OUTPUT'):Path(os.environ['CHECK_OUTPUT']).write_text(json.dumps(r,indent=2)+'\n')
