import json,time,numpy as np
START=time.time()
from control import inputs,policy
b=np.array([[2.,0.,1.],[0.,2.,1.],[2.,0.,1.]])
f=np.array([[0.,2.,1.],[2.,0.,1.],[2.,0.,1.]])
c=b.copy();old=np.zeros((3,5));counts=np.array([[2,4,3],[2,4,3],[2,4,3]])
x,z=inputs(old,b,f,c,counts,[1]);assert x.shape==(3,5) and z.shape==(3,9);assert np.allclose(x[:2,4],[np.log(5)-np.log(3),np.log(3)-np.log(5)]);assert z[0,5:].tolist()==[0.,1.,1.,0.]
m=dict(fallback=False,mean=np.zeros(9),scale=np.ones(9),coef=np.zeros((10,3)));d=np.array([1,1,0],dtype=bool)
assert policy(m,z,d,0,0,20/7)[0].tolist()==[False,False,False]
assert policy(m,z,d,1,0,20/7)[0].tolist()==[True,False,False]
assert policy(m,z,d,0,1,20/7)[0].tolist()==[False,False,False]
print(json.dumps(dict(state='PASS',engineering_seconds=time.time()-START,tail_and_CLS_direction_checked=True,ties_choose_B=True)))
