"""Small engineering checks, no scientific cache and no parameter search."""
import time
START=time.time()
import json,numpy as np
from mechanism import objective,features,utility_labels,fit,choose
cfg=dict(maxiter=500,maxfun=15000,maxls=30,maxcor=20,ftol=1e-10,gtol=1e-6)
x=np.array([[1.,2.,1.],[3.,1.,1.],[0.,1.,1.],[2.,-1.,1.]])
y=np.array([0,1,2,0]);w=np.array([.1,.2,.3,.4]);v=np.arange(9)*.03;value,grad=objective(v,x,y,w,.05);err=0.
for j in range(9):
 a=v.copy();b=v.copy();a[j]+=1e-6;b[j]-=1e-6;num=(objective(a,x,y,w,.05)[0]-objective(b,x,y,w,.05)[0])/2e-6;err=max(err,abs(num-grad[j]))
assert err<1e-8
b=np.array([[3.,1.,0.],[0.,3.,1.],[3.,1.,0.],[3.,1.,0.]])
f=np.array([[1.,3.,0.],[3.,1.,0.],[1.,3.,0.],[3.,1.,0.]])
t=utility_labels(b,f,np.array([1,1,2,0]));assert t.tolist()==[0,1,2,2]
xx,d=features(b,f,np.eye(4,3));assert d.tolist()==[True,True,True,False] and xx.shape==(4,5)
m,audit=fit(xx[d,:4],t[d],np.ones(3),.05,cfg);assert audit['success'];sel,gain=choose(m,xx[:,:4],d);assert not sel[-1]
zero=dict(fallback=False,mean=np.zeros(4),scale=np.ones(4),coef=np.zeros((5,3)));assert not choose(zero,xx[:,:4],d)[0].any()
assert choose(dict(fallback=True),xx[:,:4],d)[0].sum()==0
print(json.dumps(dict(state='PASS',finite_difference_max_error=err,selector_acceptance=audit,engineering_seconds=time.time()-START)))
