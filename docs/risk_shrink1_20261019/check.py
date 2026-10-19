import numpy as np
from core import fold_id,weights,center,problem,uniform,constrained,change

rng=np.random.default_rng(74002);n,k=17,4
c=center(rng.normal(size=(n,k)));d=center(rng.normal(size=(n,k)));y=rng.integers(k,size=n);w=rng.random(n);a=rng.random(k)
q,b,r=problem(c,d,y,w);target=center(np.eye(k)[y]);direct=((center(c+d*a)-target)**2*w[:,None]).sum()/(w.sum()*k)
assert abs(direct-(r+change(q,b,a)))<1e-12
for j in range(k):
 eps=np.eye(k)[j]*1e-6
 finite=(change(q,b,a+eps)-change(q,b,a-eps))/2e-6
 assert abs(finite-(2*q@a+2*b)[j])<1e-7
u=uniform(q,b);assert all(change(q,b,np.ones(k)*u)<=change(q,b,np.ones(k)*z)+1e-10 for z in np.linspace(0,1,101))
# Minimize (a-.8)^2 subject to a^2-.4a<=0; optimum=.4.
a,audit=constrained(np.array([[1.]]),np.array([-.8]),np.array([[[1.]]]),np.array([[-.2]]),np.array([1.]),set())
assert audit['success'] and audit['max_normalized_violation']<1e-7 and abs(a[0]-.4)<1e-6
assert np.allclose(weights(['a','a','b']),[.25,.25,.5])
assert fold_id('7')==fold_id('7') and fold_id('7') in range(3)
assert np.allclose(center(c+d*0),c) and np.allclose(center(c+d),center(c+d*np.ones(k)))
print('PASS: quadratic proxy and gradient; uniform optimum; constrained known optimum; patient weights and endpoint centering')
