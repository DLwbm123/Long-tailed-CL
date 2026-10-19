"""Fixed signed-input covariance references; no hyperparameter selection."""
import numpy as np
from scipy.linalg import cholesky,solve_triangular

def shrink(cov):
 d=len(cov);v1=float(np.trace(cov)/d);v2=float((cov.sum()-np.trace(cov))/(d*(d-1)))
 assert np.isfinite(cov).all() and v1>1e-12
 S=cov+v2*np.ones_like(cov);S.flat[::d+1]+=v1-v2
 return S,dict(gamma1=1.,gamma2=1.,average_variance=v1,average_off_diagonal=v2)

def factor(S):
 sym=float(abs(S-S.T).max());assert sym<=1e-10 and np.isfinite(S).all() and np.diag(S).min()>1e-12
 L=cholesky(S,lower=True,check_finite=True);rel=float(abs(L@L.T-S).max()/np.max(abs(S)));assert rel<=1e-8
 logdet=float(2*np.log(np.diag(L)).sum());assert np.isfinite(logdet)
 return L,logdet,dict(certified=True,symmetry_error=sym,factor_reconstruction_relative_max=rel)

def covariance(x,q):
 q=q/q.sum();mu=q@x;z=x-mu;cov=z.T@(q[:,None]*z);assert np.isfinite(mu).all() and np.isfinite(cov).all();return mu,cov

def fit_class(x,q):
 mu,cov=covariance(x,q);S,info=shrink(cov);raw,ld,rawcheck=factor(S);sd=np.sqrt(np.diag(S));C=S/sd[:,None]/sd[None,:];corr,_,corrcheck=factor(C);assert abs(np.diag(C)-1).max()<=1e-10
 return dict(mean=mu,cov=cov,raw=raw,corr=corr,logdet=ld),dict(**info,raw=rawcheck,corr=corrcheck,rows=len(x),dimensions=x.shape[1],status='CERTIFIED',factor_count=2)

def fit_pool(states):
 cov=sum(v['cov'] for v in states)/len(states);S,info=shrink(cov);sd=np.sqrt(np.diag(S));C=S/sd[:,None]/sd[None,:];L,_,check=factor(C)
 return dict(corr=L),dict(**info,corr=check,class_count=len(states),status='CERTIFIED',factor_count=1)

def distance(x,mu,L):
 z=solve_triangular(L,(x-mu).T,lower=True,check_finite=True);v=(z*z).sum(0);assert np.isfinite(v).all();return v

def predict(x,states,pool):
 out={kind:[] for kind in ['NCM','SharedM','RawM','QDA','CorrM']}
 for state in states:
  mu=state['mean'];raw=distance(x,mu,state['raw']);corr=distance(x,mu,state['corr'])
  out['NCM'].append(-((x-mu)**2).sum(1));out['SharedM'].append(-distance(x,mu,pool['corr']));out['RawM'].append(-raw);out['QDA'].append(-.5*(raw+state['logdet']));out['CorrM'].append(-corr)
 return {k:np.column_stack(v) for k,v in out.items()}

def selfcheck():
 rng=np.random.default_rng(74158);x=rng.normal(size=(24,5));q=np.arange(1,25,dtype=float);q/=q.sum();query=rng.normal(size=(10,5));state,info=fit_class(x,q);mu,cov=covariance(x,q)
 assert np.allclose(mu,sum(q[i]*x[i] for i in range(24))) and np.allclose(cov,sum(q[i]*np.outer(x[i]-mu,x[i]-mu) for i in range(24)))
 other,_=fit_class(x*.4+1,q);pool,_=fit_pool([state,other]);scores=predict(query,[state,other],pool);S,_=shrink(cov);raw=np.einsum('ij,ji->i',query-mu,np.linalg.solve(S,(query-mu).T));assert np.max(abs(scores['RawM'][:,0]+raw))<1e-10;assert np.max(abs(scores['QDA'][:,0]+.5*(raw+np.linalg.slogdet(S)[1])))<1e-10
 third,_=fit_class(x*.8-1,q);pool3,_=fit_pool([state,other,third]);after=predict(query,[state,other,third],pool3)
 for label in ['NCM','RawM','QDA','CorrM']:assert np.array_equal(scores[label],after[label][:,:2])
 return dict(state='PASSED',weighted_mean_covariance=True,pooling=True,raw_Gaussian_distance_and_logdet=True,old_class_score_invariance=True,SPD_and_factor_certificates=True,fixed_gamma=[1,1],synthetic_only=True)
