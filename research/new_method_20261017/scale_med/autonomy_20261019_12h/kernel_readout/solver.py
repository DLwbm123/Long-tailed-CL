"""Fixed weighted KRR with free intercept; no bandwidth or jitter search."""
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve

def raw_kernel(x,z,kind,params):
 d=x@z.T
 if kind=='Linear':return d
 if kind=='Poly2':return (1+d/params['s'])**2
 nx=(x*x).sum(1);nz=(z*z).sum(1);dist=np.maximum(nx[:,None]+nz[None,:]-2*d,0)
 return np.exp(-dist/params['h2'])

def scales(x,q):
 c=q/q.sum();s=float(c@(x*x).sum(1));d=np.maximum((x*x).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*x@x.T,0);i,j=np.triu_indices(len(x),1);values=d[i,j];mass=q[i]*q[j];order=np.argsort(values,kind='stable');cumulative=np.cumsum(mass[order]);h2=float(values[order[np.searchsorted(cumulative,cumulative[-1]/2)]]);params=dict(s=s,h2=h2,trace={})
 if not np.isfinite([s,h2]).all() or min(s,h2)<=1e-12:raise ValueError('DEGENERATE_BASE_KERNEL_SCALE')
 del d,i,j,values,mass,order,cumulative
 for kind in ['Linear','Poly2','RBF']:
  K=raw_kernel(x,x,kind,params);mu=K@c;t=float(c@np.diag(K)-c@mu)
  if not np.isfinite(t) or t<=1e-12:raise ValueError('DEGENERATE_BASE_KERNEL_TRACE')
  params['trace'][kind]=t
 return params

def fit(x,y,q,k,alpha,kind,params,check):
 start=time.monotonic();info=dict(status='UNRESOLVED',kind=kind,lambda_value=alpha,training_rows=len(y));check()
 try:
  c=q/q.sum();assert abs(q.sum()-k)<1e-12 and all(abs(q[y==j].sum()-1)<1e-12 for j in range(k))
  K=raw_kernel(x,x,kind,params)/params['trace'][kind];assert np.isfinite(K).all();sym=float(abs(K-K.T).max());assert sym<=1e-10
  kmu=K@c;offset=float(c@kmu);K-=kmu[:,None];K-=kmu[None,:];K+=offset
  Y=np.eye(k)[y];ymu=c@Y;Yc=Y-ymu;root=np.sqrt(q);B=root[:,None]*K*root[None,:];B.flat[::len(y)+1]+=k*alpha;rhs=root[:,None]*Yc
  factor=cho_factor(B,lower=True,overwrite_a=False,check_finite=True);solution=cho_solve(factor,rhs,check_finite=True);res=float(abs(B@solution-rhs).max());coeff=root[:,None]*solution
  head=dict(coeff=coeff,kmu=kmu,offset=offset,ymu=ymu,kind=kind,params=params,training_rows=len(y),alpha=alpha)
  pred=K@coeff+ymu;info.update(equation_residual_max=res,symmetry_error=sym,training_square_loss=float((q[:,None]*(pred-Y)**2).sum()/k),kernel_trace_base=params['trace'][kind])
  if kind=='Linear':
   mu=c@x;xc=x-mu;W=xc.T@coeff/params['trace'][kind];bias=ymu-mu@W;grad=xc.T@(q[:,None]*(xc@W-Yc))+k*alpha*params['trace'][kind]*W;g=float(abs(grad).max());err=float(abs(pred-(x@W+bias)).max());head.update(primal_W=W,primal_bias=bias);info.update(primal_W_gradient_max=g,primal_prediction_max_error=err)
   assert g<=1e-7 and err<=1e-7
  assert res<=1e-7 and np.isfinite(coeff).all() and np.isfinite(pred).all()
  info['status']='CERTIFIED';check();return head,dict(info,wall_seconds=time.monotonic()-start)
 except (np.linalg.LinAlgError,AssertionError,ValueError,FloatingPointError) as e:
  info.update(error=type(e).__name__,wall_seconds=time.monotonic()-start);return None,info

def predict(head,x,support,q):
 K=raw_kernel(x,support,head['kind'],head['params'])/head['params']['trace'][head['kind']];rowmu=K@(q/q.sum());K-=rowmu[:,None];K-=head['kmu'][None,:];K+=head['offset'];s=K@head['coeff']+head['ymu'];assert np.isfinite(s).all()
 if head['kind']=='Linear':assert abs(s-(x@head['primal_W']+head['primal_bias'])).max()<=1e-7
 return s

def selfcheck():
 rng=np.random.default_rng(74157);x=rng.normal(size=(24,5));y=np.arange(24)%3;q=np.ones(24)/8;query=rng.normal(size=(8,5));params=scales(x,q);cert={}
 for kind in ['Linear','Poly2','RBF']:
  head,info=fit(x,y,q,3,.01,kind,params,lambda:None);assert info['status']=='CERTIFIED';s=predict(head,query,x,q);assert s.shape==(8,3);cert[kind]=info['equation_residual_max']
  if kind=='Linear':
   X=np.c_[x,np.ones(len(x))];D=np.eye(6);D[-1,-1]=0;primal=np.linalg.solve(X.T@(q[:,None]*X)+3*.01*params['trace']['Linear']*D,X.T@(q[:,None]*np.eye(3)[y]));assert np.max(abs(s-np.c_[query,np.ones(len(query))]@primal))<1e-10
 c=q/q.sum();K=raw_kernel(x,x,'RBF',params);mu=K@c;kc=K-mu[:,None]-mu[None,:]+c@mu;assert abs(kc@c).max()<1e-12 and np.linalg.eigvalsh(K).min()>-1e-10
 return dict(state='PASSED',weighted_intercept_primal_dual_equivalence=True,center_identity=True,kernel_symmetry_and_PSD=True,three_kernel_residuals=cert,fixed_train_scales=True,synthetic_only=True)
