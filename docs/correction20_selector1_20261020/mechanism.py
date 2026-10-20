"""Small fixed utility probe. Features never consume labels or patient identifiers."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def ba_weights(y, classes=20):
 counts=np.bincount(y,minlength=classes)
 w=1/(classes*counts[y]);return w/w.sum()


def features(b,f,cosine):
 bp=b.argmax(1);fp=f.argmax(1);rows=np.arange(len(b))
 bs=np.partition(b,-2,axis=1);fs=np.partition(f,-2,axis=1)
 x=np.column_stack([bs[:,-1]-bs[:,-2],fs[:,-1]-fs[:,-2],b[rows,bp]-b[rows,fp],f[rows,fp]-f[rows,bp]])
 support=2*(cosine[rows,fp]-cosine[rows,bp])
 return np.column_stack([x,support]),bp!=fp


def utility_labels(b,f,y):
 bc=b.argmax(1)==y;fc=f.argmax(1)==y
 # 0=rescue, 1=harm, 2=neither. Both-correct rows cannot disagree.
 return np.where(fc&~bc,0,np.where(bc&~fc,1,2)).astype(int)


def objective(vector,x,target,w,regularizer):
 a=vector.reshape(x.shape[1],3);s=x@a;lp=s-logsumexp(s,axis=1,keepdims=True)
 value=-np.dot(w,lp[np.arange(len(target)),target])+regularizer*np.sum(a[:-1]**2)
 p=np.exp(lp);p[np.arange(len(target)),target]-=1
 grad=x.T@(w[:,None]*p);grad[:-1]+=2*regularizer*a[:-1]
 return float(value),grad.ravel()


def fit(x,target,w,regularizer,cfg):
 if not len(x) or not np.isin(target,[0,1]).any():
  return dict(fallback=True,reason='no_disagreement_or_no_nonzero_utility'),dict(success=True,fallback=True)
 w=w/w.sum();mean=w@x;scale=np.sqrt(w@((x-mean)**2));scale=np.where(scale>1e-12,scale,1.)
 design=np.column_stack([(x-mean)/scale,np.ones(len(x))]);shape=(design.shape[1],3);initial=np.zeros(np.prod(shape));hist=[]
 def callback(v):hist.append(objective(v,design,target,w,regularizer)[0])
 result=minimize(objective,initial,args=(design,target,w,regularizer),jac=True,method='L-BFGS-B',callback=callback,options={k:cfg[k] for k in ['maxiter','maxfun','maxls','maxcor','ftol','gtol']})
 val,grad=objective(result.x,design,target,w,regularizer);relative=abs(hist[-1]-hist[-2])/max(1.,abs(hist[-1]),abs(hist[-2])) if len(hist)>1 else None
 accepted=bool(result.success) and np.isfinite(result.x).all() and (np.max(np.abs(grad))<=cfg['gtol'] or (relative is not None and relative<=cfg['ftol']))
 audit=dict(success=bool(accepted),scipy_success=bool(result.success),message=str(result.message),iterations=int(result.nit),calls=int(result.nfev),objective=val,gradient_max=float(np.max(np.abs(grad))),relative_change=relative)
 return dict(fallback=False,mean=mean,scale=scale,coef=result.x.reshape(shape)),audit


def choose(model,x,disagree):
 if model['fallback']:return np.zeros(len(x),dtype=bool),np.zeros(len(x))
 design=np.column_stack([(x-model['mean'])/model['scale'],np.ones(len(x))]);s=design@model['coef'];s-=s.max(1,keepdims=True);p=np.exp(s);p/=p.sum(1,keepdims=True)
 gain=p[:,0]-p[:,1]
 return disagree&(gain>0),gain
