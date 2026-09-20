"""Independent CT2-D metric/moment controls; never an online learner."""
import numpy as np
from ct1_statistics import empty, append, joint, ridge, fit_map, transport, psd_factor, risk


def classify(scores, order):
    columns=np.argsort(order,kind='stable')
    return columns[np.argmax(scores[:,columns],axis=1)]


def metrics(scores,y,order,known,groups,meta,components=None):
    c=len(order);assert scores.shape==(len(y),c) and np.isfinite(scores).all()
    pred=classify(scores,order);cm=np.zeros((c,c),dtype=np.int64);np.add.at(cm,(y,pred),1)
    n=cm.sum(1);assert (n>0).all()
    recalls=100*cm.diagonal()/n;pcs=[]
    for k in range(c):
        mask=y==k;other=scores[mask].copy();other[:,k]=-np.inf
        pcs.append(dict(meta,head_index=k,original_label=int(order[k]),n_images=int(n[k]),
            n_correct=int(cm[k,k]),recall=float(recalls[k]),margin_mean=float(np.mean(scores[mask,k]-other.max(1))),
            n_components=None if components is None else len(np.unique(components[mask]))))
    f1=np.divide(2*cm.diagonal(),n+cm.sum(0),out=np.zeros(c),where=n+cm.sum(0)>0)
    m=dict(meta,n_images=len(y),balanced_accuracy=float(recalls.mean()),accuracy=100*float(cm.trace()/n.sum()),
           macro_f1=100*float(f1.mean()),zero_recall_classes=[int(order[k]) for k in range(c) if recalls[k]==0])
    for name,labels in groups.items():
        ix=np.flatnonzero(np.isin(order,labels));m[name+'_recall']=float(recalls[ix].mean()) if len(ix) else None
    er=[]
    for scope,ix in [('old',np.arange(known)),('current',np.arange(known,c))]:
        m[scope+'_macro_recall']=float(recalls[ix].mean()) if len(ix) else None
        if not len(ix):continue
        mask=np.isin(y,ix);yp=y[mask];pp=pred[mask]
        restricted=ix[classify(scores[mask][:,ix],order[ix])]
        m['restricted_'+scope+'_BA']=100*float(np.mean([(restricted[yp==k]==k).mean() for k in ix]))
        m['restricted_'+scope+'_accuracy']=100*float(np.mean(restricted==yp))
        events={'correct':pp==yp,'to_other':~np.isin(pp,ix),'wrong_within':(pp!=yp)&np.isin(pp,ix)}
        e=dict(meta,scope=scope,n_images=int(mask.sum()),n_classes=len(ix))
        for name,v in events.items():
            e[name+'_sample']=100*float(v.mean());e[name+'_macro']=100*float(np.mean([v[yp==k].mean() for k in ix]))
        for suffix in ('sample','macro'):assert abs(sum(e[k+'_'+suffix] for k in events)-100)<1e-9
        er.append(e)
    old=m['old_macro_recall'];cur=m['current_macro_recall']
    m['HM']=None if old is None else (2*old*cur/(old+cur) if old+cur else 0.)
    if known:np.testing.assert_allclose((old*known+cur*(c-known))/c,m['balanced_accuracy'],atol=1e-12)
    return m,pcs,er


def moments(raw,y,classes=None):
    z=joint(raw);c=int(y.max())+1;s=empty(z.shape[1]);newS=np.zeros_like(s['S'])
    for k in range(c):
        x=z[y==k];s=append(s,x,np.full(len(x),k),[k],1)
        if classes is not None and k in classes:newS+=x.T@x/len(x)
    return s,newS


def factorial(stored,fresh,newS,known):
    mu0=stored['mu'][:,:known];mu1=fresh['mu'][:,:known]
    v0=stored['S']-newS-mu0@mu0.T
    v1=fresh['S']-newS-mu1@mu1.T
    diag={}
    for name,v in [('stored',v0),('fresh',v1)]:
        ev=np.linalg.eigvalsh((v+v.T)/2)
        diag[name]=dict(min_eigenvalue=float(ev[0]),max_eigenvalue=float(ev[-1]),trace=float(np.trace(v)))
        assert ev[0]>=-1e-5*max(1.,abs(ev[-1])),('BLOCKED_COVARIANCE_NONPSD',name,diag[name])
    states={}
    for a,b in [(0,0),(1,0),(0,1),(1,1)]:
        mu=mu1 if a else mu0
        s={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in fresh.items()}
        s['S']=newS+(v1 if b else v0)+mu@mu.T;s['S']=(s['S']+s['S'].T)/2
        s['mu']=np.column_stack([mu,fresh['mu'][:,known:]])
        states[f'Q{a}{b}']=s
    return states,diag


def independent_checks():
    rng=np.random.default_rng(46001);x=rng.normal(size=(23,12));y=np.repeat(np.arange(3),[5,7,11])
    st=append(empty(12),x,y,range(3),1);a=rng.uniform(.6,1.8,12);b=rng.normal(size=12)
    t=transport(st,a,b,np.zeros(12),2);direct=append(empty(12),x*a+b,y,range(3),2)
    errors={k:float(np.max(abs(t[k]-direct[k]))) for k in ('S','mu','v')}
    for k in errors:np.testing.assert_allclose(t[k],direct[k],atol=1e-11)
    aa,bb,rr,_=fit_map(x,x,y);np.testing.assert_allclose(aa,1);np.testing.assert_allclose(bb,0,atol=1e-12)
    identity=transport(st,aa,bb,rr,2)
    for k in errors:np.testing.assert_allclose(identity[k],st[k],atol=1e-11)
    weights=1/(3*np.bincount(y)[y]);G=x.T@(weights[:,None]*x);R=x.T@(weights[:,None]*np.eye(3)[y])
    W,_=ridge(st);np.testing.assert_allclose(W,np.linalg.solve(G+.001*np.eye(12),R),atol=1e-11)
    B=rng.normal(size=(3,3));A=G+.001*np.eye(12);loss=lambda w:np.sum(weights[:,None]*(x@w-np.eye(3)[y])**2)+.001*np.sum(w*w)
    np.testing.assert_allclose(loss(W@B)-loss(W),np.trace((B-np.eye(3)).T@(W.T@A@W)@(B-np.eye(3))),atol=1e-10)
    F=psd_factor(A);np.testing.assert_allclose(F.T@F,A,atol=1e-11)
    sol,rec=risk(st,W,eta=0);assert sol is not None;np.testing.assert_allclose(sol['W'],W,atol=1e-5,rtol=1e-5)
    order=np.array([2,0,1]);scores=rng.normal(size=(23,3));scores[0]=0
    pred=order[classify(scores,order)];perm=np.array([1,2,0]);assert np.array_equal(pred,order[perm][classify(scores[:,perm],order[perm])])
    assert pred[0]==0 and np.array_equal(order[np.argsort(order)],np.arange(3))
    fresh=append(empty(12),x*a+b,y,range(3),1);new=x[y==2]*a+b;newS=new.T@new/len(new)
    # Build a coherent mixed stored bank with exact current moments.
    old=append(empty(12),x[y<2],y[y<2],range(2),1);stored=append(old,new,np.full(len(new),2),[2],1)
    states,_=factorial(stored,fresh,newS,2)
    np.testing.assert_allclose(states['Q00']['S'],stored['S'],atol=1e-12)
    np.testing.assert_allclose(states['Q11']['S'],fresh['S'],atol=1e-12)
    assert not np.allclose(states['Q10']['S'],stored['S'])
    return dict(status='PASS',affine_errors=errors,identity=True,cross_Gram=True,weighted_batch_objective=True,
                psd_factor_direction=True,eta_zero=rec,label_permutation_and_tie=True,factorial_reconstruction=True)
