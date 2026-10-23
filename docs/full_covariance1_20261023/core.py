"""Full-rank patient-reliability covariance classifier in native coordinates."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular

def fit(records, ridge):
    d=len(records[0]['mean'])
    pooled=sum(r['covariance'] for r in records)/len(records)
    eps=ridge*np.trace(pooled)/d
    assert eps>0 and np.isfinite(pooled).all()
    L=cholesky((pooled+pooled.T)/2+eps*np.eye(d),lower=True)
    models={'shared_factor':L,'classes':[]}
    for r in records:
        mu=solve_triangular(L,r['mean'],lower=True)
        temp=solve_triangular(L,r['covariance'],lower=True)
        E=solve_triangular(L,temp.T,lower=True).T
        E=(E+E.T)/2
        rho=r['patients']/(r['patients']+d)
        full=rho*E+(1-rho)*np.eye(d)
        diagonal=rho*np.diag(E)+(1-rho)
        factor=cholesky(full,lower=True)
        assert diagonal.min()>0
        models['classes'].append(dict(label=r['label'],mean=mu,rho=rho,
            factor=factor,diagonal=diagonal))
    return models

def scores(x,model):
    u=solve_triangular(model['shared_factor'],x.T,lower=True).T
    out={a:[] for a in ['SHARED','DIAGONAL','FULL']}
    for c in model['classes']:
        delta=u-c['mean']
        out['SHARED'].append(-.5*np.sum(delta*delta,1))
        out['DIAGONAL'].append(-.5*(np.sum(delta*delta/c['diagonal'],1)+np.log(c['diagonal']).sum()))
        z=solve_triangular(c['factor'],delta.T,lower=True)
        out['FULL'].append(-.5*(np.sum(z*z,0)+2*np.log(np.diag(c['factor'])).sum()))
    return {a:np.stack(v,1) for a,v in out.items()}

def independent_scores(x,records,ridge):
    """Separate original-coordinate eigensystems, including the density volume."""
    d=x.shape[1]
    P=sum(r['covariance'] for r in records)/len(records)
    P=(P+P.T)/2+ridge*np.trace(P)/d*np.eye(d)
    L=np.linalg.cholesky(P)
    logP=np.linalg.slogdet(P)[1]
    out={a:[] for a in ['SHARED','DIAGONAL','FULL']}
    for r in records:
        # NumPy dense solves are independent of the runner's SciPy triangular path.
        E=np.linalg.solve(L,np.linalg.solve(L,r['covariance']).T).T
        rho=r['patients']/(r['patients']+d)
        matrices={'SHARED':P,'DIAGONAL':L@np.diag(rho*np.diag(E)+(1-rho))@L.T,
                  'FULL':rho*r['covariance']+(1-rho)*P}
        delta=x-r['mean']
        for a,C in matrices.items():
            vals,vecs=np.linalg.eigh((C+C.T)/2)
            assert vals.min()>0
            z=delta@vecs
            out[a].append(-.5*(np.sum(z*z/vals,1)+np.log(vals).sum()-logP))
    return {a:np.stack(v,1) for a,v in out.items()}

def self_check():
    rng=np.random.default_rng(74002)
    B=rng.normal(size=(8,8));S=B@B.T+.1*np.eye(8)
    rec=[dict(label=c,mean=rng.normal(size=8),covariance=S.copy(),patients=9) for c in [2,5]]
    x=rng.normal(size=(7,8));one=scores(x,fit(rec,.001));two=independent_scores(x,rec,.001)
    assert max(np.max(abs(one[a]-two[a])) for a in one)<1e-8
    # With spherical equal class covariance, every candidate differs by a common scale.
    spherical=[dict(r,covariance=np.eye(8)) for r in rec]
    tied=scores(x,fit(spherical,.001))
    assert all(np.array_equal(tied[a].argmax(1),tied['SHARED'].argmax(1)) for a in tied)
    Q=np.linalg.qr(rng.normal(size=(8,8)))[0]
    rr=[dict(r,mean=Q@r['mean'],covariance=Q@r['covariance']@Q.T) for r in rec]
    rotated=scores(x@Q.T,fit(rr,.001))
    assert np.max(abs(one['FULL']-rotated['FULL']))<1e-8
    return {'state':'PASS','toy_dimension':8,'toy_score_rows':7,'new_model_encodes':0,'new_optimizer_updates':0}
