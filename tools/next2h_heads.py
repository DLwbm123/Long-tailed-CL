"""Fixed NEXT2-H heads from class-normalized moments; no neural training."""
import torch
import prototype_coherent as native


def validate_bank(bank):
    mu,Q=bank['mu'].double(),bank['Q'].double();k,d=mu.shape
    if Q.shape!=(k,d,d) or len(bank['n'])!=k or any(n<=0 for n in bank['n']):raise ValueError('Invalid bank dimensions/counts')
    if not torch.isfinite(mu).all() or not torch.isfinite(Q).all():raise ValueError('Nonfinite bank')
    scale=max(float(Q.norm()),1e-12)
    if float((Q-Q.transpose(1,2)).norm())>1e-8*scale:raise ValueError('Asymmetric second moments')
    covariance=(Q-mu[:,:,None]*mu[:,None,:]).mean(0)
    covariance=(covariance+covariance.T)/2
    eigenvalues=torch.linalg.eigvalsh(covariance);spectral=max(float(eigenvalues.abs().max()),1e-12)
    if float(eigenvalues.min()) < -1e-8*spectral:raise ValueError('Non-PSD class-balanced covariance')
    return covariance,dict(classes=k,dim=d,counts=bank['n'],covariance_min_eigenvalue=float(eigenvalues.min()),covariance_spectral_norm=spectral)


def build_cf_head(bank):
    W,_=native.head(bank,strength=0.);k=len(bank['n']);d=bank['mu'].shape[1]
    A=bank['Q'].sum(0)/k+.001*torch.eye(d,dtype=torch.float64);B=bank['mu'].T/k
    residual=float((A@W-B).norm()/B.norm().clamp_min(1e-12))
    if residual>1e-8:raise ValueError('C0 solve residual exceeds fixed tolerance')
    return dict(weight=W.float(),bias=torch.zeros(k),residual=residual)


def build_ncm_head(bank):
    mu=bank['mu'].double()
    return dict(weight=mu.T.float(),bias=(-.5*mu.square().sum(1)).float())


def build_cblda_head(bank,covariance):
    d=covariance.shape[0];tau=max(float(covariance.trace()/d),1e-12)
    sigma=.9*covariance+(.1+1e-6)*tau*torch.eye(d,dtype=torch.float64)
    chol=torch.linalg.cholesky(sigma)
    V=torch.cholesky_solve(bank['mu'].double().T,chol)
    if not torch.isfinite(V).all():raise ValueError('Nonfinite C2 head')
    return dict(weight=V.float(),bias=(-.5*(bank['mu'].T*V).sum(0)).float(),tau=tau,rho=.1)


def score(z,head):return z@head['weight'].numpy()+head['bias'].numpy()


def self_check():
    torch.manual_seed(74101);x=torch.randn(24,5,dtype=torch.float64);y=torch.arange(3).repeat_interleave(8)
    mu=torch.stack([x[y==c].mean(0) for c in range(3)])
    Q=torch.stack([x[y==c].T@x[y==c]/8 for c in range(3)])
    bank=dict(mu=mu,Q=Q,n=[8]*3,components=[]);C,_=validate_bank(bank);h=build_cf_head(bank)
    expected=torch.linalg.solve(Q.mean(0)+.001*torch.eye(5,dtype=torch.float64),mu.T/3)
    assert torch.allclose(h['weight'].double(),expected,atol=1e-6,rtol=1e-5)
    z=torch.randn(9,5,dtype=torch.float64);ncm=z@mu.T-.5*mu.square().sum(1)
    distances=-(z[:,None,:]-mu[None]).square().sum(2)/2
    assert torch.equal(ncm.argmax(1),distances.argmax(1))
    iso=dict(mu=mu,Q=torch.stack([torch.outer(v,v)+2*torch.eye(5,dtype=torch.float64) for v in mu]),n=[8]*3,components=[])
    C,_=validate_bank(iso);lda=build_cblda_head(iso,C)
    assert torch.equal((z.float()@lda['weight']+lda['bias']).argmax(1),ncm.argmax(1))
    permutation=torch.tensor([2,0,1]);permuted=dict(mu=mu[permutation],Q=Q[permutation],n=[8]*3,components=[])
    hp=build_cf_head(permuted);assert torch.allclose(hp['weight'],h['weight'][:,permutation])
    for c in range(3):
        a=x[y==c][:4];b=x[y==c][4:]
        assert torch.allclose((a.sum(0)+b.sum(0))/8,mu[c])
        assert torch.allclose((a.T@a+b.T@b)/8,Q[c])
    old=ncm[:,:2];expanded=torch.cat([old,ncm[:,2:]],1)
    assert torch.equal(old[:,0]-old[:,1],expanded[:,0]-expanded[:,1])
    previous=torch.tensor([.8,.7]);restricted=torch.tensor([.9,.6]);all_recall=torch.tensor([.5,.4])
    assert torch.allclose((previous-all_recall).mean(),(previous-restricted).mean()+(restricted-all_recall).mean())
    print('PASS: ridge/NCM/isotropic LDA, class permutation, sequential moments, NCM internal order, decomposition')
