"""Design-only identity-preserving covariance borrowing and tail-conditioned heads.
No medical loader, checkpoint discovery, production worker, or optimizer.
"""
import math
import torch
import core1_competition as competition
from lca_old_pair_repair import contrast_basis, native_operator, checkpoint as legacy_pack
from concm_statistical import matching_anchors

VERSION='identity_covariance_tail_direction_v2'
PRESETS={
 'MPC':(True,False,0.,0.), 'REG':(False,False,.1,0.),
 'TAIL':(False,True,0.,0.), 'TAIL_REG':(False,True,.1,0.),
 'DSM':(False,True,.1,.1), 'MPC_TAIL':(True,True,0.,0.),
 'MPC_TAIL_REG':(True,True,.1,0.), 'MPC_DSM':(True,True,.1,.1)}

def validate_design(p):
    exact=dict(version=VERSION,status='CPU_DESIGN_ONLY',primary_control='local_transport',
        production_integrated=False,GPU_launch_authorized=False,medical_data_access_authorized=False,
        checkpoint_access_authorized=False,training_feedback=False,test_accessed=False,
        validation_retuning=False,automatic_retry=False,optimizer_updates=0,seed=74002,
        future_gpus=[0,1],support_pseudocount=16.,completion_max_weight=.4,
        tail_beta=.5,regularization_relative_native_curvature=.1,ETF_direction_weight=.1,
        guard='old_max_nonincrease_new_logits_fixed',heads=list(PRESETS),regularization_center='reference_head_increment',REG_only='CPU_identity_check_no_independent_medical_arm',future_performance_heads=[v for v in PRESETS if v!='REG'],future_sidecars=126)
    if any(p.get(k)!=v for k,v in exact.items()):raise ValueError('Independent design-only scope required')

def finite(*xs):
    if not xs or any(x.dtype!=torch.float64 or x.device!=xs[0].device or not torch.isfinite(x).all() for x in xs):raise ValueError('Finite same-device float64 inputs required')

def psd(x):
    x=(x+x.T)/2;e,u=torch.linalg.eigh(x)
    tol=1e-10*max(1.,float(x.abs().max()))
    if float(e.min()) < -tol:raise ValueError('Incoherent covariance')
    return (u*e.clamp_min(0))@u.T

def covariance_parts(bank):
    mu,Q=bank['mu'],bank['Q'];finite(mu,Q);k,d=mu.shape
    if Q.shape!=(k,d,d) or not torch.allclose(Q,Q.transpose(1,2),atol=1e-10,rtol=1e-10) or any(type(r['label'])!=int or not 0<=r['label']<k for r in bank['components']) or len(bank['n'])!=k or any(type(n)!=int or n<1 for n in bank['n']):raise ValueError('Invalid class bank')
    parts=[];within=[]
    for c in range(k):
        rows=[r for r in bank['components'] if r['label']==c]
        if not rows or any(r['center'].shape!=(d,) or type(r['count'])!=int or r['count']<1 for r in rows) or sum(r['count'] for r in rows)!=bank['n'][c]:raise ValueError('Missing or inconsistent class components')
        for r in rows:
            finite(r['center'])
            if not math.isfinite(float(r['mass'])) or float(r['mass'])<=0 or not math.isfinite(float(r['radius'])) or float(r['radius'])<0:raise ValueError('Invalid component mass/radius')
        mass=sum(float(r['mass']) for r in rows);center=sum(float(r['mass'])*r['center'] for r in rows)
        if abs(mass-1)>1e-8 or not torch.allclose(center,mu[c],rtol=1e-8,atol=1e-9):raise ValueError('Component mean differs')
        between=sum(float(r['mass'])*torch.outer(r['center']-mu[c],r['center']-mu[c]) for r in rows)
        S=psd(Q[c]-torch.outer(mu[c],mu[c])-between)
        radius=sum(float(r['mass'])*float(r['radius']) for r in rows)
        if abs(float(S.trace())-radius)>1e-8*max(1.,radius):raise ValueError('Component radius differs from moments')
        parts.append(rows);within.append(S)
    return parts,torch.stack(within)

def borrow(bank):
    rows,S=covariance_parts(bank);mu=bank['mu'];tr=S.diagonal(dim1=1,dim2=2).sum(1)
    shape=S/tr.clamp_min(1e-30)[:,None,None];n=mu.new_tensor(bank['n']);reliability=n/(n+16.)
    eta=.4*16/(n+16.);out=bank['Q'].clone();weights=mu.new_zeros((len(mu),len(mu)))
    for c in range(len(mu)):
        donors=[j for j in range(len(mu)) if j!=c and float(tr[j])>1e-12]
        if not donors or float(tr[c])<=1e-12:continue
        distance=(shape[donors]-shape[c]).square().sum((1,2))
        w=reliability[donors]/(1+distance);w=w/w.sum();weights[c,donors]=w
        mixed=(w[:,None,None]*shape[donors]).sum(0)*tr[c]
        out[c]+=eta[c]*(mixed-S[c])
    revised=dict(bank,Q=out)
    _,updated=covariance_parts(revised)
    return revised,dict(mean_and_component_centers_unchanged=True,counts_masses_radii_unchanged=True,
        max_within_trace_change=float((updated.diagonal(dim1=1,dim2=2).sum(1)-tr).abs().max()),
        covariance_change_norm=float((out-bank['Q']).norm()),eta=eta.tolist(),
        donor_effective_count=[None if not w.any() else float(1/w.square().sum()) for w in weights],
        measured_donor_reliability=False,measured_old_drift=False)

def tail_geometry(bank,pairs,W,old):
    """Moment-consistent shared-shape Gaussian mixture; never draw old features."""
    parts,S=covariance_parts(bank);mu=bank['mu'];k=len(mu);edges=[]
    for c in range(old):
        tr=float(S[c].trace());q=bank['n'][c]/(bank['n'][c]+16.)
        for r in parts[c]:
            u=r['center'];ratio=float(r['radius'])/tr if tr>1e-12 else 0.
            for j in range(old):
                if j==c or not pairs[c,j]:continue
                v=W[:,c]-W[:,j];m=float(u@v);variance=float(ratio*(v@(S[c]@v)))
                if not math.isfinite(m) or not math.isfinite(variance) or variance < -1e-10:raise ValueError('Invalid projected moments')
                sigma=math.sqrt(max(0.,variance))
                if sigma<=1e-12:
                    p=float(m<=0);phi=0.;z=0.;a=torch.zeros_like(u);target=max(0.,-m)
                else:
                    z=m/sigma;p=.5*math.erfc(z/math.sqrt(2));phi=math.exp(-.5*z*z)/math.sqrt(2*math.pi)
                    a=ratio*(S[c]@v)/sigma;target=sigma
                    if p<torch.finfo(mu.dtype).eps:p=phi=z=0.
                if not p:continue
                edges.append(dict(c=c,j=j,u=u,ratio=ratio,p=p,phi=phi,z=z,a=a,target=target,
                    weight=.5*q*float(pairs[c,j])*float(r['mass'])/k))
    rhs=torch.zeros_like(W)
    for e in edges:
        b=e['weight']*e['target']*(e['p']*e['u']-e['phi']*e['a']);rhs[:,e['c']]+=b;rhs[:,e['j']]-=b
    return S,edges,rhs

def tail_operator(S,edges,W):
    out=torch.zeros_like(W);products={c:S[c]@W for c in {e['c'] for e in edges}}
    for e in edges:
        c,j=e['c'],e['j'];v=W[:,c]-W[:,j];u,a=e['u'],e['a']
        qv=e['p']*(e['ratio']*(products[c][:,c]-products[c][:,j])+u*(u@v))
        qv-=e['phi']*(u*(a@v)+a*(u@v));qv+=e['z']*e['phi']*a*(a@v)
        step=e['weight']*qv;out[:,c]+=step;out[:,j]-=step
    return out

def systems(bank,pairs,R,W0,old,preset):
    finite(bank['mu'],bank['Q'],pairs,R,W0);k,d=bank['mu'].shape
    if (preset not in PRESETS or type(old)!=int or not 0<=old<=k or W0.shape!=(d,k)
        or R.shape!=(d,d) or pairs.shape!=(k,k) or (pairs<0).any() or pairs.diag().any()
        or not torch.allclose(pairs.sum(1),pairs.new_ones(k),atol=1e-8,rtol=1e-8)
        or not torch.allclose(R,R.T,atol=1e-10,rtol=1e-10)):
        raise ValueError('Invalid control geometry or preset')
    _,targets=competition.laplacians(pairs);b=bank['mu'].T/k+.5*(bank['mu'].T/k)@targets
    H0=lambda W:native_operator(bank,pairs,R,W)
    error=float((H0(W0)-b).norm()/b.norm().clamp_min(1e-30))
    if not math.isfinite(error) or error>1e-8:raise ValueError('Reference is not native double optimum')
    V=W0.new_zeros((k,max(0,old-1)));V[:old]=contrast_basis(old,W0)
    use_mpc,use_tail,reg,eta=PRESETS[preset]
    revised,ma=borrow(bank) if use_mpc else (bank,dict(enabled=False))
    Hm=lambda W:native_operator(revised,pairs,R,W)
    S,edges,bt=tail_geometry(bank,pairs,W0,old) if use_tail else (None,[],torch.zeros_like(W0))
    Ht=lambda W:tail_operator(S,edges,W) if edges else torch.zeros_like(W)
    A0=lambda Z:H0(Z@V.T)@V
    etf=dict(enabled=bool(eta),skipped=False);E=torch.zeros_like(W0)
    if eta and old>=2:
        target,etf=matching_anchors(bank['mu'][:old],W0[:,:old]-W0[:,:old].mean(1,keepdim=True))
        if target is not None:E[:,:old]=(target-W0[:,:old])@V[:old]@V[:old].T
    force=Ht(E)@V
    def operator(Z):return (Hm(Z@V.T)+Ht(Z@V.T))@V+reg*A0(Z)
    rhs=(b-Hm(W0)+bt-Ht(W0))@V+eta*force
    base=revised['Q'].mean(0)+.001*R+reg*(bank['Q'].mean(0)+.001*R)
    C=torch.linalg.cholesky((base+base.T)/2)
    pre=lambda Z:torch.cholesky_solve(Z,C)
    audit=dict(version=VERSION,preset=preset,old_count=old,old_contrast_dimensions=max(0,old-1),
        feature_dimensions=d,native_residual=error,MPC=ma,tail_component_edges=len(edges),
        tail_risk_mass=sum(e['weight']*e['p'] for e in edges),regularization_weight=reg,
        ETF_direction_weight=eta,ETF=etf,ETF_direction_force_norm=float(force.norm()),
        new_columns_fixed=True,common_old_component_fixed=True,full_native_curvature=True,
        feature_span_projection=False,hard_energy_cap=False,validation_graph=False,
        sampled_historical_features=0,gaussian_mixture_is_proxy=True,optimizer_updates=0)
    return operator,rhs,pre,V,audit

@torch.no_grad()
def fit(bank,pairs,R,W0,old,preset):
    operator,rhs,pre,V,a=systems(bank,pairs,R,W0,old,preset)
    inactive=(a['MPC'].get('covariance_change_norm',0.)==0. and not a['tail_component_edges'] and not a['ETF_direction_force_norm'])
    if not V.shape[1] or inactive or not rhs.any():return torch.zeros_like(W0),dict(a,exact_identity=True,relative_residual=0.,delta_norm=0.)
    Z,qa=competition.pcg(operator,rhs,pre,torch.zeros_like(rhs));delta=Z@V.T
    if delta[:,old:].any():raise ArithmeticError('New columns changed')
    return delta,dict(a,exact_identity=not bool(delta.any()),relative_residual=qa['relative_residual'],iterations=qa['iterations'],delta_norm=float(delta.norm()))

def guard(base,delta,old):
    if (base.ndim!=2 or delta.shape!=base.shape or base.dtype!=delta.dtype or base.device!=delta.device
        or base.dtype not in (torch.float32,torch.float64) or type(old)!=int or not 0<=old<=base.shape[1]
        or not torch.isfinite(base).all() or not torch.isfinite(delta).all() or delta[:,old:].any()):raise ValueError('Invalid old-only score correction')
    out=base.clone()
    if old:
        raw=base[:,:old]+delta[:,:old];rm=raw.max(1,keepdim=True).values;bm=base[:,:old].max(1,keepdim=True).values
        out[:,:old]=torch.where(rm>bm,bm-(rm-raw),raw)
        if not torch.isfinite(out).all() or (out[:,:old].max(1).values>bm.flatten()).any():raise ArithmeticError('Old maximum increased')
    return out

def pack(delta,old,meta):
    state=legacy_pack(delta,old,meta);state['version']=VERSION;return state

def restore(state,old,meta,shape):
    if state.get('version')!=VERSION or state.get('metadata')!=meta or state.get('old_count')!=old or state['delta'].shape!=shape:raise ValueError('Sidecar provenance differs')
    return pack(state['delta'],old,meta)['delta'].clone()
