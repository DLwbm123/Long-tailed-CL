"""User-authorized numerical repair of the same projection QP, no relaxed scientific gates.
The frozen coordinate-sweep implementation remains in core.py for audit.
"""
import os
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import nnls
try:
    import frozen_core
except ModuleNotFoundError:
    import core as frozen_core

ARMS=frozen_core.ARMS
Model=frozen_core.Model
key=frozen_core.key
memory_indices=frozen_core.memory_indices
state=frozen_core.state
flat=frozen_core.flat
setflat=frozen_core.setflat
risks=frozen_core.risks

def solve_dual(gram,b,maxiter,tolerance):
    """Spectral square root converts the unchanged PSD dual to nonnegative LS.
    No ridge is added. Numerical-null eigenspaces are omitted, then acceptance is
    checked against the original Gram/b, not the factorized surrogate.
    """
    gram=(gram+gram.T)*.5
    eigen,U=np.linalg.eigh(gram)
    cutoff=np.finfo(np.float64).eps*len(b)*max(float(eigen.max()),1.)
    assert float(eigen.min())>=-cutoff*4,'Gram not positive semidefinite at roundoff scale'
    keep=eigen>cutoff
    assert keep.any()
    basis=U[:,keep];root=np.sqrt(eigen[keep])
    A=root[:,None]*basis.T
    target=-(basis.T@b)/root
    lam,rnorm=nnls(A,target,maxiter=maxiter,atol=np.finfo(np.float64).eps*len(b))
    q=gram@lam+b
    kkt=float(np.max(np.where(lam>1e-10,abs(q),np.maximum(-q,0.))))
    assert np.isfinite(lam).all() and np.isfinite(kkt) and kkt<=2*tolerance, 'repaired projection KKT acceptance failed'
    return lam,dict(solver='scipy1.13.1 active-set NNLS of unchanged dual',KKT_residual=kkt,
        numerical_rank=int(keep.sum()),constraints=len(b),spectral_roundoff_cutoff=cutoff,
        eigen_min=float(eigen.min()),eigen_max=float(eigen.max()),nnls_residual_norm=float(rnorm),
        regularization_added=0.,maxiter=maxiter)

def project(g,G,cfg):
    if not len(G):return g,dict(solver='empty',residual=0.,correction_norm=0.)
    norms=G.norm(dim=1);keep=norms>1e-12
    if not bool(keep.any()):return g,dict(solver='zero gradients',residual=0.,correction_norm=0.)
    H=G[keep]/norms[keep,None]
    gram=(H.double()@H.double().T).cpu().numpy();b=(H.double()@g.double()).cpu().numpy()
    try:
        lam,receipt=solve_dual(gram,b,cfg['projection_max_sweeps'],cfg['projection_solver_tolerance'])
        # FP64 accumulation before FP32 casting avoids cancellation in the correction.
        d=(g.double()+torch.tensor(lam,device=g.device,dtype=torch.float64)@H.double()).to(g.dtype)
        residual=float(torch.clamp(-(H@d),min=0).max())
        assert residual<=cfg['projection_direction_tolerance'],'infeasible repaired first-order direction'
        receipt.update(residual=residual,correction_norm=float((d-g).norm()))
        return d,receipt
    except BaseException as error:
        if cfg.get('projection_failure_dir'):
            target=Path(cfg['projection_failure_dir']);target.mkdir(parents=True,exist_ok=True)
            torch.save(dict(g=g.detach().cpu(),G=G.detach().cpu(),gram=gram,b=b,error=repr(error)),target/'QP_failure.private.pt')
        raise

# Only the numerical projection implementation changes; model, objective, clipping,
# finite-step acceptance, risk budgets and sampling remain the frozen functions.
frozen_core.project=project
step=frozen_core.step
