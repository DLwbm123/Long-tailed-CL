"""Float64 production kernels; CPU reference remains an independent oracle."""
import numpy as np
import torch
from tools.rfvila_reference import group_folds,pick_lambda,class_balanced_mse

GRID=(1e-6,1e-5,1e-4,5e-4,1e-3,2e-3,1e-2,1e-1,1.)

def tensor(x,device):return torch.as_tensor(x,dtype=torch.float64,device=device)

def transform(a,u,ra,ru,kind):
    if kind=='linear':return a if u is None else torch.cat((a,u),1)/2**.5
    m=ra.shape[1];aa=a@ra
    if u is None:return torch.relu(aa)*(2/m)**.5
    uu=u@ru
    if kind=='relu_split':return torch.cat((torch.relu(aa[:,:m//2]),torch.relu(uu[:,:m//2])),1)*(2/m)**.5
    joint=(aa+uu)/2**.5
    if kind=='random_linear':return joint/m**.5
    if kind=='relu_joint':return torch.relu(joint)*(2/m)**.5
    raise ValueError('UNKNOWN_FEATURE')

def solve(S,Q,lam):
    if not np.isfinite(lam) or lam<=0 or not Q.shape[1]:raise ValueError('INVALID_LAMBDA')
    A=(S+S.T)/2;A=A.clone();A.diagonal().add_(Q.shape[1]*lam)
    L=torch.linalg.cholesky(A);W=torch.cholesky_solve(Q,L)
    residual=float(torch.linalg.vector_norm(A@W-Q)/torch.linalg.vector_norm(Q).clamp_min(1e-30))
    if not torch.isfinite(W).all() or residual>1e-8:raise ArithmeticError('SOLVER_RESIDUAL:'+str(residual))
    return W,residual

def select_cv(x,y,components,ids,fold=None):
    fold=group_folds(y,components) if fold is None else fold
    if fold is None:return .001,{'status':'TASK1_GROUP_CV_UNAVAILABLE','folds':0,'grid':list(GRID),'scores':[],'spectral_decompositions':0,'candidate_solves':0}
    scores=[];max_residual=0.
    for f in range(int(fold.max())+1):
        tr=fold!=f;va=~tr;yt=np.asarray(y)[tr]
        weight=np.array([1/(len(ids)*np.sum(yt==c)) for c in yt])
        lookup={c:j for j,c in enumerate(ids)}
        X=x[tensor(tr,x.device).bool()]*tensor(np.sqrt(weight[:,None]),x.device)
        Y=tensor(np.eye(len(ids))[[lookup[int(c)] for c in yt]]*np.sqrt(weight[:,None]),x.device)
        xv=x[tensor(va,x.device).bool()]
        dual=len(X)<X.shape[1]
        A=X@X.T if dual else X.T@X
        d,V=torch.linalg.eigh((A+A.T)/2)
        rhs=Y if dual else X.T@Y
        coeff=V.T@rhs
        values=[]
        for lam in GRID:
            z=V@(coeff/(d[:,None]+lam))
            residual=float(torch.linalg.vector_norm(A@z+lam*z-rhs)/torch.linalg.vector_norm(rhs).clamp_min(1e-30))
            max_residual=max(max_residual,residual)
            if residual>1e-8 or not torch.isfinite(z).all():raise ArithmeticError('CV_RESIDUAL')
            W=X.T@z if dual else z
            values.append(class_balanced_mse((xv@W).cpu().numpy(),np.asarray(y)[va],ids))
        scores.append(values)
    losses=np.mean(scores,axis=0).tolist()
    return pick_lambda(GRID,losses),{'status':'GROUPED_TASK1_READOUT_CV','folds':len(scores),'grid':list(GRID),'scores':scores,'losses':losses,'spectral_decompositions':len(scores),'candidate_solves':len(scores)*len(GRID),'max_residual':max_residual}

def append(S,Q,seen,counts,cid,x):
    if cid in seen or len(x)==0 or not torch.isfinite(x).all():raise ValueError('DUPLICATE_OR_BAD_CLASS')
    S.add_(x.T@x/len(x));Q=torch.cat((Q,x.mean(0)[:,None]),1)
    seen.append(int(cid));counts.append(len(x));return S,Q
