"""Matched image-level moments, nested projection, and fixed readouts."""
import math
import torch


def weights(groups, device='cpu'):
    from collections import Counter
    counts=Counter(groups)
    return torch.tensor([1/(len(counts)*counts[g]) for g in groups],dtype=torch.float64,device=device)


def augmented(x):
    return torch.cat([x.double(),torch.ones(len(x),1,dtype=torch.float64,device=x.device)],1)


def transform(projection):
    matrix=torch.zeros(projection.shape[0]+1,projection.shape[1]+1,dtype=torch.float64,device=projection.device)
    matrix[:-1,:-1]=projection.double();matrix[-1,-1]=1
    return matrix


def project_moments(gram, label_sum, projection):
    matrix=transform(projection)
    return matrix.T@gram@matrix,matrix.T@label_sum


def ridge(grams, targets, alpha):
    gram=torch.stack(grams).sum(0)
    target=torch.stack(targets,1)
    return torch.linalg.solve(gram+alpha*torch.eye(len(gram),dtype=gram.dtype,device=gram.device),target)


def moments(gram):
    assert abs(float(gram[-1,-1])-1)<1e-9
    mean=gram[:-1,-1]
    covariance=gram[:-1,:-1]-torch.outer(mean,mean)
    return mean,(covariance+covariance.T)/2


def gda_parameters(grams, groups, base_count, tau, relative_floor):
    summaries=[moments(g) for g in grams]
    target=torch.stack([s[1] for s in summaries[:base_count]]).mean(0)
    eps=relative_floor*float(target.diag().mean())
    assert eps>0
    eye=torch.eye(len(target),dtype=target.dtype,device=target.device)
    fitted=[]
    for (mean,cov),n in zip(summaries,groups):
        beta=n/(n+tau)
        chol=torch.linalg.cholesky(beta*cov+(1-beta)*target+eps*eye)
        fitted.append((mean,chol,2*chol.diag().log().sum()))
    return fitted,eps


def gda_score(x, fitted):
    outputs=[]
    for mean,chol,ld in fitted:
        residual=torch.linalg.solve_triangular(chol,(x.double()-mean).T,upper=False).T
        outputs.append(-.5*(residual.square().sum(1)+ld+len(mean)*math.log(2*math.pi)))
    return torch.stack(outputs,1)
