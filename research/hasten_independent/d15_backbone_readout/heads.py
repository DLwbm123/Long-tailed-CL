"""Fixed class-balanced ridge and cosine NCM; TRAIN sufficient statistics only."""
import torch

def add_class(x):
    x=x.double();z=torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)
    return z.T@z/len(z),z.mean(0)

def solve(gram,means):
    c=len(means);g=gram/c;h=torch.stack(means,1)/c
    lam=.01*torch.trace(g)/len(g)
    ridge=torch.linalg.solve(g+lam*torch.eye(len(g),dtype=g.dtype),h)
    proto=torch.nn.functional.normalize(torch.stack(means)[:,:-1],dim=1)
    return dict(ridge=ridge,prototype=proto,lambda_value=float(lam),classes=c)

def check():
    torch.manual_seed(17)
    x=torch.nn.functional.normalize(torch.randn(31,7,dtype=torch.double),dim=1);y=torch.tensor([0]*7+[1]*9+[2]*15)
    stats=[add_class(x[y==c]) for c in range(3)];head=solve(sum(a for a,b in stats),[b for a,b in stats])
    z=torch.cat([x,torch.ones(31,1,dtype=x.dtype)],1);w=1/(3*torch.bincount(y)[y].double());g=z.T@(w[:,None]*z);h=z.T@(w[:,None]*torch.nn.functional.one_hot(y).double())
    expected=torch.linalg.solve(g+.01*torch.trace(g)/8*torch.eye(8,dtype=torch.double),h)
    error=float((head['ridge']-expected).abs().max());assert error<1e-12
    assert torch.allclose(head['prototype'],torch.stack([torch.nn.functional.normalize(x[y==c].mean(0),dim=0) for c in range(3)]))
    return dict(direct_vs_sufficient_max_error=error,intercept_regularized=True,passed=True)

if __name__=='__main__':print(check())
