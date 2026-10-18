"""Independent argmax witness checks; exact fallback uses stored dyadic values."""
import numpy as np

def dyadic(v):
    out=[]
    for x in v:
        n,d=float(x).as_integer_ratio();out.append((n,d.bit_length()-1))
    return out

def exact_sign(row,a,b):
    # Compute x*a-x*b exactly, without first rounding a-b.
    terms=[]
    for (xn,xe),(an,ae),(bn,be) in zip(row,a,b):
        if xn and an:terms.append((xn*an,xe+ae))
        if xn and bn:terms.append((-xn*bn,xe+be))
    if not terms:return 0
    e=max(p for _,p in terms);value=sum(n<<(e-p) for n,p in terms)
    return (value>0)-(value<0)

def verify(X,y,W,k):
    assert X.dtype==W.dtype==np.float64 and np.isfinite(W).all()
    scores=X@W;pred=scores.argmax(1);classes=W.shape[1]
    counts=[int(((pred==c)&(y==c)).sum()) for c in range(classes)]
    eps=np.finfo(np.float64).eps;gamma=4*X.shape[1]*eps/(1-4*X.shape[1]*eps)
    error=gamma*(np.abs(X)@np.abs(W));cert=np.zeros(len(y),dtype=bool);wd=[dyadic(W[:,c]) for c in range(classes)];ambiguous=0
    for i in np.flatnonzero(pred==y):
        c=int(y[i]);row=None;ok=True
        for j in range(classes):
            if j==c:continue
            gap=scores[i,c]-scores[i,j];bound=error[i,c]+error[i,j]
            if gap>bound:continue
            if row is None:row=dyadic(X[i]);ambiguous+=1
            sign=exact_sign(row,wd[c],wd[j])
            if sign<0 or (sign==0 and j<c):ok=False;break
        cert[i]=ok
    certified=[int((cert&(y==c)).sum()) for c in range(classes)]
    finite=bool(np.isfinite(scores).all())
    return dict(numerical_correct=counts,certified_correct=certified,quota=list(map(int,k)),meets_all=finite and all(a>=b for a,b in zip(counts,k)) and all(a>=b for a,b in zip(certified,k)),ambiguous_rows_exact_checked=ambiguous,finite_scores=finite)

def selfcheck():
    row=dyadic([1.,.5]);a=dyadic([1.,1.]);b=dyadic([1.,0.])
    assert exact_sign(row,a,b)==1 and exact_sign(row,b,a)==-1 and exact_sign(row,a,a)==0
    X=np.array([[1.,0.],[0.,1.]],dtype=np.float64);W=np.array([[0.,0.],[0.,1.]],dtype=np.float64);y=np.array([0,1])
    q=verify(X,y,W,[1,1]);assert q['meets_all'] and q['ambiguous_rows_exact_checked']==1
