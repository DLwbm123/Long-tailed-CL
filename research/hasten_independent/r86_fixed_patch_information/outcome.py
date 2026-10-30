import torch
def counts(pred,y,n,ids,qa,qb):
    out={}
    for tag,subset in [('full',None),('Qa',set(qa)),('Qb',set(qb))]:
        mask=torch.ones(len(y),dtype=torch.bool) if subset is None else torch.tensor([i in subset for i in ids])
        yy=y[mask];pp=pred[mask]
        out[tag]=dict(counts=torch.bincount(yy,minlength=n).tolist(),correct=torch.bincount(yy[yy==pp],minlength=n).tolist())
    return out
