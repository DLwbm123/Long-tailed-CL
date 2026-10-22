"""Fixed landmark coordinates and exact patient/class-mass analytical prototypes."""
import hashlib
import torch
import numpy as np
ARMS=['LINEAR_BALANCED','KERNEL_NATURAL','KERNEL_BALANCED']

def select_anchors(rows,number,seed):
    patients={}
    for i,r in enumerate(rows):patients.setdefault((r['label'],r['group']),[]).append(i)
    key=lambda g,p='':hashlib.sha256(f'{seed}:{g[0]}:{g[1]}:{p}'.encode()).digest()
    groups=sorted(patients,key=lambda g:(key(g),g));chosen=[];used=set()
    for c in sorted({r['label'] for r in rows}):
        g=next(g for g in groups if g[0]==c and g[1] not in used);chosen.append(g);used.add(g[1])
    for g in groups:
        if len(chosen)>=number:break
        if g[1] not in used:chosen.append(g);used.add(g[1])
    assert len(chosen)==number

    return [min(patients[g],key=lambda i:(key(g,rows[i]['relative_path']),i)) for g in chosen]

def fixed_reference(features,cfg):
    anchors=features.double();distance=torch.cdist(anchors,anchors).square()
    sigma=distance[torch.triu(torch.ones_like(distance,dtype=torch.bool),diagonal=1)].median().clamp_min(1e-12)
    center=torch.exp(-distance/sigma).mean(0)
    return dict(anchors=anchors.cpu(),sigma=sigma.cpu(),center=center.cpu())

def feature_map(x,ref,kind):
    x=x.double()
    if kind=='linear':phi=x
    else:
        anchors=ref['anchors'].to(x.device);dist=(x.square().sum(1,keepdim=True)+anchors.square().sum(1)[None]-2*x@anchors.T).clamp_min(0)
        phi=torch.exp(-dist/ref['sigma'].to(x.device))-ref['center'].to(x.device)
    phi=phi/phi.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return torch.cat([phi,torch.ones(len(phi),1,device=x.device,dtype=x.dtype)],1)/(2**.5)

def patient_weights(rows):
    counts={}
    for r in rows:counts[r['group']]=counts.get(r['group'],0)+1
    n=len(counts);return np.array([1/(n*counts[r['group']]) for r in rows]),n

def moments(phi,weights):
    w=torch.as_tensor(weights,device=phi.device,dtype=torch.float64);return phi.T@(phi*w[:,None]),(phi*w[:,None]).sum(0)

def system(records,arm,lam):
    kind='linear' if arm=='LINEAR_BALANCED' else 'kernel';cs=[r[kind] for r in records];masses=torch.tensor([r['patients'] for r in records],device=cs[0]['gram'].device,dtype=torch.float64)
    masses=masses/masses.sum() if arm=='KERNEL_NATURAL' else torch.ones_like(masses)/len(records)
    gram=sum(w*r['gram'] for w,r in zip(masses,cs));rhs=torch.stack([w*r['mean'] for w,r in zip(masses,cs)],1)
    mat=gram+lam*torch.eye(len(gram),device=gram.device,dtype=torch.float64)
    return mat,rhs,masses

def solve(records,arm,lam):
    mat,rhs,mass=system(records,arm,lam);head=torch.linalg.solve(mat,rhs);residual=float((mat@head-rhs).abs().max());assert residual<1e-9 and torch.isfinite(head).all()
    return head,residual
