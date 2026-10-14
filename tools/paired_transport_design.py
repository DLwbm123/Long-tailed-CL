"""CPU-only paired affine moment probe; no medical loader or training entry point."""
import math

import torch

import prototype_coherent as native

VERSION='paired_affine_moment_CPU_design_v1'
RELATIVE_RIDGE=.001


def finite(*values):
    if not values or any(x.dtype!=torch.float64 or x.device!=values[0].device or not torch.isfinite(x).all() for x in values):
        raise ValueError('Finite same-device float64 statistics required')


@torch.no_grad()
def fit(before,after,labels,before_ids,after_ids,source_frame,target_frame,space):
    finite(before,after)
    if (before.ndim!=2 or after.shape!=before.shape or not len(before)
            or labels.shape!=(len(before),) or labels.dtype!=torch.long or labels.device!=before.device
            or before_ids.dtype!=torch.long or after_ids.dtype!=torch.long
            or before_ids.shape!=labels.shape or not torch.equal(before_ids,after_ids)
            or len(before_ids.unique())!=len(before_ids)
            or not isinstance(source_frame,str) or not source_frame or not isinstance(target_frame,str) or not target_frame
            or source_frame==target_frame or space not in ('global','region_0','region_1','region_2','region_3')):
        raise ValueError('Aligned distinct current rows, frames and feature space required')
    d=before.shape[1];classes=sorted(labels.unique().tolist())
    weight=before.new_zeros(len(before))
    for c in classes:
        take=labels==c;weight[take]=1/(len(classes)*int(take.sum()))
    center=(weight[:,None]*before).sum(0)
    shift=torch.stack([(after-before)[labels==c].mean(0) for c in classes]).mean(0)
    z=weight.sqrt()[:,None]*(before-center)
    delta=weight.sqrt()[:,None]*(after-before-shift)
    tolerance=32*torch.finfo(before.dtype).eps*max(1.,float(before.abs().max()),float(after.abs().max()))
    left=before.new_empty(d,0);right=before.new_empty(0,d);rank=0;ridge=0.;effective=0.
    if float(delta.abs().max())>tolerance:
        u,s,vh=torch.linalg.svd(z,full_matrices=False)
        threshold=torch.finfo(before.dtype).eps*max(z.shape)*s[0]
        rank=int((s>threshold).sum())
        if rank:
            s=s[:rank];left=vh[:rank].T
            ridge=RELATIVE_RIDGE*float(s.square().sum())/rank
            right=(s/(s.square()+ridge))[:,None]*(u[:,:rank].T@delta)
            effective=float((s.square()/(s.square()+ridge)).sum())
    offset=shift-(center@left)@right
    state=dict(version=VERSION,left=left.detach(),right=right.detach(),offset=offset.detach(),
               source_frame=source_frame,target_frame=target_frame,space=space,
               rank=rank,effective_rank=effective,ridge=ridge,rows=len(before),classes=len(classes),
               outside_input_span_is_identity=True,true_unobserved_drift_identified=False,
               zero_drift=bool(torch.equal(before,after)))
    residual=predict(before,state)-after
    state['paired_weighted_MSE']=float((weight*residual.square().sum(1)).sum())
    return state


def validate(state,like,space):
    if state.get('version')!=VERSION or state.get('space')!=space:
        raise ValueError('Map version/feature-space mismatch')
    left,right,offset=state['left'],state['right'],state['offset'];finite(like,left,right,offset)
    d=like.shape[-1]
    if (left.ndim!=2 or left.shape[0]!=d or right.shape!=(left.shape[1],d) or offset.shape!=(d,)
            or any(x.requires_grad for x in (left,right,offset))
            or state['source_frame']==state['target_frame']):
        raise ValueError('Invalid detached low-rank affine map')


def predict(x,state):
    validate(state,x,state['space'])
    return x+(x@state['left'])@state['right']+state['offset']


def framed(bank,frame):
    if not isinstance(frame,str) or not frame:
        raise ValueError('Explicit source frame required')
    if '_frame' in bank and bank['_frame']!=frame:
        raise ValueError('Cannot relabel already transported statistics')
    return dict(bank,_frame=frame,_transport_version=VERSION)


@torch.no_grad()
def push_bank(snapshot,state):
    finite(snapshot['mu'],snapshot['Q']);validate(state,snapshot['mu'],'global')
    k,d=snapshot['mu'].shape
    if (snapshot.get('_frame')!=state['source_frame'] or snapshot['Q'].shape!=(k,d,d)
            or len(snapshot['n'])!=k or any(type(n) is not int or n<=0 for n in snapshot['n'])):
        raise ValueError('Use the immutable task-start bank in the map source frame')
    if state['zero_drift']:
        return dict(snapshot,_frame=state['target_frame'])
    left,right,b=state['left'],state['right'],state['offset'];mu=snapshot['mu'];q=snapshot['Q']
    transformed=mu+(mu@left)@right
    ql=q@left;qb=ql@right
    second=q+qb+qb.transpose(1,2)+right.T@(left.T@ql)@right
    second+=transformed[:,:,None]*b[None,None,:]+b[None,:,None]*transformed[:,None,:]+torch.outer(b,b)[None]
    second=(second+second.transpose(1,2))/2
    components=[]
    norm=float(torch.linalg.matrix_norm(right,ord=2)) if right.numel() else 0.
    lo,hi=max(0.,1-norm)**2,(1+norm)**2
    for c in snapshot['components']:
        original=c.get('radius_bounds')
        if original is None:
            radius=float(c['radius']);original=(radius,radius)
        if len(original)!=2 or not all(math.isfinite(a) and a>=0 for a in original) or original[0]>original[1]:
            raise ValueError('Valid component radius or inherited bounds required')
        row=dict(c,center=predict(c['center'],state))
        if right.numel():
            row.update(radius=None,radius_bounds=(lo*original[0],hi*original[1]),radius_identified=False)
        components.append(row)
    return dict(snapshot,mu=transformed+b,Q=second,n=list(snapshot['n']),components=components,
                _frame=state['target_frame'],_transport_version=VERSION)


@torch.no_grad()
def push_regions(snapshot,maps):
    x=snapshot['values']
    if len(maps)!=4 or x.ndim!=3 or x.shape[1]!=4:
        raise ValueError('Four independent paired regional spaces required')
    for i,m in enumerate(maps):
        validate(m,x[:,i],f'region_{i}')
        if snapshot['frame']!=m['source_frame'] or m['target_frame']!=maps[0]['target_frame']:
            raise ValueError('Regional map source/target frame differs')
    return dict(values=torch.stack([predict(x[:,i],m) for i,m in enumerate(maps)],1),frame=maps[0]['target_frame'])


def append_current(old,x,y,groups,weights,frame):
    if old.get('_frame')!=frame:
        raise ValueError('Current statistics must join the current frame once')
    return framed(native.append(old,x,y,groups,weights),frame)
