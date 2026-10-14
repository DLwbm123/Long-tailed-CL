"""Detached task-start moment transport; native objectives and current statistics stay intact."""
import torch
import paired_transport_design as affine
import prototype_coherent as native
import module_additions as modules
from run_prototype_coherent import common_shift

VERSION='paired_moment_transport_v1'
MODES=(('base','global'),('local_transport','global_regional'),('local_transport','global'))


def label(module,mode):
    return 'pmt' if module=='base' else 'local_transport_pmt'+('_global_only' if mode=='global' else '')


@torch.no_grad()
@torch.no_grad()
def extract(model,loader,budget,with_local):
    xs=[];ys=[];regions=[];ids=[]
    for x,y,row_ids in loader:
        budget();capture=[];hook=None
        try:
            if with_local:hook=model.net.backbone.norm.register_forward_hook(lambda m,a,out:capture.append(out.detach()))
            z=model(x.cuda());xs.append(z.double());ys.append(y.to(z.device));ids.append(row_ids.to(z.device))
            if with_local:regions.append(modules.local_parts(capture).double())
        finally:
            if hook is not None:hook.remove()
    return torch.cat(xs),torch.cat(ys),torch.cat(regions) if with_local else None,torch.cat(ids)


def summary(state):
    return {k:v for k,v in state.items() if type(v) in (int,float,bool)} | dict(
        linear_delta_frobenius=float((state['left']@state['right']).norm()),offset_norm=float(state['offset'].norm()))


@torch.no_grad()
def prepare(snapshot,local_snapshot,before,current,before_local,current_local,y,ids,after_ids,group,weights,source,target,mode):
    state=affine.fit(before,current,y,ids,after_ids,source,target,'global')
    snapshot=affine.framed(snapshot,source)
    old=affine.push_bank(snapshot,state)
    candidate=affine.append_current(old,current,y,group,weights,target)
    maps=[];local=None
    if current_local is not None:
        if mode=='global_regional':
            maps=[affine.fit(before_local[:,r],current_local[:,r],y,ids,after_ids,source,target,f'region_{r}') for r in range(4)]
            moved=None if local_snapshot is None else affine.push_regions(dict(values=local_snapshot,frame=source),maps)['values']
            means=torch.stack([current_local[y==c].mean(0) for c in sorted(y.unique().tolist())])
            local=means if moved is None else torch.cat([moved,means])
        else:
            local=modules.local_memory(local_snapshot,common_shift(before_local,current_local,y),current_local,y)
    elif mode=='global_regional':raise ValueError('Regional PMT requires independently paired regions')
    audit=dict(**summary(state),regional_maps=len(maps),detached=True,task_start_snapshot=True,
        current_statistics_unmapped=True,old_radius_unknown=sum(c['radius'] is None for c in old['components']),
        regional_rank_sum=sum(m['rank'] for m in maps),regional_residual_sum=sum(m['paired_weighted_MSE'] for m in maps),
        regional_delta_norm_sum=sum(summary(m)['linear_delta_frobenius'] for m in maps),
        old_radius_lower_min=min((float(c.get('radius_bounds',(c['radius'],c['radius']))[0]) for c in old['components']),default=0.),
        old_radius_upper_max=max((float(c.get('radius_bounds',(c['radius'],c['radius']))[1]) for c in old['components']),default=0.))
    return old,candidate,local,audit


def checkpoint_bank(bank):
    from run_prototype_coherent import cpu_bank
    return dict(cpu_bank(bank),_frame=bank['_frame'],_transport_version=bank['_transport_version'])


@torch.no_grad()
def shadow(device,ds,mode,dimension=1536,full_rank=False):
    """Synthetic highest-bank probe; no medical observations or optimizer updates."""
    def bank(x,y,k):
        weights=x.new_zeros(len(x))
        for c in range(k):weights[y==c]=1/int((y==c).sum())
        return native.append(native.empty(x.shape[1],x.device),x,y,y,weights)
    import core1_competition as competition
    from torch.nn.functional import normalize
    d=dimension;oldk=6 if ds=='ISIC' else 21;k=oldk+2
    gen=torch.Generator(device=device).manual_seed(1074005)
    basis=torch.eye(d,device=device,dtype=torch.float64) if full_rank else torch.linalg.qr(torch.randn(d,min(8,d),generator=gen,device=device,dtype=torch.float64)).Q
    yy=torch.arange(oldk,device=device).repeat_interleave(3)
    xx=normalize(torch.randn(len(yy),basis.shape[1],generator=gen,device=device,dtype=torch.float64)@basis.T,dim=1)
    snapshot=affine.framed(bank(xx,yy,oldk),'shadow_start')
    per_class=d//2+1 if full_rank else 8
    cy=torch.arange(2,device=device).repeat_interleave(per_class)+oldk
    before=normalize(torch.randn(len(cy),basis.shape[1],generator=gen,device=device,dtype=torch.float64)@basis.T,dim=1)
    current=before+.02*(before@basis[:,:1])@basis[:,:1].T+.001*basis[:,0]
    ids=torch.arange(len(cy),device=device);local_before=torch.stack([before+.001*(r+1)*basis[:,1] for r in range(4)],1)
    local_after=local_before+.003*basis[:,2]+.007*local_before[:,:,0:1]*basis[:,3]
    local_snapshot=torch.stack([snapshot['mu']+.001*r*basis[:,1] for r in range(4)],1)
    weights=before.new_full((len(cy),),1/per_class)
    old,candidate,local,audit=prepare(snapshot,local_snapshot if mode[0]=='local_transport' else None,before,current,
        local_before if mode[0]=='local_transport' else None,local_after if mode[0]=='local_transport' else None,
        cy,ids,ids,cy,weights,'shadow_start','shadow_end',mode[1])
    state=affine.fit(before,current,cy,ids,ids,'shadow_start','shadow_end','global')
    virtual=bank(torch.cat([affine.predict(xx,state),current]),torch.cat([yy,cy]),k)
    moment_error=max(float((candidate[key]-virtual[key]).abs().max()) for key in ('mu','Q'))
    if moment_error>1e-10:raise ValueError('Highest-bank virtual sample moments differ')
    R,_=modules.metric(candidate,mode[0],ds,list(range(k)),local)
    h,a=competition.bank_head(candidate,competition.competition(candidate,native.head(candidate)[0]),regularizer=R)
    inv,cross=native.proximal_base(old,R,k);target=torch.nn.functional.one_hot(cy,k).double();alpha=weights/k
    nh=native.proximal_head(current,target,alpha,inv,cross,h)
    q,mu,mass=competition.moments(old,current,cy,alpha,k)
    w,b=competition.solve(q,mu,mass,competition.competition(candidate,h),previous=h,proximal=.01,
        inverse=inv,x=current,alpha=alpha,native=nh,regularizer=R)
    if not torch.isfinite(w).all() or max(a['relative_residual'],b['relative_residual'])>1e-8:raise ValueError('Synthetic highest-bank solve failed')
    return dict(synthetic_only=True,optimizer_updates=0,formal_statistics_written=False,dimension=d,full_rank_memory_probe=full_rank,old_classes=oldk,
        seen_classes=k,virtual_moment_error=moment_error,relative_residual=max(a['relative_residual'],b['relative_residual']),**audit)
