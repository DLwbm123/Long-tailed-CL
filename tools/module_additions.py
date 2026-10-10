"""Small, explicitly adapted modules for the existing statistical competition model."""
import math

import torch
from torch.nn import functional as F

import prototype_coherent as native

ARMS = ('base', 'hierarchy', 'local', 'paced', 'fusion', 'local_transport', 'local_readout', 'local_residual', 'local_auxiliary', 'local_detached', 'local_supervised','local_normalized')
REGIONAL_ARMS = ('local_transport','local_readout','local_residual','local_auxiliary','local_detached','local_supervised','local_normalized')
SUPERVISED_ARMS = ('local_detached','local_supervised','local_normalized')
STAT_ARMS = ('local_residual','local_auxiliary')
# Research groupings from class names, not a validated clinical ontology.
FAMILIES = {
    'ISIC': {0:'melanocytic', 1:'melanocytic', 2:'keratinocytic',
             3:'keratinocytic', 4:'keratinocytic', 5:'keratinocytic',
             6:'vascular', 7:'fibrous'},
    'HK': {0:'barretts', 16:'barretts', 1:'bbps', 2:'bbps',
           3:'polyp_procedure', 4:'polyp_procedure', 13:'polyp_procedure',
           11:'oesophagitis', 12:'oesophagitis',
           **{c:'ulcerative_colitis' for c in range(17,23)},
           **{c:f'leaf_{c}' for c in (5,6,7,8,9,10,14,15)}}}


def metric(bank, arm, dataset, seen, local=None):
    """Keep original moments; extra evidence changes only the quadratic prior."""
    if arm not in ARMS:
        raise ValueError('Unknown module')
    extra = []
    if arm == 'hierarchy':
        families = FAMILIES[dataset]
        for family in sorted({families[c] for c in seen}):
            ids = [i for i,c in enumerate(seen) if families[c] == family]
            if len(ids) > 1:
                extra.append(F.normalize(bank['mu'][ids].mean(0), dim=0))
    if arm=='local' or arm in REGIONAL_ARMS:
        if local is None or local.shape != (len(seen),4,bank['mu'].shape[1]):
            raise ValueError('Missing current/old local aggregate statistics')
        extra = [v/2 for v in F.normalize(local, dim=-1).reshape(-1,local.shape[-1])]
    if not extra:
        return native.metric(bank), dict(extra_columns=0, prior_delta=0.)
    P = torch.cat([native.evidence(bank), torch.stack(extra,dim=1)],dim=1)
    eye = torch.eye(P.shape[0],device=P.device,dtype=P.dtype)
    small = torch.eye(P.shape[1],device=P.device,dtype=P.dtype)+P.T@P
    result = eye-P@torch.linalg.solve(small,P.T)
    result = (result+result.T)/2
    return result, dict(extra_columns=len(extra),prior_delta=float((result-native.metric(bank)).norm()))


def head(bank, arm, dataset, seen, local=None):
    R, audit = metric(bank,arm,dataset,seen,local)
    k = len(bank['n'])
    W = torch.linalg.solve(bank['Q'].sum(0)/k+.001*R,bank['mu'].T/k)
    if not torch.isfinite(W).all():
        raise ValueError('Nonfinite module head')
    return W,R,audit


def local_parts(tokens):
    """CLS-conditioned pooling in four fixed spatial regions; no learned slots."""
    if len(tokens) != 2 or tokens[0].shape != tokens[1].shape:
        raise ValueError('Expected the two existing adapter branches')
    # Existing output is regular branch followed by few branch.
    z = F.normalize(torch.cat([tokens[1],tokens[0]],dim=-1),dim=-1)
    side = math.isqrt(z.shape[1]-1)
    if side*side != z.shape[1]-1 or side < 2:
        raise ValueError('Patch grid is not square')
    patch = z[:,1:].reshape(len(z),side,side,-1)
    result=[]
    for rows in (slice(0,side//2),slice(side//2,side)):
        for cols in (slice(0,side//2),slice(side//2,side)):
            part=patch[:,rows,cols].flatten(1,2)
            weights=(part*z[:,0,None,:]).sum(-1).softmax(1)
            result.append((part*weights[:,:,None]).sum(1))
    result=torch.stack(result,dim=1)
    return F.normalize(result-result.mean(1,keepdim=True),dim=-1)


@torch.no_grad()
def extract(model, loader, budget, with_local):
    xs,ys,locals_ = [],[],[]
    for x,y in loader:
        budget(); capture=[]
        hook=None
        try:
            if with_local:
                hook=model.net.backbone.norm.register_forward_hook(
                    lambda m,a,out:capture.append(out.detach()))
            z=model(x.cuda())
            xs.append(z.cpu().numpy());ys.append(y.numpy())
            if with_local:
                locals_.append(local_parts(capture).cpu())
        finally:
            if hook is not None:hook.remove()
    import numpy as np
    return (torch.as_tensor(np.concatenate(xs),device='cuda',dtype=torch.float64),
            torch.as_tensor(np.concatenate(ys),device='cuda'),
            torch.cat(locals_).to(device='cuda',dtype=torch.float64) if with_local else None)


def local_memory(previous, shift, current, labels):
    if current is None:return None
    means=torch.stack([current[labels==c].mean(0) for c in sorted(labels.unique().tolist())])
    if previous is None:return means
    if shift.shape not in ((current.shape[-1],), current.shape[1:]):
        raise ValueError('Expected global or per-region common shift')
    # shortcut: common regional displacement is not class-specific transport; revisit if evidence supports it.
    return torch.cat([previous+shift[None,...],means])


def local_scores(parts, centers):
    if parts.ndim!=3 or centers.ndim!=3 or parts.shape[1:]!=centers.shape[1:]:
        raise ValueError('Local query and class center dimensions differ')
    scores=torch.einsum('nrd,krd->nk',F.normalize(parts,dim=-1),F.normalize(centers,dim=-1))/parts.shape[1]
    if not torch.isfinite(scores).all():raise ValueError('Nonfinite local readout')
    return scores


def forward_local(model, images):
    capture=[]
    hook=model.net.backbone.norm.register_forward_hook(lambda m,a,out:capture.append(out))
    try:
        global_features=model(images)
        return global_features,local_parts(capture)
    finally:
        hook.remove()


def local_class_loss(parts, centers, labels, alpha, detached):
    if labels.shape!=alpha.shape or labels.shape!=(len(parts),) or not torch.isfinite(alpha).all() or (alpha<0).any():
        raise ValueError('Invalid local classification weights')
    scores=local_scores(parts.detach() if detached else parts,centers.detach().to(parts))
    return (alpha*F.cross_entropy(scores,labels,reduction='none')).sum()


def current_class_alpha(alpha, seen_count, current_count):
    if not 0<current_count<=seen_count:
        raise ValueError('Invalid current/seen class counts')
    return alpha*(seen_count/current_count)


def normalized_self_check():
    weights=torch.tensor([.5,.5,.2,.2,.2,.2,.2],dtype=torch.float64)
    alpha=weights/5
    result=current_class_alpha(alpha,5,2)
    assert torch.equal(current_class_alpha(alpha,5,5),alpha)
    assert torch.allclose(result.sum(),torch.tensor(1.,dtype=result.dtype))
    assert torch.allclose(result[:2].sum(),result[2:].sum())
    logits=torch.tensor([[.1,.2]]*7,dtype=torch.float64,requires_grad=True)
    losses=F.cross_entropy(logits,torch.tensor([0,0,1,1,1,1,1]),reduction='none')
    a=torch.autograd.grad((alpha*losses).sum(),logits,retain_graph=True)[0]
    b=torch.autograd.grad((result*losses).sum(),logits)[0]
    assert torch.allclose(b,2.5*a)
    try:current_class_alpha(alpha,2,3)
    except ValueError:pass
    else:raise AssertionError('Invalid class counts accepted')
    return dict(status='PASS',initial_task_weights_equal=True,current_class_mass_equal=True,expected_auxiliary_mass_one=True,gradient_scale_matches_normalization=True)


def supervised_self_check():
    generator=torch.Generator().manual_seed(133)
    tokens=[torch.randn(6,5,4,generator=generator,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    parts=local_parts(tokens);centers=parts.detach().reshape(3,2,4,8).mean(1)
    labels=torch.arange(3).repeat_interleave(2);alpha=torch.full((6,),1/6,dtype=torch.float64)
    active=local_class_loss(parts,centers,labels,alpha,False)
    inactive=local_class_loss(parts,centers,labels,alpha,True)
    assert active.item()==inactive.item() and not inactive.requires_grad
    grads=torch.autograd.grad(active,tokens)
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
    global_loss=sum(t.square().mean() for t in tokens)
    expected=torch.autograd.grad(global_loss,tokens,retain_graph=True)
    actual=torch.autograd.grad(global_loss+inactive,tokens)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    try:local_class_loss(parts,centers,labels,-alpha,False)
    except ValueError:pass
    else:raise AssertionError('Invalid weights accepted')
    return dict(status='PASS',detached_global_gradients_equal=True,supervised_token_gradients_finite_nonzero=True,extra_trainable_head_parameters=0)


class Updates:
    """Historical parameter state only; no old images or validation feedback."""
    def __init__(self, encoder, arm):
        self.arm=arm
        self.parameters={n:p for n,p in encoder.named_parameters() if p.requires_grad}
        self.previous=None;self.average=None;self.history=[]
        self.rank=0;self.multiplier=1.;self.fisher=None;self.gradient=None

    def begin(self):
        self.start={n:p.detach().cpu().clone() for n,p in self.parameters.items()}
        if self.arm == 'paced' and self.history:
            X=torch.stack(self.history).double()
            values=torch.linalg.eigvalsh(X@X.T).clamp_min(0).flip(0)
            total=values.sum()
            self.rank=(int(torch.searchsorted(values.cumsum(0),.95*total))+1) if total>0 else 0
            self.multiplier=1/math.sqrt(max(1,self.rank))

    def collect(self, loss):
        if self.arm != 'fusion' or self.previous is None:return
        grads=torch.autograd.grad(loss,list(self.parameters.values()),retain_graph=True,allow_unused=True)
        if self.fisher is None:
            self.fisher={n:torch.zeros_like(p,device='cpu') for n,p in self.parameters.items()}
            self.gradient={n:torch.zeros_like(p,device='cpu') for n,p in self.parameters.items()}
            self.samples=0
        for (n,p),g in zip(self.parameters.items(),grads):
            if g is not None:
                v=g.detach().cpu();self.fisher[n]+=v.square();self.gradient[n]+=v
        self.samples+=1

    @torch.no_grad()
    def finish(self, task):
        optimized={n:p.detach().cpu().clone() for n,p in self.parameters.items()}
        audit=dict(history_rank=self.rank,learning_rate_multiplier=self.multiplier,fused=False)
        if self.arm == 'paced':
            self.history.append(torch.cat([(optimized[n]-self.start[n]).flatten() for n in self.parameters]))
        if self.arm == 'fusion':
            if self.previous is not None:
                if not self.samples:raise ValueError('Missing current-task curvature pass')
                f={n:v/self.samples for n,v in self.fisher.items()}
                g={n:v/self.samples for n,v in self.gradient.items()}
                minimum=min(float(v.min()) for v in f.values())
                mean=sum(float(v.sum()) for v in f.values())/sum(v.numel() for v in f.values())
                counts=0;changes=0.;means=0.;fallbacks=0
                for n,p in self.parameters.items():
                    curvature=1.25*(f[n]-minimum)/max(mean-minimum,1e-20)+1.
                    difference=self.average[n]+self.previous[n]-2*optimized[n]
                    valid=difference.abs()>1e-12
                    safe=torch.where(valid,difference,torch.ones_like(difference))
                    beta=torch.where(valid,(1-g[n]/safe)/(curvature+1),1/(curvature+1)).clamp(.001,.499)
                    fused=beta*self.average[n]+beta*self.previous[n]+(1-2*beta)*optimized[n]
                    if not torch.isfinite(fused).all():raise ValueError('Nonfinite fused adapter')
                    p.copy_(fused.to(p));changes+=float((fused-optimized[n]).square().sum())
                    means+=float(beta.sum());counts+=beta.numel();fallbacks+=int((~valid).sum())
                audit.update(fused=True,fusion_delta_norm=math.sqrt(changes),beta_mean=means/counts,
                             zero_displacement_fallback_coordinates=fallbacks,fisher_batches=self.samples)
            self.average=optimized if self.average is None else {
                n:(task-1)/task*self.average[n]+optimized[n]/task for n in optimized}
            self.previous={n:p.detach().cpu().clone() for n,p in self.parameters.items()}
            self.fisher=None;self.gradient=None
        return audit


def self_check():
    torch.manual_seed(128);torch.set_num_threads(2)
    x=torch.randn(18,7,dtype=torch.float64);y=torch.arange(3).repeat_interleave(6)
    bank=native.append(native.empty(7),x,y,y,torch.full((18,),1/6,dtype=x.dtype))
    w,r,_=head(bank,'base','ISIC',[0,1,2]);wn,rn=native.head(bank)
    assert torch.equal(w,wn) and torch.equal(r,rn)
    rw,a=metric(bank,'hierarchy','ISIC',[0,1,2]);assert a['extra_columns']==1 and a['prior_delta']>0
    assert torch.linalg.eigvalsh(rw).min()>0
    tokens=[torch.randn(2,17,4),torch.randn(2,17,4)]
    parts=local_parts(tokens);assert parts.shape==(2,4,8) and torch.isfinite(parts).all()
    lp=torch.randn(18,4,7,dtype=x.dtype);mem=local_memory(None,x.new_zeros(7),lp,y)
    r,a=metric(bank,'local','ISIC',[0,1,2],mem);assert a['extra_columns']==12 and torch.linalg.eigvalsh(r).min()>0
    shift=torch.arange(28,dtype=x.dtype).reshape(4,7)/100
    transported=local_memory(mem,shift,lp,y)
    assert torch.allclose(transported[:3],mem+shift) and torch.equal(transported[3:],mem)
    legacy=local_memory(mem,x[0],lp,y);assert torch.equal(legacy[:3],mem+x[0])
    queries=torch.eye(3,dtype=x.dtype)[:,None,:].repeat(1,4,1)
    score=local_scores(queries,queries)
    assert torch.equal(score,torch.eye(3,dtype=x.dtype)) and torch.equal(local_scores(queries*7,queries*2),score)
    wt,rt,_=head(bank,'local_transport','ISIC',[0,1,2],mem)
    wr,rr,_=head(bank,'local_readout','ISIC',[0,1,2],mem)
    assert torch.equal(wt,wr) and torch.equal(rt,rr)
    layer=torch.nn.Linear(3,2);u=Updates(layer,'paced');u.begin()
    with torch.no_grad():layer.weight.add_(.1)
    u.finish(1);u.begin();assert u.rank==1 and u.multiplier==1
    u.history=[torch.zeros_like(u.history[0]),torch.zeros_like(u.history[0])]
    u.history[0][0]=1;u.history[1][1]=1
    u.begin();assert u.rank==2 and u.multiplier<1
    layer=torch.nn.Linear(3,2);u=Updates(layer,'fusion');u.begin();u.finish(1);u.begin()
    with torch.no_grad():layer.weight.add_(.2)
    u.collect(layer(torch.ones(2,3)).square().mean());a=u.finish(2)
    assert a['fused'] and a['fusion_delta_norm']>0 and .001<=a['beta_mean']<=.499
    print('PASS: exact base head, positive definite hierarchy/local priors, patch shape, historical pacing, finite fusion')


if __name__ == '__main__':self_check()
