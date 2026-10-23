"""TRAIN-only discriminative teacher with an image-geometry gradient budget."""
import torch
import torch.nn.functional as F

def teacher_prob(raw, weight):
    with torch.no_grad():
        x=F.normalize(raw.double(),dim=-1)
        z=torch.cat([x,torch.ones(len(x),1,device=x.device,dtype=x.dtype)],1)@weight.to(x)
        positive=z.clamp_min(0);mass=positive.sum(1,keepdim=True)
        q=positive/mass.clamp_min(1e-12)
        q=torch.where(mass>1e-12,q,torch.full_like(q,1/q.shape[1]))
        assert torch.isfinite(q).all() and torch.allclose(q.sum(1),torch.ones(len(q),device=q.device,dtype=q.dtype))
    return q

def consistent_mask(q, labels):
    return (q.argmax(1)==labels)&(q[torch.arange(len(labels),device=labels.device),labels]>1/q.shape[1]+1e-12)

def class_balanced_kl(logits, q, labels):
    keep=consistent_mask(q,labels)
    if not keep.any():return logits.sum()*0
    per=F.kl_div(F.log_softmax(logits[keep],dim=1),q[keep],reduction='none').sum(1)
    selected=labels[keep]
    return torch.stack([per[selected==c].mean() for c in selected.unique()]).mean()

def guarded_vector(base, teacher):
    nb=base.norm();dot=(teacher*base).sum()
    tangent=teacher-torch.minimum(dot,torch.zeros_like(dot))*base/nb.square().clamp_min(1e-24)
    alpha=torch.minimum(torch.ones_like(nb),.1*nb/tangent.norm().clamp_min(1e-12))
    added=alpha*tangent
    assert torch.isfinite(added).all()
    assert float(added.norm())<=float(.1*nb)+1e-10
    assert float((added*base).sum())>=-1e-10*max(1.,float(nb*added.norm()))
    return added,dict(alpha=float(alpha),conflict=bool(dot<0),base_norm=float(nb),raw_teacher_norm=float(teacher.norm()),added_norm=float(added.norm()))

def selfcheck():
    torch.manual_seed(42)
    for base,teacher in [(torch.randn(17,dtype=torch.double),torch.randn(17,dtype=torch.double)),(torch.ones(17,dtype=torch.double),-torch.ones(17,dtype=torch.double)),(torch.zeros(17,dtype=torch.double),torch.ones(17,dtype=torch.double))]:
        extra,_=guarded_vector(base,teacher)
        assert extra.norm()<=.1*base.norm()+1e-12 and (extra*base).sum()>=-1e-12
    x=torch.randn(6,4,dtype=torch.double,requires_grad=True);w=torch.randn(5,3,dtype=torch.double,requires_grad=True)
    q=teacher_prob(x,w);assert not q.requires_grad
    labels=torch.tensor([0,0,0,1,1,2]);z=x@torch.randn(4,3,dtype=torch.double);loss=class_balanced_kl(z,q,labels)
    assert torch.isfinite(torch.autograd.grad(loss,x)[0]).all()
    assert torch.allclose(teacher_prob(x,torch.zeros_like(w)),torch.full_like(q,1/3))
    return dict(passed=True,parameter_gradient_budget=True,nonconflict=True,zero_base_noop=True,teacher_detached=True)

_HEADS={}

def prepare_anchor(learner, logits, inputs, replay, labels):
    import os
    from pathlib import Path
    ds=os.environ['TEACHER_DATASET'];stage=learner._cur_task;key=(ds,stage)
    if key not in _HEADS:
        head=torch.load(Path(os.environ['TEACHER_ROOT'])/ds/'heads'/f'Biomed_{stage}.private.pt',map_location='cpu',weights_only=False)
        assert head['classes']==learner._total_classes
        _HEADS[key]=head['ridge']
    raw=inputs if replay is None else torch.cat([inputs,replay],0)
    assert len(raw)==len(labels)==len(logits)
    q=teacher_prob(raw,_HEADS[key]);kd=class_balanced_kl(learner._network.logit_scale.exp()*logits,q,labels)
    named=[(name,p) for name,p in learner._network.named_parameters() if p.requires_grad]
    assert any(name=='learned_lorentz_exp.exp_map.0.weight' for name,p in named)
    keep=consistent_mask(q,labels)
    gradients=torch.autograd.grad(kd,[p for name,p in named],retain_graph=True,allow_unused=True) if keep.any() else [None]*len(named)
    learner._teacher_pending=(named,[None if g is None else g.detach() for g in gradients])
    audit=learner.audit.setdefault('teacher',dict(calls=0,examples=0,conflict_batches=0,max_applied_parameter_ratio=0.,min_added_base_cosine=1.,alpha_min=1.,alpha_max=0.,max_KD=0.))
    audit['calls']+=1;audit['examples']+=len(labels);audit['max_KD']=max(audit['max_KD'],float(kd.detach()))
    audit.setdefault('eligible_examples',0);audit.setdefault('zero_eligible_batches',0)
    audit.setdefault('seen_by_class',[0]*q.shape[1]);audit.setdefault('eligible_by_class',[0]*q.shape[1])
    audit['eligible_examples']+=int(keep.sum());audit['zero_eligible_batches']+=int(not keep.any())
    for key,values in [('seen_by_class',labels),('eligible_by_class',labels[keep])]:
        for c,count in enumerate(torch.bincount(values,minlength=q.shape[1]).tolist()):audit[key][c]+=count

def apply_anchor(learner):
    named,teacher=learner._teacher_pending
    rawbase=[None if p.grad is None else p.grad.detach().clone() for name,p in named]
    base=[];teach=[]
    for (name,p),b,t in zip(named,rawbase,teacher):
        b=torch.zeros_like(p) if b is None else b;t=torch.zeros_like(p) if t is None else t
        if name=='learned_lorentz_exp.exp_map.0.weight' and learner._cur_task>0:
            b=learner._network.learned_lorentz_exp.project_gradient(b)
            t=learner._network.learned_lorentz_exp.project_gradient(t)
        base.append(b.double().flatten());teach.append(t.double().flatten())
    base=torch.cat(base);teach=torch.cat(teach);extra,record=guarded_vector(base,teach)
    offset=0
    for (name,p),raw,t in zip(named,rawbase,teacher):
        a=extra[offset:offset+p.numel()].view_as(p).to(p);offset+=p.numel()
        if raw is not None or t is not None:p.grad=(torch.zeros_like(p) if raw is None else raw)+a
    a=learner.audit['teacher'];a['conflict_batches']+=int(record['conflict']);a['alpha_min']=min(a['alpha_min'],record['alpha']);a['alpha_max']=max(a['alpha_max'],record['alpha'])
    learner._teacher_pending=(named,base)

def verify_anchor(learner):
    named,baseline=learner._teacher_pending
    actual=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().double().flatten() for name,p in named])
    added=actual-baseline;nb=float(baseline.norm());na=float(added.norm())
    ratio=na/max(nb,1e-12);cos=float((added*baseline).sum()/max(na*nb,1e-24))
    # Native float32 projection can add a small second-projection roundoff residual.
    assert ratio<=.1001 and (cos>=-1e-4 or na<=1e-6*max(nb,1e-12)),(ratio,cos)
    a=learner.audit['teacher'];a['max_applied_parameter_ratio']=max(a['max_applied_parameter_ratio'],ratio);a['min_added_base_cosine']=min(a['min_added_base_cosine'],cos)
    learner._teacher_pending=None
