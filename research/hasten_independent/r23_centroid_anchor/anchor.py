"""Equal-per-step arrived TRAIN centroid supervision; H0 remains primary."""
import torch
import torch.nn.functional as F

def class_balanced_ce(logits, labels):
    loss=F.cross_entropy(logits,labels,reduction='none')
    return torch.stack([loss[labels==c].mean() for c in labels.unique()]).mean()

def guarded_vector(base, auxiliary):
    nb=base.norm();dot=(auxiliary*base).sum()
    tangent=auxiliary-torch.minimum(dot,torch.zeros_like(dot))*base/nb.square().clamp_min(1e-24)
    alpha=torch.minimum(torch.ones_like(nb),.1*nb/tangent.norm().clamp_min(1e-12))
    added=alpha*tangent
    assert torch.isfinite(added).all()
    assert float(added.norm())<=float(.1*nb)+1e-10
    assert float((added*base).sum())>=-1e-10*max(1.,float(nb*added.norm()))
    return added,dict(alpha=float(alpha),conflict=bool(dot<0),base_norm=float(nb),raw_auxiliary_norm=float(auxiliary.norm()),added_norm=float(added.norm()))

def selfcheck():
    torch.manual_seed(42)
    z=torch.randn(5,3,dtype=torch.double,requires_grad=True);y=torch.tensor([0,0,1,2,2])
    loss=class_balanced_ce(z,y);idx=torch.tensor([0,1,0,1,2,3,4])
    assert torch.allclose(loss,class_balanced_ce(z[idx],y[idx]))
    perm=torch.tensor([4,2,0,3,1]);assert torch.allclose(loss,class_balanced_ce(z[perm],y[perm]))
    grad=torch.autograd.grad(loss,z)[0];assert torch.isfinite(grad).all() and (grad.norm(dim=1)>0).all()
    for base,aux in [(torch.randn(17,dtype=torch.double),torch.randn(17,dtype=torch.double)),(torch.ones(17,dtype=torch.double),-torch.ones(17,dtype=torch.double)),(torch.zeros(17,dtype=torch.double),torch.ones(17,dtype=torch.double))]:
        extra,_=guarded_vector(base,aux);assert extra.norm()<=.1*base.norm()+1e-12 and (extra*base).sum()>=-1e-12
    return dict(passed=True,class_balance_duplication_invariant=True,permutation_invariant=True,full_row_gradients=True,parameter_budget=True,nonconflict=True,zero_base_noop=True)

def centroid_loss(net, text_features, classes):
    from utils import lorentz as geometry
    means=net.fc.weight[:classes].detach()
    assert means.shape==(classes,512) and torch.isfinite(means).all()
    assert (means.norm(dim=1)>0).all()
    image_features=net.trans_hyper(net.Image_encode(means))
    scores=-geometry.pairwise_dist(image_features,text_features,net.get_curv())
    assert scores.shape==(classes,classes) and torch.isfinite(scores).all()
    labels=torch.arange(classes,device=scores.device)
    return F.cross_entropy(net.logit_scale.exp()*scores,labels)

def prepare_anchor(learner, logits, inputs, replay, labels, text_features):
    classes=learner._total_classes
    assert logits.shape==(len(labels),classes) and text_features.shape[0]==classes
    # Every arrived TRAIN class contributes its existing raw NCM mean once per step.
    # Means are not normalized, sampled, updated, or reused as inference replacements.
    aux=centroid_loss(learner._network,text_features,classes)
    named=[(name,p) for name,p in learner._network.named_parameters() if p.requires_grad]
    assert any(name=='learned_lorentz_exp.exp_map.0.weight' for name,p in named)
    gradients=torch.autograd.grad(aux,[p for name,p in named],retain_graph=True,allow_unused=True)
    assert any(g is not None and float(g.norm())>0 for g in gradients)
    learner._anchor_pending=(named,[None if g is None else g.detach() for g in gradients])
    audit=learner.audit.setdefault('centroid_anchor',dict(calls=0,examples=0,source_training_examples=0,conflict_batches=0,max_applied_parameter_ratio=0.,min_added_base_cosine=1.,alpha_min=1.,alpha_max=0.,max_auxiliary_loss=0.,seen_by_class=[0]*classes,source='existing_arrived_TRAIN_raw_NCM_means',no_sampling=True,detached_means=True))
    audit['calls']+=1;audit['examples']+=classes;audit['source_training_examples']+=len(labels)
    audit['max_auxiliary_loss']=max(audit['max_auxiliary_loss'],float(aux.detach()))
    audit['seen_by_class']=[count+1 for count in audit['seen_by_class']]

def apply_anchor(learner):
    named,auxiliary=learner._anchor_pending
    rawbase=[None if p.grad is None else p.grad.detach().clone() for name,p in named]
    base=[];aux=[]
    for (name,p),b,t in zip(named,rawbase,auxiliary):
        b=torch.zeros_like(p) if b is None else b;t=torch.zeros_like(p) if t is None else t
        if name=='learned_lorentz_exp.exp_map.0.weight' and learner._cur_task>0:
            b=learner._network.learned_lorentz_exp.project_gradient(b)
            t=learner._network.learned_lorentz_exp.project_gradient(t)
        base.append(b.double().flatten());aux.append(t.double().flatten())
    base=torch.cat(base);aux=torch.cat(aux);extra,record=guarded_vector(base,aux)
    offset=0
    for (name,p),raw,t in zip(named,rawbase,auxiliary):
        a=extra[offset:offset+p.numel()].view_as(p).to(p);offset+=p.numel()
        if raw is not None or t is not None:p.grad=(torch.zeros_like(p) if raw is None else raw)+a
    a=learner.audit['centroid_anchor'];a['conflict_batches']+=int(record['conflict']);a['alpha_min']=min(a['alpha_min'],record['alpha']);a['alpha_max']=max(a['alpha_max'],record['alpha'])
    learner._anchor_pending=(named,base)

def verify_anchor(learner):
    named,baseline=learner._anchor_pending
    actual=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().double().flatten() for name,p in named])
    added=actual-baseline;nb=float(baseline.norm());na=float(added.norm())
    ratio=na/max(nb,1e-12);cos=float((added*baseline).sum()/max(na*nb,1e-24))
    # Native float32 projection can add a small second-projection roundoff residual.
    assert ratio<=.1001 and (cos>=-1e-4 or na<=1e-6*max(nb,1e-12)),(ratio,cos)
    a=learner.audit['centroid_anchor'];a['max_applied_parameter_ratio']=max(a['max_applied_parameter_ratio'],ratio);a['min_added_base_cosine']=min(a['min_added_base_cosine'],cos)
    learner._anchor_pending=None
