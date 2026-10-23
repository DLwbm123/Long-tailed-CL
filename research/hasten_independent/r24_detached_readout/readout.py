"""Supervised readout has its own optimizer and no gradient path to H0."""
import torch
from torch import nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self,classes,device):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(classes,512,device=device))
        self.bias=nn.Parameter(torch.zeros(classes,device=device))
    def forward(self,x):return F.linear(F.normalize(x.detach(),dim=-1),self.weight,self.bias)

def expand(previous,classes,device):
    head=Head(classes,device)
    if previous is not None:
        old=len(previous.bias);assert old<=classes
        with torch.no_grad():head.weight[:old].copy_(previous.weight);head.bias[:old].copy_(previous.bias)
        assert torch.equal(head.weight[:old],previous.weight) and torch.equal(head.bias[:old],previous.bias)
    return head

def balanced_ce(logits,labels):
    losses=F.cross_entropy(logits,labels,reduction='none')
    return torch.stack([losses[labels==c].mean() for c in labels.unique()]).mean()

def prepare_readout(learner,logits,inputs,replay,labels):
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    n=learner._total_classes
    if 'detached_readout' not in learner.audit:
        learner._readout=expand(getattr(learner,'_readout',None),n,inputs.device)
        head=learner._readout
        assert not ({id(p) for p in head.parameters()}&{id(p) for p in learner._network.parameters()})
        learner._head_optimizer=torch.optim.AdamW(head.parameters(),lr=learner.init_lr,weight_decay=learner.weight_decay)
        learner._head_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(learner._head_optimizer,T_max=learner.tuned_epoch,eta_min=learner.min_lr)
        learner.audit['detached_readout']=dict(calls=0,examples=0,optimizer_steps=0,rng_preserved_calls=0,core_gradient_none_proofs=0,max_loss=0.,seen_by_class=[0]*n,core_optimizer_disjoint=True,zero_new_rows=True,old_rows_restored_exact=True)
    head=learner._readout;a=learner.audit['detached_readout']
    x=inputs if replay is None else torch.cat([inputs,replay],dim=0)
    assert len(x)==len(labels) and logits.shape==(len(labels),n)
    loss=balanced_ce((learner._network.logit_scale.exp()*logits).detach()+head(x),labels)
    assert torch.isfinite(loss)
    if a['calls']==0:
        core=[p for p in learner._network.parameters() if p.requires_grad]
        g=torch.autograd.grad(loss,core,allow_unused=True,retain_graph=True)
        assert all(v is None for v in g);a['core_gradient_none_proofs']+=1
    learner._head_optimizer.zero_grad();loss.backward()
    for p in head.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    learner._head_optimizer.step()
    for p in head.parameters():assert torch.isfinite(p).all()
    a['calls']+=1;a['optimizer_steps']+=1;a['examples']+=len(labels);a['max_loss']=max(a['max_loss'],float(loss.detach()))
    for i,count in enumerate(torch.bincount(labels,minlength=n).tolist()):a['seen_by_class'][i]+=count
    if a['calls']%len(learner.train_loader)==0:learner._head_scheduler.step()
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    a['rng_preserved_calls']+=1

def head_state(learner):
    return {k:v.detach().clone() for k,v in learner._readout.state_dict().items()}

def restore_readout(learner,state):
    h=Head(len(state['bias']),learner._device);h.load_state_dict(state,strict=True);learner._readout=h

def residual_score(learner,x,hyper):
    # CE trains scaled logits; convert back to original unscaled hyper score units.
    residual=learner._readout(x)/learner._network.logit_scale.exp()
    assert residual.shape==hyper.shape and torch.isfinite(residual).all()
    return hyper+residual
