"""Exact executed categorical distribution with a frozen verifier prior."""
import torch
from torch import nn
from .state_features import CLASS_DIM, ACTION_DIM


@torch.no_grad()
def verifier_prior(scores):
    assert scores.ndim==1 and len(scores)>0 and torch.isfinite(scores).all()
    n=len(scores); q=torch.full_like(scores,.1/n)
    # Stable tie break fixed before results, including zero as a legal top-k.
    top=torch.argsort(scores,descending=True,stable=True)[:min(4,n)]
    q[top]+=.4/len(top); q[0]+=.5
    assert bool((q>0).all())
    return q/q.sum()


class ResidualPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(CLASS_DIM,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.class_action=nn.Sequential(nn.Linear(64+ACTION_DIM,64),nn.Tanh())
        self.head=nn.Sequential(nn.Linear(128+ACTION_DIM,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('class_mean',torch.zeros(CLASS_DIM));self.register_buffer('class_std',torch.ones(CLASS_DIM))
        self.register_buffer('action_mean',torch.zeros(ACTION_DIM));self.register_buffer('action_std',torch.ones(ACTION_DIM))

    def log_probs(self, classes, actions, prior):
        assert len(actions)==len(prior) and bool((prior>0).all())
        assert abs(float(prior.sum())-1)<1e-5
        x=self.encoder((classes-self.class_mean)/self.class_std)
        actions=(actions-self.action_mean)/self.action_std
        if actions.ndim==2:actions=actions[:,None].expand(-1,len(x),-1)
        assert actions.shape[1]==len(x)
        conditioned=self.class_action(torch.cat([x[None].expand(len(actions),-1,-1),actions],-1))
        state=torch.cat([conditioned.mean(1),conditioned.max(1).values],-1)
        b=2*torch.tanh(self.head(torch.cat([state,actions.mean(1)],-1)).squeeze(-1))
        return (prior.detach().log()+b).log_softmax(0)


def sft_loss(logp, nearopt):
    assert bool(nearopt.any())
    return -torch.logsumexp(logp[nearopt.to(logp.device)],0)


def ppo_loss(new_logp, old_logp, choices, advantages, reference_logp, clip=.2, kl_weight=.01):
    assert new_logp.ndim==2 and old_logp.shape==new_logp.shape
    row=torch.arange(len(choices),device=new_logp.device)
    ratio=(new_logp[row,choices]-old_logp[row,choices]).exp()
    surrogate=torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages)
    kl=(new_logp.exp()*(new_logp-reference_logp)).sum(1)
    # Caller supplies one row per real decision, fixed H, averages trajectories.
    return -surrogate.mean()+kl_weight*kl.mean(), dict(ratio=ratio.detach(),kl=kl.detach())

