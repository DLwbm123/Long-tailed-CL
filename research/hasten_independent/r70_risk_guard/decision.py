"""One-step categorical GRPO/TARS for CLIP; not generative TARS reproduction."""
import math
import torch
import torch.nn.functional as F

GROUP_SIZE=8
CLIP=.2
BETA=.04
GAMMA=.75
EPS=1e-8
ARMS=('unprotected_tars_grpo','protected_tars_grpo','unprotected_tars_exact','protected_tars_exact')
_BASE_ARMS=('sft_continue','grpo','tars_grpo','tars_exact')

@torch.no_grad()
def tailness(counts,correct):
    counts=counts.double();correct=correct.double()
    assert counts.ndim==1 and bool((counts>0).all())
    assert bool(((correct>=0)&(correct<=counts)).all())
    raw=(counts.mean()/counts*(1-correct/counts)).pow(GAMMA)
    median=torch.quantile(raw,.5)
    weights=(raw/median.clamp_min(EPS)).clamp_min(1)
    assert torch.isfinite(weights).all()
    return weights

@torch.no_grad()
def rollout(old_logits,labels,generator):
    assert old_logits.ndim==2 and old_logits.shape[0]==len(labels)
    actions=torch.multinomial(old_logits.softmax(1),GROUP_SIZE,replacement=True,generator=generator)
    rewards=(actions==labels[:,None]).to(old_logits.dtype)
    adv=(rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True,correction=0)+EPS)
    assert torch.isfinite(adv).all()
    return actions,adv,rewards

def exact_group_surrogate(logp,old_logp,labels):
    """Exact expectation over all Bernoulli group counts and wrong-class identities."""
    old=old_logp.detach().exp().double();p=old.gather(1,labels[:,None]).squeeze(1)
    ratio=(logp.double()-old_logp.detach().double()).exp();clipped=ratio.clamp(1-CLIP,1+CLIP)
    correct=F.one_hot(labels,logp.shape[1]).bool();wrong=old.masked_fill(correct,0)/(1-p[:,None]).clamp_min(1e-30)
    result=torch.zeros_like(p)
    for k in range(1,GROUP_SIZE):
        f=k/GROUP_SIZE;sd=math.sqrt(f*(1-f))+EPS
        a_pos=(1-f)/sd;a_neg=-f/sd
        prob=math.comb(GROUP_SIZE,k)*p.pow(k)*(1-p).pow(GROUP_SIZE-k)
        pos=torch.minimum(ratio*a_pos,clipped*a_pos).gather(1,labels[:,None]).squeeze(1)
        neg=(wrong*torch.minimum(ratio*a_neg,clipped*a_neg)).sum(1)
        result+=prob*(f*pos+(1-f)*neg)
    return result.to(logp.dtype)

def _base_objective(logits,old_logits,ref_logits,labels,weights,arm,generator=None):
    assert arm in _BASE_ARMS and logits.shape==old_logits.shape==ref_logits.shape
    logp=logits.log_softmax(1);old=old_logits.detach().log_softmax(1);ref=ref_logits.detach().log_softmax(1)
    kl=(logp.exp()*(logp-ref)).sum(1)
    stats=dict(groups=0,zero_advantage_groups=0,reward_sum=0,actions=0,clipped_actions=0)
    if arm=='sft_continue':return F.cross_entropy(logits,labels,reduction='none'),stats
    if arm=='tars_exact':surrogate=exact_group_surrogate(logp,old,labels)
    else:
        actions,adv,rewards=rollout(old_logits.detach(),labels,generator)
        ratio=(logp.gather(1,actions)-old.gather(1,actions)).exp()
        surrogate=torch.minimum(ratio*adv,ratio.clamp(1-CLIP,1+CLIP)*adv).mean(1)
        stats=dict(groups=len(labels),zero_advantage_groups=int((adv.abs().sum(1)==0).sum()),reward_sum=int(rewards.sum()),actions=rewards.numel(),clipped_actions=int(((ratio<1-CLIP)|(ratio>1+CLIP)).sum()))
    scale=torch.ones_like(surrogate) if arm=='grpo' else weights[labels].to(logits.dtype).detach()
    return -scale*surrogate+BETA*kl,stats

def policy_logits(raw,counts,arm):
    assert arm in ARMS
    return raw+counts.to(raw.dtype).log()[None,:]

def objective(logits,old_logits,ref_logits,labels,weights,arm,generator=None):
    assert arm in ARMS
    loss,stats=_base_objective(logits,old_logits,ref_logits,labels,weights,'tars_exact' if arm.endswith('exact') else 'tars_grpo',generator)
    with torch.no_grad():
        lp=logits.log_softmax(1);rp=ref_logits.detach().log_softmax(1)
        stats.update(KL_sum=float((lp.exp()*(lp-rp)).sum()),mean_weight_sum=float(weights[labels].sum()))
    return loss,stats

@torch.no_grad()
def project_update(delta,guide,enabled):
    assert len(delta)==len(guide) and delta
    dot=sum((d.double()*g.double()).sum() for d,g in zip(delta,guide))
    norm=sum(g.double().square().sum() for g in guide)
    fired=bool(enabled and dot>0 and norm>0)
    result=[d-(dot/norm).to(d.dtype)*g for d,g in zip(delta,guide)] if fired else delta
    after=sum((d.double()*g.double()).sum() for d,g in zip(result,guide))
    scale=sum(d.double().square().sum() for d in result).sqrt()*norm.sqrt()
    assert all(torch.isfinite(d).all() for d in result)
    if enabled:assert float(after)<=1e-6*float(scale)+1e-12
    return result,dict(projected=fired,proposal_guide_dot=float(dot),result_guide_dot=float(after),guide_norm_squared=float(norm),projection_tolerance=1e-6*float(scale)+1e-12)
