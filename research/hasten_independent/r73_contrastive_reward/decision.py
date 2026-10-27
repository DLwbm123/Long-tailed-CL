"""One-step categorical GRPO/TARS for CLIP; not generative TARS reproduction."""
import math
import torch
import torch.nn.functional as F

GROUP_SIZE=8
CLIP=.2
BETA=.04
GAMMA=.75
EPS=1e-8
ARMS=('binary_grpo','hardnegative_grpo','binary_exact','hardnegative_exact')
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

@torch.no_grad()
def hardnegative_table(prototypes):
    assert prototypes.ndim==2 and len(prototypes)>1
    similarity=prototypes@prototypes.T;similarity.fill_diagonal_(-torch.inf)
    table=similarity.argmax(1);assert bool((table!=torch.arange(len(table),device=table.device)).all())
    return table

def ternary_exact_surrogate(logp,old_logp,labels,hard_labels,contrastive):
    old=old_logp.detach().exp().double();ratio=(logp.double()-old_logp.detach().double()).exp();clipped=ratio.clamp(1-CLIP,1+CLIP)
    positive=F.one_hot(labels,logp.shape[1]).bool();negative=F.one_hot(hard_labels,logp.shape[1]).bool();neutral=~(positive|negative)
    masks=[positive,negative,neutral];mass=[(old*m).sum(1) for m in masks];conditional=[old*m/q[:,None].clamp_min(1e-30) for m,q in zip(masks,mass)];rewards=[1.,-1. if contrastive else 0.,0.];answer=torch.zeros_like(mass[0])
    for kp in range(GROUP_SIZE+1):
        for kn in range(GROUP_SIZE-kp+1):
            kz=GROUP_SIZE-kp-kn;ks=[kp,kn,kz];fractions=[k/GROUP_SIZE for k in ks];mean=sum(f*r for f,r in zip(fractions,rewards));variance=sum(f*(r-mean)**2 for f,r in zip(fractions,rewards))
            if variance==0:continue
            probability=math.factorial(GROUP_SIZE)/(math.factorial(kp)*math.factorial(kn)*math.factorial(kz))
            for q,k in zip(mass,ks):probability=probability*q.pow(k)
            value=torch.zeros_like(answer)
            for f,reward,cond in zip(fractions,rewards,conditional):
                advantage=(reward-mean)/(math.sqrt(variance)+EPS);value+=f*(cond*torch.minimum(ratio*advantage,clipped*advantage)).sum(1)
            answer+=probability*value
    return answer.to(logp.dtype)

def objective(logits,old_logits,ref_logits,labels,weights,arm,generator=None,hard_labels=None):
    assert arm in ARMS and hard_labels is not None and bool((hard_labels!=labels).all())
    logp=logits.log_softmax(1);old=old_logits.detach().log_softmax(1);ref=ref_logits.detach().log_softmax(1);kl=(logp.exp()*(logp-ref)).sum(1);contrastive=arm.startswith('hardnegative_')
    stats=dict(groups=0,zero_advantage_groups=0,reward_sum=0,actions=0,clipped_actions=0,correct_actions=0,hardnegative_actions=0,neutral_wrong_actions=0,informative_all_wrong_groups=0)
    if arm.endswith('exact'):surrogate=ternary_exact_surrogate(logp,old,labels,hard_labels,contrastive)
    else:
        actions=torch.multinomial(old_logits.detach().softmax(1),GROUP_SIZE,replacement=True,generator=generator);correct=actions==labels[:,None];hard=actions==hard_labels[:,None];rewards=correct.to(logits.dtype)-(hard.to(logits.dtype) if contrastive else 0);adv=(rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True,correction=0)+EPS)
        ratio=(logp.gather(1,actions)-old.gather(1,actions)).exp();surrogate=torch.minimum(ratio*adv,ratio.clamp(1-CLIP,1+CLIP)*adv).mean(1);nonzero=adv.abs().sum(1)>0
        stats.update(groups=len(labels),zero_advantage_groups=int((~nonzero).sum()),reward_sum=float(rewards.sum()),actions=rewards.numel(),clipped_actions=int(((ratio<1-CLIP)|(ratio>1+CLIP)).sum()),correct_actions=int(correct.sum()),hardnegative_actions=int(hard.sum()),neutral_wrong_actions=int((~(correct|hard)).sum()),informative_all_wrong_groups=int((~correct.any(1)&nonzero).sum()))
    scale=weights[labels].to(logits.dtype).detach()
    with torch.no_grad():stats.update(KL_sum=float(kl.sum()),mean_weight_sum=float(weights[labels].sum()))
    return -scale*surrogate+BETA*kl,stats
