"""Unchanged R117 PG/KL kernel; only pure helper imports are localized."""
import torch
from .core import normalize, advantages

def rl_corrected(self,features,labels,opt,pg_coeff,kl_coeff,seed):
    G,n=8,len(labels);f=features.repeat_interleave(G,0)
    generator=torch.Generator(device='cuda').manual_seed(seed)
    tokens,mask,sampled,text,done=self.engine.generate(f,generator,True)
    target=[normalize(self.names[int(c)]) for c in labels for _ in range(G)]
    reward=torch.tensor([int(t==y and bool(d)) for t,y,d in zip(text,target,done)],device='cuda',dtype=torch.float32).reshape(n,G)
    adv=advantages(reward).flatten().detach()
    with torch.no_grad(),self.engine.reference():
        ref=torch.cat([self.engine.token_logps(f[i:i+2],tokens[i:i+2],mask[i:i+2]) for i in range(0,n*G,2)])
    with torch.no_grad():
        old=torch.cat([self.engine.token_logps(f[i:i+2],tokens[i:i+2],mask[i:i+2]) for i in range(0,n*G,2)])
    pgw=pg_coeff[labels.cuda()].repeat_interleave(G);klw=kl_coeff[labels.cuda()].repeat_interleave(G)
    opt.zero_grad(set_to_none=True);loss_total,kl_total,weighted_kl,delta_max=0.,0.,0.,0.
    for lo in range(0,n*G,2):
        lp=self.engine.token_logps(f[lo:lo+2],tokens[lo:lo+2],mask[lo:lo+2]);m=mask[lo:lo+2]
        delta_max=max(delta_max,float(((lp.detach()-old[lo:lo+2]).abs()*m).max()))
        ratio=(lp-old[lo:lo+2]).exp();a=adv[lo:lo+2,None]
        pg=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)
        d=ref[lo:lo+2]-lp;assert float(d.detach().abs().max())<80
        kl=torch.expm1(d)-d;length=m.sum(-1)
        pg=(pg*m).sum(-1)/length;kl=(kl*m).sum(-1)/length
        loss=(-pgw[lo:lo+2]*pg+.04*klw[lo:lo+2]*kl).sum()/(n*G)
        assert torch.isfinite(loss);loss.backward()
        loss_total+=float(loss.detach());kl_total+=float(kl.detach().sum())/(n*G)
        weighted_kl+=float((klw[lo:lo+2]*kl.detach()).sum())/(n*G)
    assert delta_max==0,'UNCHANGED_BEHAVIOR_RATIO_NOT_ONE'
    norm=self.step(opt);self.counts['generation_responses']+=n*G
    return {'loss':loss_total,'KL':kl_total,'importance_weighted_KL':weighted_kl,'gradient_norm':norm,
            'reward_mean':float(reward.mean()),'groups':n,'constant_groups':int((reward.std(-1,unbiased=False)==0).sum()),
            'correct_by_class':torch.bincount(labels,weights=reward.sum(-1).cpu(),minlength=23).tolist(),
            'responses_by_class':(torch.bincount(labels,minlength=23)*G).tolist(),'tokens':int(mask.sum()),
            'truncated_responses':int((~done).sum()),'unchanged_policy_logp_max_difference':delta_max,
            'cached_sampling_vs_behavior_forward_max_difference':float(((sampled-old).abs()*mask).max()),
            'mean_surrogate_coefficient':float(pgw.mean()),'mean_KL_coefficient':float(klw.mean())}
