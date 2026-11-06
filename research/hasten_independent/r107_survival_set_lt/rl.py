"""Teacher-forced masked sequence likelihood, GSPO and matched action-level PG."""
import torch

def batch_trace(traces):
    device=traces[0][0]['x'].device
    states=[s for tr in traces for s in tr];sizes=torch.tensor([len(s['x']) for s in states],device=device);offset=sizes.cumsum(0)-sizes;index=torch.repeat_interleave(torch.arange(len(states),device=device),sizes);chosen=offset+torch.tensor([s['choice'] for s in states],device=device);traj=torch.repeat_interleave(torch.arange(len(traces),device=device),torch.tensor([len(t) for t in traces],device=device))
    assert all(len(t)>0 for t in traces);return {'x':torch.cat([s['x'] for s in states]),'old':torch.cat([s['old_logp'] for s in states]),'index':index,'chosen':chosen,'traj':traj,'length':torch.tensor([len(t) for t in traces],device=device),'ns':len(states)}

def loss(actor,b,returns,kind):
    logits=actor(b['x']).double();mx=torch.full((b['ns'],),-torch.inf,device=logits.device,dtype=torch.double);mx.scatter_reduce_(0,b['index'],logits,reduce='amax',include_self=True);norm=torch.zeros_like(mx).scatter_add(0,b['index'],(logits-mx[b['index']]).exp());lp=logits-mx[b['index']]-norm[b['index']].log()
    delta=lp[b['chosen']]-b['old'][b['chosen']];A=((returns-returns.mean())/(returns.std(unbiased=False)+1e-8)).detach();A=torch.where(returns.std(unbiased=False)<=1e-8,torch.zeros_like(A),A)
    if kind=='GSPO':
        ratio=(torch.zeros_like(returns).scatter_add(0,b['traj'],delta)/b['length']).exp();sur=torch.minimum(ratio*A,ratio.clamp(.8,1.2)*A).mean()
    else:
        ratio=delta.exp();adv=A[b['traj']];v=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv);sur=(torch.zeros_like(returns).scatter_add(0,b['traj'],v)/b['length']).mean()
    kl=torch.zeros(b['ns'],device=logits.device,dtype=torch.double).scatter_add(0,b['index'],b['old'].exp()*(b['old']-lp));trust=(torch.zeros_like(returns).scatter_add(0,b['traj'],kl)/b['length']).mean();return -sur+.01*trust,{'surrogate':float(sur.detach()),'trust_KL':float(trust.detach()),'ratio_clip_fraction':float(((ratio<.8)|(ratio>1.2)).double().mean()),'zero_advantage_group':bool(returns.std(unbiased=False)<=1e-8),'max_ratio_deviation':float((ratio-1).abs().max().detach())}

def update(actor,opt,traces,returns,kind,counter):
    b=batch_trace(traces);returns=torch.tensor(returns,device=b['x'].device,dtype=torch.double);out=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True);value,stats=loss(actor,b,returns,kind);assert torch.isfinite(value);value.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);assert torch.isfinite(norm)
        if step==0:assert stats['max_ratio_deviation']<2e-6,'OLD_POLICY_NOT_ACTUAL_BEHAVIOR'
        opt.step();counter['policy_optimizer']=counter.get('policy_optimizer',0)+1;out.append(dict(stats,gradient_norm=float(norm),loss=float(value.detach())))
    return out
