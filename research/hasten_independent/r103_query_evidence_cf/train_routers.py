"""Shared query schedule, actual old policy fixed for four inner updates."""
import time
import torch
from common import write,save,guard
from router import Router,ARMS,Q
from objectives import objective
from grpo_reference import old_tail_penalty,reference_kl

def train(c,folder,OOF,IN,labels,birth,stage,seed,counts):
    C=len(birth);protected_class=(birth<stage)&(torch.tensor(c['train_counts'][:C],device='cuda')<250);OT=int(protected_class.sum());ix=[torch.where(labels==cl)[0] for cl in range(C)]
    gen=torch.Generator(device='cuda').manual_seed(seed);plan=[]
    for _ in range(128):
        classes=torch.randint(C,(128,),generator=gen,device='cuda');query=torch.stack([ix[int(cl)][torch.randint(len(ix[int(cl)]),(1,),generator=gen,device='cuda')].squeeze() for cl in classes]);plan.append(query)
    save(folder/'QUERY_PLAN.private.pt',torch.stack(plan).cpu());models={};diagnostics=[]
    torch.manual_seed(seed);init=Router().cuda().state_dict()
    for arm in ARMS:
        model=Router().cuda();model.load_state_dict(init);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        table=IN if arm=='GRPO_IN' else OOF;x=table['features'];correct=table['correct'];pred=table['scores'].argmax(-1);assert not x.requires_grad
        with torch.no_grad():assert torch.equal(model(x[:2]).argmax(-1),torch.zeros(min(2,len(x)),device='cuda',dtype=torch.long))
        save(folder/f'{arm}_step0.private.pt',{'model':model.state_dict(),'optimizer':opt.state_dict()});ag=torch.Generator(device='cuda').manual_seed(seed+17)
        start=time.time();cpu=time.process_time();rec=[]
        for outer,qi in enumerate(plan):
            guard(c);xx=x[qi];cc=correct[qi];protected=protected_class[labels[qi]]
            with torch.no_grad():old=model(xx).double().softmax(-1).detach().clone();r=cc.double()-cc[:,0:1].double()
            actions=torch.multinomial(old,8,replacement=True,generator=ag) if arm in ['GRPO_CF','GRPO_IN'] else None
            if actions is not None:
                counts['sampled_query_groups']+=128;counts['sampled_actions']+=1024;rr=r.gather(1,actions);std=rr.std(-1,unbiased=False);zero=float((std<=1e-8).double().mean())
            else:std=torch.zeros(128,device='cuda');zero=None
            old_snapshot=old.clone()
            for inner in range(4):
                opt.zero_grad(set_to_none=True);logits=model(xx);loss=objective(arm,logits,old,cc,protected,C,OT,actions);assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                with torch.no_grad():
                    ratio=(logits.double().log_softmax(-1)-old.log()).exp();rho=ratio.gather(1,actions) if actions is not None else ratio
                    h=old_tail_penalty(logits,cc,protected,eligible_class_count=C,protected_class_count=OT);kl=reference_kl(logits).mean();p=logits.double().softmax(-1)
                opt.step();counts['router_optimizer']+=1;step=outer*4+inner+1
                rec.append({'outer':outer,'inner':inner,'loss':float(loss.detach()),'ratio_clip_fraction':float(((rho<.8)|(rho>1.2)).double().mean()),'max_ratio_deviation':float((rho-1).abs().max()),'gradient_norm':float(norm),'gradient_clipped':float(norm>1),'sampled_zero_group_fraction':zero,'sampled_reward_std':float(std.mean()) if actions is not None else None,'analytic_old_tail_harm_estimate':float(h),'KL':float(kl),'entropy':float(-(p*p.log()).sum(-1).mean())})
                if step in [256,512]:save(folder/f'{arm}_step{step}.private.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step})
            assert torch.equal(old,old_snapshot)
            write(folder/'ROUTER_ACTIVE.json',{'arm':arm,'outer':outer,'updates':step,'counts':counts})
        pack=torch.load(folder/f'{arm}_step512.private.pt',weights_only=False);other=Router().cuda();other.load_state_dict(pack['model']);oo=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=.0001);oo.load_state_dict(pack['optimizer'])
        with torch.no_grad():assert torch.equal(model(x[:2]),other(x[:2]))
        assert all(int(v['step'])==512 for v in oo.state.values());del other,oo
        write(folder/f'{arm}_TRAIN_DIAGNOSTICS.json',{'arm':arm,'updates':512,'old_fixed_four_steps':True,'strict_restore':True,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'records':rec})
        model.eval().requires_grad_(False);models[arm]=model
    return models
