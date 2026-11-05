"""Eight matched policies on one frozen FULL feedback table and natural query plan."""
import time,copy
import torch
from common import write,save,guard
from router import Router,ARMS
from objectives import objective
from grpo_reference import reference_kl

def train(c,table,labels,weights,counts):
    from pathlib import Path
    r=Path(c['output']);g=torch.Generator(device='cuda').manual_seed(c['seed']);plan=torch.randint(len(labels),(128,128),device='cuda',generator=g);save(r/'QUERY_PLAN.private.pt',plan.cpu())
    sampled_class=torch.bincount(labels[plan.flatten()],minlength=23);write(r/'QUERY_EXPOSURE.json',{'sampling':'uniform TRAIN row, natural class frequency','query_draws_per_arm':16384,'classes':sampled_class.tolist(),'shared_plan_all_arms':True})
    torch.manual_seed(c['seed']);init=copy.deepcopy(Router().cuda().state_dict());records=[]
    for arm in ARMS:
        m=Router().cuda();m.load_state_dict(init);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);trace=[];log=[];cpu=time.process_time();wall=time.time()
        save(r/f'{arm}_step0.private.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),'step':0});ag=torch.Generator(device='cuda').manual_seed(c['seed']+17)
        for outer,ix in enumerate(plan):
            guard(c);x=table['features'][ix];correct=table['correct'][ix];ww=weights[labels[ix]];sampled=arm.removeprefix('TARS_')=='GRPO'
            with torch.no_grad():old=m(x).double().softmax(-1).detach().clone();reward=correct.double()-correct[:,0:1].double();actions=torch.multinomial(old,8,replacement=True,generator=ag) if sampled else None
            structural=(reward.max(-1).values==reward.min(-1).values);zero=None
            if sampled:
                std=reward.gather(1,actions).std(-1,unbiased=False);zero=float((std<=1e-8).double().mean());counts['sampled_groups']+=128;counts['sampled_actions']+=1024
            trace.append({'outer':outer,'old_probabilities':old.cpu(),'actions':actions.cpu() if sampled else None});snapshot=old.clone()
            for inner in range(4):
                opt.zero_grad(set_to_none=True);logits=m(x);loss=objective(arm,logits,old,correct,ww,actions);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm)
                with torch.no_grad():
                    ratio=logits.double().softmax(-1)/old;rho=ratio.gather(1,actions) if sampled else ratio;kl=float(reference_kl(logits).mean());p=logits.double().softmax(-1)
                opt.step();counts['router_optimizer']+=1;step=outer*4+inner+1
                log.append({'outer':outer,'inner':inner,'loss':float(loss.detach()),'gradient_norm':float(norm),'gradient_clipped':float(norm>1),'ratio_clip_fraction':float(((rho<.8)|(rho>1.2)).double().mean()),'KL':kl,'entropy':float(-(p*p.log()).sum(-1).mean()),'structural_zero_group_fraction':float(structural.double().mean()),'sampled_zero_group_fraction':zero,'mean_query_weight':float(ww.mean()) if arm.startswith('TARS_') else 1.})
                if step in [256,512]:save(r/f'{arm}_step{step}.private.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step})
            assert torch.equal(old,snapshot);write(r/'ACTIVE_COUNTS.private.json',{'phase':'ROUTERS','arm':arm,'outer':outer,'step':step,'counts':counts})
        pack=torch.load(r/f'{arm}_step512.private.pt',weights_only=False);other=Router().cuda();other.load_state_dict(pack['model']);oo=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=.0001);oo.load_state_dict(pack['optimizer'])
        with torch.no_grad():assert torch.equal(m(table['features'][:2]),other(table['features'][:2]))
        assert all(int(v['step'])==512 for v in oo.state.values());save(r/f'{arm}_POLICY_TRACE.private.pt',trace)
        write(r/f'{arm}_TRAIN_DIAGNOSTICS.json',{'arm':arm,'updates':512,'strict_restore':True,'true_old_fixed_four_steps':True,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'records':log})
        del m,other,oo,opt,trace
    assert counts['router_optimizer']==4096 and counts['sampled_groups']==32768 and counts['sampled_actions']==262144
