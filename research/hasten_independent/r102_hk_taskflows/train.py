"""Matched supervised cycles, then a frozen-geometry policy step; no F reads."""
import json,time,copy
from pathlib import Path
import torch
from common import write,save,guard
from model import Joint,ARMS,supervised,auxiliary,diagnostics
from data_contract import load

COUNTS={'supervised_optimizer':0,'policy_optimizer':0,'formal_optimizer':0,'supervised_query_rows':0,'support_transform_rows':0,'aux_query_rows':0,'sampled_query_score_rows':0,'joint_G8_groups':0,'joint_rollouts':0,'Gaussian_action_coordinates':0,'Gaussian_draw_coordinates':0,'image_encoder':0,'F_rows':0,'VAL_rows':0,'TEST_rows':0}

def optimizers(m):
    sup=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    pol=None if m.arm=='UNIFORM' else torch.optim.AdamW(m.policy_parameters(),lr=1e-4,weight_decay=.01)
    return sup,pol

def noise(d,step,device='cuda'):
    g=torch.Generator(device=device).manual_seed(74002+1000003*d['stage']+7919*step)
    return [torch.randn(8,len(ix),device=device,generator=g) for ix in d['ix']]

def cycle(m,sup,pol,d,step):
    m.zero_grad(set_to_none=True);l,st=supervised(m,d);l.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.0));assert torch.isfinite(l)
    sup.step();m.zero_grad(set_to_none=True);auxstats={};pnorm=0.
    if pol is not None:
        l2,auxstats=auxiliary(m,d,noise(d,step) if m.arm=='PROTO_GRPO' else None);assert torch.isfinite(l2);l2.backward()
        assert all(p.grad is None for p in m.representation_parameters()),'GEOMETRY_UPDATED_INSIDE_ROLLOUT'
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.policy_parameters())
        pnorm=float(torch.nn.utils.clip_grad_norm_(m.policy_parameters(),1.0));pol.step();m.zero_grad(set_to_none=True)
    return {'supervised':st,'auxiliary':auxstats,'supervised_grad_norm_before_clip':norm,'policy_grad_norm_before_clip':pnorm,'supervised_loss':float(l.detach())}

def run(c,meta):
    r=Path(c['output']);models={};opts={};snapshots=[];diags=[]
    for arm in ARMS:
        torch.manual_seed(74002);torch.cuda.manual_seed_all(74002);m=Joint(arm).cuda();models[arm]=m;opts[arm]=optimizers(m)
    for s in range(len(c['stages'])):
        d=load(c,s,meta)
        for arm in ARMS:
            m=models[arm];sup,pol=opts[arm]
            save(r/f'STAGE_START_s{s}_{arm}.private.pt',{'stage':s,'arm':arm,'model':m.state_dict(),'supervised_optimizer':sup.state_dict(),'policy_optimizer':pol.state_dict() if pol else None,'stage_carry':s>0,'reference_EMA_from_previous_saved_endpoint':s>0})
            diags.append({'stage':s,'arm':arm,'cycle':0,**diagnostics(m,d)})
            for step in range(c['cycles']):
                guard(c);stats=cycle(m,sup,pol,d,step)
                COUNTS['supervised_optimizer']+=1;COUNTS['formal_optimizer']+=1
                COUNTS['support_transform_rows']+=len(d['S']);COUNTS['supervised_query_rows']+=len(d['B'])
                if pol is not None:
                    COUNTS['policy_optimizer']+=1;COUNTS['formal_optimizer']+=1;COUNTS['support_transform_rows']+=len(d['S']);COUNTS['aux_query_rows']+=len(d['B'])
                if arm=='PROTO_GRPO':
                    COUNTS['joint_G8_groups']+=1;COUNTS['joint_rollouts']+=8;COUNTS['Gaussian_action_coordinates']+=8*sum(len(ix) for ix in d['ix'] if len(ix)>1);COUNTS['Gaussian_draw_coordinates']+=8*len(d['S']);COUNTS['sampled_query_score_rows']+=8*len(d['B'])
                with (r/'TRAIN_DIAGNOSTICS.private.jsonl').open('a') as f:f.write(json.dumps({'stage':s,'arm':arm,'cycle':step+1,**stats})+'\n')
                if (step+1)%16==0:
                    write(r/'ACTIVE_COUNTS.private.json',{'stage':s,'arm':arm,'cycle':step+1,'counts':COUNTS})
                if step+1 in [256,c['cycles']]:
                    diags.append({'stage':s,'arm':arm,'cycle':step+1,**diagnostics(m,d)})
                    if step+1==256:
                        save(r/f's{s}_{arm}_middle.private.pt',{'stage':s,'arm':arm,'cycle':step+1,'model':m.state_dict(),'supervised_optimizer':sup.state_dict(),'policy_optimizer':pol.state_dict() if pol else None})
            # Save before reference EMA, then carry updated reference to next stage.
            name=f's{s}_{arm}.private.pt'
            save(r/name,{'stage':s,'arm':arm,'cycle':c['cycles'],'model':m.state_dict(),'supervised_optimizer':sup.state_dict(),'policy_optimizer':pol.state_dict() if pol else None,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'support_ids':d['Sids'],'counts':dict(COUNTS),'config':{k:c[k] for k in ['cycles','source_commit','seed']}})
            snapshots.append(name);m.update_reference();write(r/'STATUS.json',{'phase':'TRAIN','stage':s,'arm_complete':arm,'counts':COUNTS})
        del d
    assert COUNTS['supervised_optimizer']==10240 and COUNTS['policy_optimizer']==7680 and COUNTS['formal_optimizer']==17920 and len(snapshots)==20
    write(r/'TRAIN_DIAGNOSTICS.json',diags)
    write(r/'SEAL.json',{'final_endpoints':20,'middle_endpoints':20,'supervised_steps':10240,'policy_steps':7680,'total_optimizer_steps':17920,'F_VAL_TEST_before_seal':0,'cycles_per_stage_arm':512,'deploy':'deterministic logits, fixed S bank/teacher anchor, shared current adapter for support and query','no_endpoint_selection':True,'reference_updated_only_after_saved_stage':True})
    write(r/'TRAIN_COMPLETE.json',{'counts':COUNTS,'endpoints':len(snapshots)})
