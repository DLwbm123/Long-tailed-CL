"""One candidate, fixed teacher: old and additional-competition gradient residuals."""
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from check_firststage_query_audit import comparison, distribution, unit_checks as prior_unit_checks
from check_reserved_competition_memory import (DT, initial_model, clone_model, frame, scores,
    response, risk_gradient, wasserstein_1d, sliced, lowrank_fixture, initialization, project_ball)


def residual_terms(z,p,c,model,q,old_slots,expanded_slots,target):
    old=risk_gradient(z,c,model,q,old_slots,p,True)[1]
    expanded=risk_gradient(z,c,model,q,expanded_slots,p,True)[1]
    ro,rd=target
    eo=(old-ro).square().sum();ed=(expanded-old-rd).square().sum()
    scale=(ro.square().sum()+rd.square().sum()).clamp_min(1e-12)
    return eo,ed,scale


def construct(current,seed,teacher,q,old_slots,expanded_slots,journal):
    start=time.monotonic();model=clone_model(teacher,True);targets=[]
    for c,x in current.items():
        a=risk_gradient(x,c,model,q,old_slots)[1].detach()
        b=risk_gradient(x,c,model,q,expanded_slots)[1].detach()
        targets.append((a,b-a))
    n,w=initialization(current,seed);n.requires_grad_();w.requires_grad_()
    opt=torch.optim.Adam([n,w],lr=.02);components=[]
    dominance=[dict(class_index=c,old=0,innovation=0,tied=0) for c in current]
    for step in range(48):
        objective=n.sum()*0
        for i,c in enumerate(current):
            eo,ed,scale=residual_terms(F.normalize(n[i],dim=1),w[i].softmax(0),c,model,q,old_slots,expanded_slots,targets[i])
            value=torch.maximum(eo,ed)/scale;objective=objective+value/len(current)
            a,b=float(eo.detach()),float(ed.detach())
            dominance[i]['old' if a>b else 'innovation' if b>a else 'tied']+=1
            if step in (0,47):components.append(dict(class_index=c,preupdate_step=step,old_error_squared=a,innovation_error_squared=b,scale=float(scale),objective=float(value.detach())))
        assert bool(torch.isfinite(objective))
        opt.zero_grad();n.grad,w.grad=torch.autograd.grad(objective,(n,w));opt.step()
        with torch.no_grad():
            n.copy_(project_ball(n));w.clamp_(-8,8)
            assert bool(torch.isfinite(n).all()) and bool(torch.isfinite(w).all()) and float(n.norm(dim=2).min())>1e-12
        if step in (0,47):journal(dict(phase='candidate_construction',completed_updates=step+1))
    memory={c:(F.normalize(n[i].detach(),dim=1),w[i].detach().softmax(0)) for i,c in enumerate(current)}
    return memory,dict(seconds=time.monotonic()-start,updates=48,method='candidate',objective_components=components,dominance=dominance,
        target_gradient_norms=[dict(class_index=c,old_norm_squared=float(a.square().sum()),innovation_norm_squared=float(b.square().sum()),scale=float((a.square().sum()+b.square().sum()).clamp_min(1e-12))) for c,(a,b) in zip(current,targets)],
        min_effective_nodes=min(float(1/p.square().sum()) for _,p in memory.values()))


def unit_checks():
    result=prior_unit_checks();gen=torch.Generator().manual_seed(349)
    teacher=initial_model(8,349);teacher=(teacher[0],torch.randn(8,8,generator=gen,dtype=DT)*.03)
    model=clone_model(teacher,True);q=frame(8,4,349)
    x=F.normalize(torch.randn(6,8,generator=gen,dtype=DT),dim=1)
    a=risk_gradient(x,0,model,q,[0,1])[1].detach();b=risk_gradient(x,0,model,q,[0,1,2])[1].detach();target=(a,b-a)
    n=torch.randn(3,8,generator=gen,dtype=DT).requires_grad_();w=torch.randn(3,generator=gen,dtype=DT).requires_grad_()
    dn=torch.randn(3,8,generator=gen,dtype=DT);dw=torch.randn(3,generator=gen,dtype=DT)
    def fun(nn,ww):
        eo,ed,s=residual_terms(F.normalize(nn,dim=1),ww.softmax(0),0,model,q,[0,1],[0,1,2],target)
        return torch.maximum(eo,ed)/s,abs(float((eo-ed).detach()))
    v,gap=fun(n,w);assert gap>1e-6
    gn,gw=torch.autograd.grad(v,(n,w));eps=1e-6
    fd=(fun(n+eps*dn,w+eps*dw)[0]-fun(n-eps*dn,w-eps*dw)[0])/(2*eps)
    analytic=(gn*dn).sum()+(gw*dw).sum();error=abs(float(fd-analytic));assert error<1e-5
    result.update(max_objective_node_weight_FD_error=error,branch_gap=gap,synthetic_updates=0)
    return result


def run_scale(cfg,lock,journal):
    start=time.monotonic();journal(dict(phase='load_saved_teacher_and_moment'))
    state=torch.load(Path(os.environ['Q349_PRIOR'])/(cfg['id']+'.state.private.pt'),map_location='cpu',weights_only=True)
    assert state['case']==cfg
    teacher=state['teacher'];q=state['frame'];d=cfg['dimension'];k=cfg['classes']
    old_n=cfg['task_sizes'][0];expanded_n=sum(cfg['task_sizes'][:2]);moment=state['memories']['moment']
    assert q.shape==(d,k) and all(a.shape==(d,8) and a.dtype==DT for a in teacher)
    assert set(moment)==set(range(old_n)) and all(z.shape==(16,d) and p.shape==(16,) for z,p in moment.values())
    x,y,counts=lowrank_fixture(cfg);current={c:x[y==c] for c in range(old_n)}
    load_seconds=time.monotonic()-start
    journal(dict(phase='candidate_construction',completed_updates=0))
    candidate,cost=construct(current,cfg['seed'],teacher,q,list(range(old_n)),list(range(expanded_n)),journal)
    memories={'moment':moment,'candidate':candidate};costs=[dict(mode='candidate',**cost)]
    frozen_teacher=clone_model(teacher);frozen_memory={m:{c:(z.clone(),p.clone()) for c,(z,p) in mem.items()} for m,mem in memories.items()}
    torch.save(dict(candidate=candidate,case=cfg),Path(os.environ['Q349_ROOT'])/(cfg['id']+'.candidate.private.pt'))
    diagnostic_start=time.monotonic();model=clone_model(teacher,True)
    gradients=[];responses=[];losses=[];aggregates=[];gg={};rr={};ll={}
    for c in range(old_n):
        journal(dict(phase='fixed_teacher_queries',class_index=c,classes=old_n))
        representations={'full_support':(current[c],None),**{m:mem[c] for m,mem in memories.items()}}
        for mode,(z,p) in representations.items():
            gg[c,mode]={};rr[c,mode]={};ll[c,mode]={}
            for query,n in [('old',old_n),('expanded',expanded_n)]:
                value,g=risk_gradient(z,c,model,q,list(range(n)),p)
                gg[c,mode][query]=g.detach();ll[c,mode][query]=float(value.detach())
                with torch.no_grad():
                    resp=response(z,c,model,q,list(range(n)));rr[c,mode][query]=resp
                    weights=torch.ones(len(z),dtype=DT)/len(z) if p is None else p
                    correct=scores(z,model,q)[:,:n].argmax(1)==c
                    ref=rr[c,'full_support'][query]
                    responses.append(dict(class_index=c,mode=mode,query=query,count=len(z),
                        teacher_correct_mass=float((weights*correct).sum()),
                        effective_nodes=float(1/weights.square().sum()),
                        seen=distribution(resp[:,0],weights),reserved=distribution(resp[:,1],weights),
                        seen_W1=float(wasserstein_1d(ref[:,0],resp[:,0],weights)),
                        reserved_W1=float(wasserstein_1d(ref[:,1],resp[:,1],weights)),
                        sliced_W1=float(sliced(ref,resp,weights))))
            gg[c,mode]['innovation']=gg[c,mode]['expanded']-gg[c,mode]['old']
            ll[c,mode]['innovation']=ll[c,mode]['expanded']-ll[c,mode]['old']
        for query in ('old','expanded','innovation'):
            ref=gg[c,'full_support'][query]
            for mode in representations:
                g=gg[c,mode][query];middle=g.numel()//2
                gradients.append(dict(class_index=c,mode=mode,query=query,**comparison(g,ref),
                    U=comparison(g[:middle],ref[:middle]),V=comparison(g[middle:],ref[middle:])))
                value=ll[c,mode][query];reference=ll[c,'full_support'][query]
                losses.append(dict(class_index=c,mode=mode,query=query,CE=value,reference_CE=reference,absolute_error=abs(value-reference)))
    for query in ('old','expanded','innovation'):
        reference=torch.stack([gg[c,'full_support'][query] for c in range(old_n)])
        for mode in ('moment','candidate'):
            stack=torch.stack([gg[c,mode][query] for c in range(old_n)]);err=stack-reference
            mean_error=float(err.norm(dim=1).mean());aggregate_error=float(err.mean(0).norm())
            aggregates.append(dict(mode=mode,query=query,classes=old_n,**comparison(stack.mean(0),reference.mean(0)),
                mean_class_absolute_error=mean_error,cancellation_ratio=aggregate_error/mean_error if mean_error>1e-12 else None,
                mean_class_relative_error=sum(comparison(g,r)['relative_error'] for g,r in zip(stack,reference))/old_n if bool((reference.norm(dim=1)>1e-12).all()) else None))
    assert all(torch.equal(a,b) for a,b in zip(teacher,frozen_teacher))
    assert all(torch.equal(a,b) for a,b in zip(model,teacher))
    assert all(torch.equal(a,b) for m,mem in memories.items() for c,zp in mem.items() for a,b in zip(zp,frozen_memory[m][c]))
    assert len(gradients)==old_n*9 and len(responses)==old_n*6 and len(losses)==old_n*9
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,old_classes=old_n,expanded_classes=expanded_n,
        gradients=gradients,responses=responses,losses=losses,aggregates=aggregates,training=dict(seconds=0,updates=0,reused=True),construction=costs,
        teacher_and_memories_unchanged=True,state_saved_private=True,task2_updates=0,total_adapter_updates=0,load_seconds=load_seconds,
        actual_memory_updates=48,adapter_parameters=2*d*8,memory_nodes_per_class=16,
        diagnostic_seconds=time.monotonic()-diagnostic_start,CPU_pipeline_seconds=time.monotonic()-start)



def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,val_accessed=False,test_accessed=False)
    assert not result['GPU_initialized']
    return result
