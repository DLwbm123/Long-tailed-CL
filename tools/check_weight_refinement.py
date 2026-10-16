"""Fixed moment nodes; one weight-only endpoint with per-class acceptance and fallback."""
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from check_firststage_query_audit import comparison, distribution, unit_checks as prior_unit_checks
from check_reserved_competition_memory import (DT, initial_model, clone_model, frame, scores,
    response, risk_gradient, wasserstein_1d, sliced, lowrank_fixture, initialization, project_ball)


def weight_logits(p):
    v=p.log();v=v-(v.min()+v.max())/2
    assert bool(torch.isfinite(v).all()) and float(v.abs().max())<=8+1e-12
    assert torch.allclose(v.softmax(0),p,atol=1e-14,rtol=1e-12)
    return v


def errors(z,p,c,model,q,old_slots,expanded_slots,target):
    a=risk_gradient(z,c,model,q,old_slots,p,True)[1]
    b=risk_gradient(z,c,model,q,expanded_slots,p,True)[1]
    ro,re=target
    return torch.stack([(a-ro).square().sum(),((b-a)-(re-ro)).square().sum(),(b-re).square().sum()])


def endpoint_gate(base_g,can_g,truth_r,base_r,can_r):
    checks={};strict=False
    for query in ('old','innovation','expanded'):
        delta=can_g[query]['absolute_error']-base_g[query]['absolute_error']
        checks['gradient_'+query]=delta<=1e-12;strict=strict or delta< -1e-12
    for query in ('old','expanded'):
        ref,base,can=truth_r[query],base_r[query],can_r[query]
        for key in ('seen_W1','reserved_W1','sliced_W1'):
            checks[query+'_'+key]=can[key]<=base[key]+1e-12
        checks[query+'_coverage_error']=abs(can['teacher_correct_mass']-ref['teacher_correct_mass'])<=abs(base['teacher_correct_mass']-ref['teacher_correct_mass'])+1e-12
        checks[query+'_near_zero_error']=abs(can['seen']['near_zero_mass']-ref['seen']['near_zero_mass'])<=abs(base['seen']['near_zero_mass']-ref['seen']['near_zero_mass'])+1e-12
    assert len(checks)==13
    return dict(accepted=all(checks.values()) and strict,checks=checks,strict_gradient_improvement=strict,
                failed_conditions=[k for k,v in checks.items() if not v]+([] if strict else ['no_strict_gradient_improvement']))


def construct(current,moment,teacher,q,old_slots,expanded_slots,journal):
    start=time.monotonic();model=clone_model(teacher,True);targets=[]
    for c,x in current.items():
        a=risk_gradient(x,c,model,q,old_slots)[1].detach()
        b=risk_gradient(x,c,model,q,expanded_slots)[1].detach();targets.append((a,b))
    w=torch.stack([weight_logits(moment[c][1]) for c in current]).requires_grad_()
    opt=torch.optim.Adam([w],lr=.02);components=[];baselines=[]
    for step in range(48):
        objective=w.sum()*0
        for i,c in enumerate(current):
            e=errors(moment[c][0],w[i].softmax(0),c,model,q,old_slots,expanded_slots,targets[i])
            if step==0:baselines.append(e.detach().clone())
            value=(e/baselines[i].clamp_min(1e-12)).mean();objective=objective+value/len(current)
            if step in (0,47):components.append(dict(class_index=c,preupdate_step=step,errors_squared=e.detach().tolist(),baseline_errors_squared=baselines[i].tolist(),objective=float(value.detach())))
        assert bool(torch.isfinite(objective))
        opt.zero_grad();w.grad=torch.autograd.grad(objective,w)[0];opt.step()
        with torch.no_grad():w.clamp_(-8,8)
        assert bool(torch.isfinite(w).all())
        if step in (0,47):journal(dict(phase='weight_refinement',completed_updates=step+1))
    candidate={c:(moment[c][0],w[i].detach().softmax(0)) for i,c in enumerate(current)}
    return candidate,dict(seconds=time.monotonic()-start,updates=48,method='candidate',objective_components=components,
        query_order=['old','innovation','expanded'],baseline_errors_squared=[dict(class_index=c,values=b.tolist(),floor_hits=int((b<1e-12).sum())) for c,b in zip(current,baselines)],
        nodes_fixed=True,optimized_weights_per_class=16,min_effective_nodes=min(float(1/p.square().sum()) for _,p in candidate.values()))


def unit_checks():
    import copy
    result=prior_unit_checks();gen=torch.Generator().manual_seed(350)
    p=torch.tensor([.1,.2,.3,.4],dtype=DT);assert torch.allclose(weight_logits(p).softmax(0),p,atol=1e-14)
    teacher=initial_model(8,350);teacher=(teacher[0],torch.randn(8,8,generator=gen,dtype=DT)*.03)
    model=clone_model(teacher,True);q=frame(8,4,350)
    x=F.normalize(torch.randn(6,8,generator=gen,dtype=DT),dim=1)
    target=tuple(risk_gradient(x,0,model,q,slots)[1].detach() for slots in ([0,1],[0,1,2]))
    z=F.normalize(torch.randn(4,8,generator=gen,dtype=DT),dim=1);w=weight_logits(p).requires_grad_();dw=torch.randn(4,generator=gen,dtype=DT)
    base=errors(z,p,0,model,q,[0,1],[0,1,2],target).detach().clamp_min(1e-12)
    def fun(v):return (errors(z,v.softmax(0),0,model,q,[0,1],[0,1,2],target)/base).mean()
    grad=torch.autograd.grad(fun(w),w)[0];eps=1e-6
    error=abs(float((fun(w+eps*dw)-fun(w-eps*dw))/(2*eps)-(grad*dw).sum()));assert error<1e-5
    bg={k:dict(absolute_error=1.) for k in ('old','innovation','expanded')}
    rr={k:dict(seen_W1=1.,reserved_W1=1.,sliced_W1=1.,teacher_correct_mass=.5,seen=dict(near_zero_mass=.1)) for k in ('old','expanded')}
    assert not endpoint_gate(bg,bg,rr,rr,rr)['accepted']
    cg=copy.deepcopy(bg);cg['old']['absolute_error']=.9;assert endpoint_gate(bg,cg,rr,rr,rr)['accepted']
    bad=copy.deepcopy(rr);bad['expanded']['reserved_W1']=1.1
    assert not endpoint_gate(bg,cg,rr,rr,bad)['accepted']
    cg['innovation']['absolute_error']=1.1;assert not endpoint_gate(bg,cg,rr,rr,rr)['accepted']
    result.update(weight_objective_FD_error=error,weight_roundtrip=True,endpoint_accept_reject_checks=True,synthetic_updates=0)
    return result


def run_scale(cfg,lock,journal):
    start=time.monotonic();journal(dict(phase='load_saved_teacher_and_moment'))
    state=torch.load(Path(os.environ['Q350_PRIOR'])/(cfg['id']+'.state.private.pt'),map_location='cpu',weights_only=True)
    assert state['case']==cfg
    teacher=state['teacher'];q=state['frame'];d=cfg['dimension'];k=cfg['classes']
    old_n=cfg['task_sizes'][0];expanded_n=sum(cfg['task_sizes'][:2]);moment=state['memories']['moment']
    assert q.shape==(d,k) and all(a.shape==(d,8) and a.dtype==DT for a in teacher)
    assert set(moment)==set(range(old_n)) and all(z.shape==(16,d) and p.shape==(16,) for z,p in moment.values())
    x,y,counts=lowrank_fixture(cfg);current={c:x[y==c] for c in range(old_n)}
    load_seconds=time.monotonic()-start
    journal(dict(phase='candidate_construction',completed_updates=0))
    original_teacher=clone_model(teacher);original_moment={c:(z.clone(),p.clone()) for c,(z,p) in moment.items()}
    candidate,cost=construct(current,moment,teacher,q,list(range(old_n)),list(range(expanded_n)),journal)
    memories={'moment':moment,'candidate':candidate};costs=[dict(mode='candidate',**cost)]
    frozen_teacher=clone_model(teacher);frozen_memory={m:{c:(z.clone(),p.clone()) for c,(z,p) in mem.items()} for m,mem in memories.items()}
    diagnostic_start=time.monotonic();model=clone_model(teacher,True)
    gradients=[];responses=[];losses=[];aggregates=[];gg={};rr={};ll={};acceptance=[];selected={}
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
        gi={m:{v['query']:v for v in gradients if v['class_index']==c and v['mode']==m} for m in ('moment','candidate')}
        ri={m:{v['query']:v for v in responses if v['class_index']==c and v['mode']==m} for m in ('full_support','moment','candidate')}
        gate=endpoint_gate(gi['moment'],gi['candidate'],ri['full_support'],ri['moment'],ri['candidate'])
        source='candidate' if gate['accepted'] else 'moment';acceptance.append(dict(class_index=c,selected_source=source,**gate))
        selected[c]=memories[source][c];gg[c,'selected']=gg[c,source]
        for rows in (gradients,responses,losses):
            rows.extend([dict(v,mode='selected',selected_source=source) for v in rows if v['class_index']==c and v['mode']==source])
    torch.save(dict(candidate=candidate,selected=selected,acceptance=acceptance,case=cfg),Path(os.environ['Q350_ROOT'])/(cfg['id']+'.candidate.private.pt'))
    for query in ('old','expanded','innovation'):
        reference=torch.stack([gg[c,'full_support'][query] for c in range(old_n)])
        for mode in ('moment','candidate','selected'):
            stack=torch.stack([gg[c,mode][query] for c in range(old_n)]);err=stack-reference
            mean_error=float(err.norm(dim=1).mean());aggregate_error=float(err.mean(0).norm())
            aggregates.append(dict(mode=mode,query=query,classes=old_n,**comparison(stack.mean(0),reference.mean(0)),
                mean_class_absolute_error=mean_error,cancellation_ratio=aggregate_error/mean_error if mean_error>1e-12 else None,
                mean_class_relative_error=sum(comparison(g,r)['relative_error'] for g,r in zip(stack,reference))/old_n if bool((reference.norm(dim=1)>1e-12).all()) else None))
    assert all(torch.equal(a,b) for a,b in zip(teacher,frozen_teacher))
    assert all(torch.equal(a,b) for a,b in zip(model,teacher))
    assert all(torch.equal(a,b) for m,mem in memories.items() for c,zp in mem.items() for a,b in zip(zp,frozen_memory[m][c]))
    assert len(gradients)==old_n*12 and len(responses)==old_n*8 and len(losses)==old_n*12
    assert all(torch.equal(a,b) for a,b in zip(teacher,original_teacher))
    assert all(torch.equal(a,b) for c,zp in moment.items() for a,b in zip(zp,original_moment[c]))
    assert all(torch.equal(moment[c][0],candidate[c][0]) for c in moment)
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,old_classes=old_n,expanded_classes=expanded_n,
        acceptance=acceptance,fixed_nodes_unchanged=True,gradients=gradients,responses=responses,losses=losses,aggregates=aggregates,training=dict(seconds=0,updates=0,reused=True),construction=costs,
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
