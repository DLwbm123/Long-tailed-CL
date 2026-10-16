"""Frozen first-stage reconstruction and read-only query audit; no task-2 training."""
import os
import time
from pathlib import Path
import torch
from check_fullslot_retention_memory import train, compress
from check_reserved_competition_memory import (DT, initial_model, clone_model, frame, scores,
    response, risk_gradient, wasserstein_1d, sliced, lowrank_fixture)


def comparison(g, ref):
    gn, rn = float(g.norm()), float(ref.norm())
    error = float((g-ref).norm())
    return dict(norm=gn, reference_norm=rn, absolute_error=error,
                relative_error=error/rn if rn>1e-12 else None,
                cosine=float(torch.dot(g,ref))/(gn*rn) if gn>1e-12 and rn>1e-12 else None)


def distribution(v, p):
    values, order = v.sort(); weights = p[order]
    cumulative = weights.cumsum(0); cumulative[-1] = 1.
    levels = v.new_tensor([0., .1, .25, .5, .75, .9, 1.])
    idx = torch.searchsorted(cumulative, levels).clamp_max(len(v)-1)
    return dict(values=values.tolist(), weights=weights.tolist(), quantile_levels=levels.tolist(),
                quantiles=values[idx].tolist(), mean=float((v*p).sum()),
                nonpositive_mass=float((p*(v<=0)).sum()), near_zero_mass=float((p*(v.abs()<=.25)).sum()))


def unit_checks():
    z=torch.tensor([[.2,-.4,.7,-.1],[.3,.1,-.2,.4]],dtype=DT,requires_grad=True)
    old=-z[:,:2].log_softmax(1)[:,0].mean(); expanded=-z.log_softmax(1)[:,0].mean()
    innovation=torch.nn.functional.softplus(z[:,2:].logsumexp(1)-z[:,:2].logsumexp(1)).mean()
    a=torch.autograd.grad(expanded-old,z,retain_graph=True)[0]
    b=torch.autograd.grad(innovation,z)[0]
    assert abs(float(expanded-old-innovation))<1e-12 and float((a-b).abs().max())<1e-12
    v=torch.tensor([-1.,0.,2.],dtype=DT);p=torch.ones(3,dtype=DT)/3
    assert float(wasserstein_1d(v,v,p))<1e-12
    assert abs(float(wasserstein_1d(v,v+2,p))-2)<1e-12
    dist=distribution(v,v.new_tensor([.1,.7,.2]));assert dist['quantiles'][3]==0.
    assert comparison(v,v)['absolute_error']==0 and comparison(v,v*0)['relative_error'] is None
    return dict(status='PASS',CE_decomposition_error=abs(float(expanded-old-innovation)),
                gradient_decomposition_error=float((a-b).abs().max()),W1_and_quantiles=True,synthetic_updates=0)


def run_scale(cfg, lock, journal):
    start=time.monotonic();x,y,counts=lowrank_fixture(cfg);d=cfg['dimension'];k=cfg['classes']
    old_n=cfg['task_sizes'][0];expanded_n=sum(cfg['task_sizes'][:2])
    current={c:x[y==c] for c in range(old_n)};q=frame(d,k,cfg['seed']);initial=initial_model(d,cfg['seed'])
    journal(dict(phase='task1_teacher_reconstruction'))
    teacher,training=train(initial,current,{},q,list(range(old_n)),steps=48,full=False)
    memories={};costs=[]
    for mode in ('moment','decision'):
        journal(dict(phase='task1_memory_reconstruction',mode=mode))
        memories[mode],cost=compress(current,cfg['seed'],mode,initial,teacher,q,list(range(old_n)))
        costs.append(dict(mode=mode,**cost))
    assert training['updates']==48 and sum(c['updates'] for c in costs)==96
    frozen_teacher=clone_model(teacher);frozen_memory={m:{c:(z.clone(),p.clone()) for c,(z,p) in mem.items()} for m,mem in memories.items()}
    # Save the rebuilt state once; diagnostics never feed back into construction.
    torch.save(dict(teacher=teacher,memories=memories,frame=q,case=cfg),Path(os.environ['Q348_ROOT'])/(cfg['id']+'.state.private.pt'))
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
        for mode in ('moment','decision'):
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
        gradients=gradients,responses=responses,losses=losses,aggregates=aggregates,training=training,construction=costs,
        teacher_and_memories_unchanged=True,state_saved_private=True,task2_updates=0,total_adapter_updates=48,
        actual_memory_updates=96,adapter_parameters=2*d*8,memory_nodes_per_class=16,
        diagnostic_seconds=time.monotonic()-diagnostic_start,CPU_pipeline_seconds=time.monotonic()-start)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,val_accessed=False,test_accessed=False)
    assert not result['GPU_initialized']
    return result
