"""CPU mechanism contrast: frozen query construction versus moving query-memory game."""
from check_progressive_decision_memory import (DT, scalar, project_ball, family, losses, grouped,
    memory_losses, gap, constrain_family, fixture, moments_objective)
from check_barycentric_decision_memory import simplex
import os
import time
import torch
import torch.nn.functional as F


def unit_checks():
    g=torch.Generator().manual_seed(271)
    x=F.normalize(torch.randn(12,6,generator=g,dtype=DT),dim=1)
    y=torch.tensor([0]*6+[1]*6)
    queries=[p.detach().clone() for p in family(6,2,4,271)]
    snapshot=[p.clone() for p in queries]
    target=grouped(losses(x,queries),y,2);target_saved=target.clone()
    coeff=[torch.randn(3,6,generator=g,dtype=DT).softmax(1).requires_grad_() for _ in range(2)]
    weights=torch.zeros(2,3,dtype=DT,requires_grad=True)
    support=[x[y==j] for j in range(2)]
    def objective(a):
        nodes=torch.stack([a[0]@support[0],a[1]@support[1]])
        return (target-memory_losses(nodes,weights,queries)).square().mean()
    grads=torch.autograd.grad(objective(coeff),coeff)
    directions=[torch.randn(3,6,generator=g,dtype=DT) for _ in coeff]
    for p in directions:p.sub_(p.mean(1,keepdim=True));p.div_(p.norm())
    eps=1e-6
    fd=(objective([a.detach()+eps*d for a,d in zip(coeff,directions)])-objective([a.detach()-eps*d for a,d in zip(coeff,directions)]))/(2*eps)
    error=abs(float(fd-sum((a*b).sum() for a,b in zip(grads,directions))))
    assert error<1e-6
    optimizer=torch.optim.Adam(coeff+[weights],lr=.02)
    for _ in range(3):
        optimizer.zero_grad();objective(coeff).backward();optimizer.step()
        with torch.no_grad():
            for p in coeff:p.copy_(simplex(p))
        assert all(torch.equal(a,b) and a.grad is None for a,b in zip(queries,snapshot))
        assert torch.equal(target,target_saved)
    perm=torch.randperm(12,generator=g)
    perm_error=float((target-grouped(losses(x[perm],queries),y[perm],2)).abs().max())
    assert perm_error<1e-12
    return dict(status='PASS',fixed_query_gradient_fd_error=error,query_and_target_immutable=True,
        synthetic_unit_optimizer_updates=3,permutation_error=perm_error,
        warning='Immutability and gradient engineering evidence, not optimization convergence or unseen-query coverage.')


def run_scale(cfg, lock, journal):
    begin=time.monotonic(); x,y,counts=fixture(cfg); c,d=cfg['classes'],cfg['dimension']; m=lock['nodes_per_class']
    means=torch.stack([x[y==j].mean(0) for j in range(c)])
    second=torch.stack([x[y==j].T@x[y==j]/counts[j] for j in range(c)])
    gen=torch.Generator().manual_seed(cfg['seed']+1)
    support=[x[y==j] for j in range(c)]
    indices=[torch.randint(counts[j],(m,),generator=gen) for j in range(c)]
    init=torch.stack([support[j][indices[j]] for j in range(c)])
    zero=torch.zeros(c,m,dtype=DT)
    memories={'mean':(means[:,None].expand(-1,m,-1).clone(),zero.clone()),'random':(init.clone(),zero.clone())}
    resource=[]; trace=[]; fit_diagnostics=[]
    def emit(phase,step):
        journal(dict(phase=phase,step=step,elapsed_seconds=time.monotonic()-begin,
                     completed_memory_modes=list(memories),trace=trace,mode_costs=resource))
    emit('fixture',0)
    for mode in ('moment','barycentric','frozen_query'):
        started=time.monotonic();nodes=init.clone().requires_grad_();logits=zero.clone().requires_grad_()
        coeff=[F.one_hot(indices[j],counts[j]).to(DT).requires_grad_() for j in range(c)] if mode in ('barycentric','frozen_query') else []
        opt=torch.optim.Adam(coeff+[logits] if coeff else [nodes,logits],lr=lock['memory_lr'])
        params=[p.requires_grad_() for p in family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+2)]
        adv=torch.optim.Adam(params,lr=lock['query_lr'])
        query_snapshot=None
        if mode=='frozen_query':
            for step in range(lock['outer_steps']*lock['adversary_steps']):
                adv.zero_grad()
                objective=-gap(x,y,init,zero,params).square().mean()
                objective.backward();adv.step();constrain_family(params)
                if step%16==0:emit('frozen_query_construction',step+1)
            frozen=[p.detach().clone() for p in params]
            query_snapshot=[p.clone() for p in frozen]
            with torch.no_grad():
                fixed_target=grouped(losses(x,frozen),y,c)
                initial_fixed_mse=scalar((fixed_target-memory_losses(init,zero,frozen)).square().mean())
        for step in range(lock['outer_steps']):
            if mode in ('barycentric','frozen_query'):nodes=torch.stack([a@v for a,v in zip(coeff,support)])
            if mode=='barycentric':
                for _ in range(lock['adversary_steps']):
                    adv.zero_grad()
                    error=gap(x,y,nodes.detach(),logits.detach(),params)
                    objective=-error.square().mean();objective.backward();adv.step();constrain_family(params)
                frozen=[p.detach() for p in params]
                with torch.no_grad(): target=grouped(losses(x,frozen),y,c)
                objective=(target-memory_losses(nodes,logits,frozen)).square().mean()
            elif mode=='frozen_query':
                assert all(torch.equal(a,b) for a,b in zip(frozen,query_snapshot))
                objective=(fixed_target-memory_losses(nodes,logits,frozen)).square().mean()
            else: objective=moments_objective(nodes,logits,means,second)
            opt.zero_grad();objective.backward();opt.step()
            with torch.no_grad():
                if coeff:
                    for a in coeff:a.copy_(simplex(a))
                else:nodes.copy_(project_ball(nodes))
                logits.clamp_(-8,8)
            if step%8==0 or step+1==lock['outer_steps']:
                trace.append(dict(mode=mode,step=step+1,objective=scalar(objective)))
                emit(mode,step+1)
        with torch.no_grad():
            if coeff:
                nodes=torch.stack([a@v for a,v in zip(coeff,support)])
                assert all(float(a.min())>=0 and float((a.sum(1)-1).abs().max())<1e-10 for a in coeff)
                assert float(nodes.norm(dim=2).max())<=1+1e-10
        if mode in ('barycentric','frozen_query'):
            with torch.no_grad():
                terminal_queries=frozen
                terminal_target=grouped(losses(x,terminal_queries),y,c)
                final_mse=scalar((terminal_target-memory_losses(nodes,logits,terminal_queries)).square().mean())
                before_mse=scalar((terminal_target-memory_losses(init,zero,terminal_queries)).square().mean())
                assert mode!='frozen_query' or all(torch.equal(a,b) for a,b in zip(frozen,query_snapshot))
                fit_diagnostics.append(dict(mode=mode,initial_memory_MSE_on_terminal_queries=before_mse,
                    final_memory_MSE_on_terminal_queries=final_mse,
                    query_frozen_during_compression=mode=='frozen_query',
                    training_query_count=lock['queries']))
        memories[mode]=(nodes.detach().clone(),logits.detach().clone())
        resource.append(dict(mode=mode,wall_seconds=time.monotonic()-started,
                             memory_optimizer_steps=lock['outer_steps'],
                             query_optimizer_steps=lock['outer_steps']*lock['adversary_steps'] if mode in ('barycentric','frozen_query') else 0))
    # Shared evaluation queries; independent initializations; no memory updates from these queries.
    families=[family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+100+i) for i in range(4)]
    evaluation_started=time.monotonic()
    for i in range(2):
        params=[p.clone().requires_grad_() for p in families[i]]
        opt=torch.optim.Adam(params,lr=lock['query_lr'])
        for _ in range(lock['evaluation_adversary_steps']):
            opt.zero_grad()
            # Discover difficult directions jointly for ALL frozen methods, not just the new method.
            target=grouped(losses(x,params),y,c)
            obj=-torch.stack([(target-memory_losses(n,w,params)).square().mean() for n,w in memories.values()]).mean()
            obj.backward();opt.step();constrain_family(params)
        families.append([p.detach() for p in params])
    rows=[]
    with torch.no_grad():
        targets=[grouped(losses(x,p),y,c) for p in families]
        for mode,(nodes,weights) in memories.items():
            errors=torch.cat([(target-memory_losses(nodes,weights,p)).abs() for target,p in zip(targets,families)],1)
            random_errors=errors[:,:4*lock['queries']]; stress_errors=errors[:,4*lock['queries']:]
            rows.append(dict(mode=mode,mean_abs_error=float(errors.mean()),max_abs_error=float(errors.max()),
                             random_query_max_error=float(random_errors.max()),stress_query_max_error=float(stress_errors.max()),
                             per_class_max_errors=[float(v) for v in errors.max(1).values],
                             moment_objective=float(moments_objective(nodes,weights,means,second)),
                             logical_memory_bytes=nodes.numel()*8+weights.numel()*8,
                             max_node_norm=float(nodes.norm(dim=2).max()),
                             min_effective_nodes=float((1/weights.softmax(1).square().sum(1)).min())))
    def row(name):return next(r for r in rows if r['mode']==name)
    # Fixed prospective screening rule; passing is only a finite-query signal, never a uniform certificate.
    passed=all(row('frozen_query')['max_abs_error'] <= .9*row(b)['max_abs_error'] and
               row('frozen_query')['mean_abs_error'] <= row(b)['mean_abs_error'] for b in ('moment','random'))
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,all_support_rows_used=len(x),
                real_medical_fit_allocation=False,fit_diagnostics=fit_diagnostics,construction_coefficient_elements=m*len(x),
                construction_coefficient_bytes=m*len(x)*8,construction_support_bytes=x.numel()*8,
                construction_note='Coefficients/support temporary; deployment stores only synthetic nodes and weights. Adam state and workspaces reflected in process RSS.',rows=rows,mode_costs=resource,trace=trace,
                heldout_query_count=6*lock['queries'],evaluation_wall_seconds=time.monotonic()-evaluation_started,
                finite_query_screen_pass=passed,uniform_error_bound=None,classification_improvement=None,
                classification_training_performed=False,CPU_pipeline_seconds=time.monotonic()-begin)



def run(cfg, lock, journal):
    torch.set_num_threads(lock['threads'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_default_dtype(DT)
    result=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,
                  medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not result['GPU_initialized']
    return result
