"""CPU finite-query convex calibration of fixed synthetic moment nodes."""
from check_progressive_decision_memory import (DT, scalar, project_ball, family, losses, grouped,
    memory_losses, gap, constrain_family, fixture, moments_objective)
import os
import time
import torch
import torch.nn.functional as F


def weight_mse(table, target, weights):
    error=torch.einsum('cm,cmq->cq',weights,table)-target
    return error.square().mean(1),error


def fw_step(table,target,weights):
    before,error=weight_mse(table,target,weights)
    gradient=2*torch.einsum('cq,cmq->cm',error,table)/table.shape[2]
    vertices=F.one_hot(gradient.argmin(1),weights.shape[1]).to(DT)
    direction=vertices-weights
    delta=torch.einsum('cm,cmq->cq',direction,table)
    denom=delta.square().sum(1)
    alpha=torch.where(denom>0,-(error*delta).sum(1)/denom.clamp_min(1e-30),torch.zeros_like(denom)).clamp(0,1)
    updated=weights+alpha[:,None]*direction
    after,_=weight_mse(table,target,updated)
    assert bool(torch.all(after<=before+1e-12*(1+before)))
    assert float(updated.min())>=-1e-14 and float((updated.sum(1)-1).abs().max())<1e-12
    return updated


def fw_gap(table,target,weights):
    _,error=weight_mse(table,target,weights)
    gradient=2*torch.einsum('cq,cmq->cm',error,table)/table.shape[2]
    return (gradient*weights).sum(1)-gradient.min(1).values


def unit_checks():
    table=torch.tensor([[[0.],[2.]]],dtype=DT);target=torch.tensor([[.5]],dtype=DT)
    weights=torch.tensor([[.5,.5]],dtype=DT)
    result=fw_step(table,target,weights)
    exact_error=float((result-torch.tensor([[.75,.25]],dtype=DT)).abs().max())
    assert exact_error<1e-12 and float(fw_gap(table,target,result).max())<1e-12
    # Infeasible target has a boundary optimum; zero residual is not assumed.
    bad=torch.tensor([[3.]],dtype=DT);boundary=fw_step(table,bad,weights)
    residual=float(weight_mse(table,bad,boundary)[0].item())
    assert residual==1 and float(fw_gap(table,bad,boundary).max())==0
    g=torch.Generator().manual_seed(275)
    a=torch.randn(2,5,3,generator=g,dtype=DT);t=torch.randn(2,3,generator=g,dtype=DT)
    w=torch.randn(2,5,generator=g,dtype=DT).softmax(1).requires_grad_()
    grad=torch.autograd.grad(weight_mse(a,t,w)[0].sum(),w)[0]
    direction=torch.randn(2,5,generator=g,dtype=DT);direction-=direction.mean(1,keepdim=True);direction/=direction.norm()
    eps=1e-6
    fd=(weight_mse(a,t,w.detach()+eps*direction)[0].sum()-weight_mse(a,t,w.detach()-eps*direction)[0].sum())/(2*eps)
    fd_error=abs(float(fd-(grad*direction).sum()));assert fd_error<1e-6
    frozen=a.clone();before=w.detach().clone()
    for _ in range(4):before=fw_step(a,t,before)
    assert torch.equal(a,frozen)
    perm=torch.randperm(5,generator=g);permutation_error=float((weight_mse(a,t,before)[0]-weight_mse(a[:,perm],t,before[:,perm])[0]).abs().max());assert permutation_error<1e-12
    return dict(status='PASS',exact_line_solution_error=exact_error,weight_gradient_fd_error=fd_error,
        infeasible_target_optimal_MSE=residual,infeasible_target_FW_gap=0.,permutation_error=permutation_error,
        fixed_nodes_verified=True,warning='FW gap bounds fixed-node finite-query convex optimum only, not future query error.')


def run_scale(cfg,lock,journal):
    begin=time.monotonic();x,y,counts=fixture(cfg);c,d=cfg['classes'],cfg['dimension'];m=lock['nodes_per_class']
    assert lock['moment_steps']+lock['weight_steps']==lock['outer_steps']==48
    means=torch.stack([x[y==j].mean(0) for j in range(c)])
    second=torch.stack([x[y==j].T@x[y==j]/counts[j] for j in range(c)])
    gen=torch.Generator().manual_seed(cfg['seed']+1)
    init=torch.stack([x[y==j][torch.randint(counts[j],(m,),generator=gen)] for j in range(c)])
    zero=torch.zeros(c,m,dtype=DT)
    memories={'mean':(means[:,None].expand(-1,m,-1).clone(),zero.clone()),'random':(init.clone(),zero.clone())}
    trace=[];resource=[]
    def emit(phase,step):journal(dict(phase=phase,step=step,elapsed_seconds=time.monotonic()-begin,completed_memory_modes=list(memories),trace=trace,mode_costs=resource))
    started=time.monotonic();nodes=init.clone().requires_grad_();logits=zero.clone().requires_grad_()
    optimizer=torch.optim.Adam([nodes,logits],lr=lock['memory_lr'])
    prefix_seconds=None
    for step in range(lock['outer_steps']):
        optimizer.zero_grad();obj=moments_objective(nodes,logits,means,second);obj.backward();optimizer.step()
        with torch.no_grad():nodes.copy_(project_ball(nodes));logits.clamp_(-8,8)
        if step+1==lock['moment_steps']:
            memories['moment24']=(nodes.detach().clone(),logits.detach().clone());prefix_seconds=time.monotonic()-started
        if step%8==0 or step+1==lock['outer_steps']:
            trace.append(dict(mode='moment',step=step+1,objective=scalar(obj)));emit('moment',step+1)
    memories['moment']=(nodes.detach().clone(),logits.detach().clone())
    resource.append(dict(mode='shared_moment_construction',wall_seconds=time.monotonic()-started,prefix24_seconds_included=prefix_seconds,memory_optimizer_steps=48,query_optimizer_steps=0))
    started=time.monotonic();fixed_nodes,initial_logits=memories['moment24'];node_snapshot=fixed_nodes.clone()
    params=[p.requires_grad_() for p in family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+2)]
    optimizer=torch.optim.Adam(params,lr=lock['query_lr'])
    for step in range(lock['outer_steps']*lock['adversary_steps']):
        optimizer.zero_grad();obj=-gap(x,y,fixed_nodes,initial_logits,params).square().mean();obj.backward();optimizer.step();constrain_family(params)
        if step%16==0:emit('query_construction',step+1)
    frozen=[p.detach().clone() for p in params]
    with torch.no_grad():
        target=grouped(losses(x,frozen),y,c)
        table=losses(fixed_nodes.reshape(c*m,d),frozen).reshape(c,m,-1)
        weights=initial_logits.softmax(1);initial_mse=weight_mse(table,target,weights)[0]
        for step in range(lock['weight_steps']):
            weights=fw_step(table,target,weights)
            if step%4==0 or step+1==lock['weight_steps']:
                trace.append(dict(mode='convex_weight',step=step+1,objective=float(weight_mse(table,target,weights)[0].mean()),max_FW_gap=float(fw_gap(table,target,weights).max())));emit('convex_weight',step+1)
        assert torch.equal(fixed_nodes,node_snapshot)
        final_mse=weight_mse(table,target,weights)[0];gaps=fw_gap(table,target,weights)
        memories['convex_weight']=(fixed_nodes.clone(),weights.log())
        fit_diagnostics=dict(initial_class_MSE=initial_mse.tolist(),final_class_MSE=final_mse.tolist(),FW_gap_per_class=gaps.tolist(),
            mean_initial_MSE=float(initial_mse.mean()),mean_final_MSE=float(final_mse.mean()),max_FW_gap=float(gaps.max()),
            nodes_unchanged=True,queries_frozen=True,weight_steps=lock['weight_steps'],node_steps=lock['moment_steps'],
            interpretation='Nonnegative FW gap is an upper bound on per-class suboptimality for this fixed-node squared finite-query loss; not a universal loss bound.')
    resource.append(dict(mode='query_and_weight_calibration',wall_seconds=time.monotonic()-started,memory_optimizer_steps=lock['weight_steps'],query_optimizer_steps=96,
        shared_prefix24_not_charged_twice=True))
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
    passed=all(row('convex_weight')['max_abs_error'] <= .9*row(b)['max_abs_error'] and
               row('convex_weight')['mean_abs_error'] <= row(b)['mean_abs_error'] for b in ('moment','random'))
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,all_support_rows_used=len(x),
                real_medical_fit_allocation=False,fit_diagnostics=fit_diagnostics,rows=rows,mode_costs=resource,trace=trace,
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
