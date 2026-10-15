"""CPU domain contrast: unit-sphere synthetic nodes versus unit-ball moment construction."""
from check_progressive_decision_memory import (DT, scalar, project_ball, family, losses, grouped,
    memory_losses, gap, constrain_family, fixture, moments_objective)
from check_convex_weight_memory import weight_mse, fw_step, fw_gap
import os
import time
import torch
import torch.nn.functional as F


def sphere(nodes):
    norm=nodes.norm(dim=-1,keepdim=True)
    assert bool(torch.all(norm>1e-12)), 'Undefined spherical projection; retain failure rather than invent a direction'
    return nodes/norm


def geometry(nodes,logits,means,second):
    w=logits.softmax(1);norms=nodes.norm(dim=2)
    mu=(nodes*w[:,:,None]).sum(1)
    q=torch.einsum('cm,cmd,cme->cde',w,nodes,nodes)
    return dict(per_class_min_radius=norms.min(1).values.tolist(),per_class_max_radius=norms.max(1).values.tolist(),
        per_class_weighted_radius=(w*norms).sum(1).tolist(),per_class_second_trace=(w*norms.square()).sum(1).tolist(),
        per_class_mean_sq_error=(mu-means).square().sum(1).tolist(),per_class_second_fro_sq_error=(q-second).square().sum((1,2)).tolist())


def unit_checks():
    g=torch.Generator().manual_seed(279)
    nodes=torch.randn(3,5,7,generator=g,dtype=DT).requires_grad_()
    out=sphere(nodes);norm_error=float((out.norm(dim=2)-1).abs().max());assert norm_error<1e-12
    weights=torch.randn(3,5,generator=g,dtype=DT).softmax(1)
    trace_error=float(((weights*out.square().sum(2)).sum(1)-1).abs().max());assert trace_error<1e-12
    direction=torch.randn(nodes.shape,generator=g,dtype=DT);direction/=direction.norm()
    def f(v):return memory_losses(sphere(v),weights.log(),family(7,2,4,279)).square().mean()
    grad=torch.autograd.grad(f(nodes),nodes)[0];eps=1e-6
    fd=(f(nodes.detach()+eps*direction)-f(nodes.detach()-eps*direction))/(2*eps)
    error=abs(float(fd-(grad*direction).sum()));assert error<1e-6
    zero_rejected=False
    try:sphere(torch.zeros(1,1,7,dtype=DT))
    except AssertionError:zero_rejected=True
    assert zero_rejected
    d=128;q=torch.eye(d,dtype=DT)/d
    missing_fro=float(q.square().sum());assert abs(missing_fro-1/d)<1e-15
    return dict(status='PASS',sphere_norm_error=norm_error,weighted_trace_error=trace_error,
        sphere_composition_gradient_fd_error=error,zero_direction_rejected=zero_rejected,
        isotropic_dimension=d,zero_memory_missing_trace=1.,zero_memory_second_Fro_error=missing_fro,
        warning='Trace example proves a possible moment-objective blind spot, not the cause of all previous failures.')


def run_scale(cfg,lock,journal):
    begin=time.monotonic();x,y,counts=fixture(cfg);c,d=cfg['classes'],cfg['dimension'];m=lock['nodes_per_class']
    assert lock['moment_steps']+lock['weight_steps']==lock['outer_steps']==48
    means=torch.stack([x[y==j].mean(0) for j in range(c)])
    second=torch.stack([x[y==j].T@x[y==j]/counts[j] for j in range(c)])
    gen=torch.Generator().manual_seed(cfg['seed']+1)
    init=torch.stack([x[y==j][torch.randint(counts[j],(m,),generator=gen)] for j in range(c)])
    zero=torch.zeros(c,m,dtype=DT)
    memories={'random':(init.clone(),zero.clone())}
    trace=[];resource=[]
    def emit(phase,step):journal(dict(phase=phase,step=step,elapsed_seconds=time.monotonic()-begin,completed_memory_modes=list(memories),trace=trace,mode_costs=resource))
    for domain in ('ball','sphere'):
        started=time.monotonic();nodes=init.clone().requires_grad_();logits=zero.clone().requires_grad_()
        optimizer=torch.optim.Adam([nodes,logits],lr=lock['memory_lr'])
        prefix_seconds=None
        for step in range(lock['outer_steps']):
            optimizer.zero_grad();obj=moments_objective(nodes,logits,means,second);obj.backward();optimizer.step()
            with torch.no_grad():
                nodes.copy_(project_ball(nodes) if domain=='ball' else sphere(nodes));logits.clamp_(-8,8)
            if domain=='sphere' and step+1==lock['moment_steps']:
                memories['sphere24']=(nodes.detach().clone(),logits.detach().clone());prefix_seconds=time.monotonic()-started
            if step%8==0 or step+1==lock['outer_steps']:
                trace.append(dict(mode=domain,step=step+1,objective=scalar(obj)));emit(domain,step+1)
        memories['moment' if domain=='ball' else 'sphere48']=(nodes.detach().clone(),logits.detach().clone())
        resource.append(dict(mode=domain+'_moment_construction',wall_seconds=time.monotonic()-started,prefix24_seconds_included=prefix_seconds,memory_optimizer_steps=48,query_optimizer_steps=0))
    started=time.monotonic();fixed_nodes,initial_logits=memories['sphere24'];node_snapshot=fixed_nodes.clone()
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
                trace.append(dict(mode='sphere_weight',step=step+1,objective=float(weight_mse(table,target,weights)[0].mean()),max_FW_gap=float(fw_gap(table,target,weights).max())));emit('sphere_weight',step+1)
        assert torch.equal(fixed_nodes,node_snapshot)
        final_mse=weight_mse(table,target,weights)[0];gaps=fw_gap(table,target,weights)
        memories['sphere_weight']=(fixed_nodes.clone(),weights.log())
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
                             geometry=geometry(nodes,weights,means,second),
                             min_effective_nodes=float((1/weights.softmax(1).square().sum(1)).min())))
    def row(name):return next(r for r in rows if r['mode']==name)
    # Fixed prospective screening rule; passing is only a finite-query signal, never a uniform certificate.
    passed=all(row('sphere_weight')['max_abs_error'] <= .9*row(b)['max_abs_error'] and
               row('sphere_weight')['mean_abs_error'] <= row(b)['mean_abs_error'] for b in ('moment','random'))
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
