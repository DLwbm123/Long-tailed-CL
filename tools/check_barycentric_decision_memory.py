"""New CPU structural contrast: class-support convex synthetic nodes, no medical inputs."""
from check_progressive_decision_memory import (DT, scalar, project_ball, family, losses, grouped,
    memory_losses, gap, constrain_family, fixture, moments_objective)
import os
import time
import torch
import torch.nn.functional as F


def simplex(x):
    ordered=x.sort(dim=1,descending=True).values
    partial=ordered.cumsum(1)-1
    k=torch.arange(1,x.shape[1]+1,dtype=x.dtype,device=x.device)
    rho=(ordered-partial/k>0).sum(1)-1
    threshold=partial.gather(1,rho[:,None])/(rho[:,None]+1)
    return (x-threshold).clamp_min(0)


def unit_checks():
    g=torch.Generator().manual_seed(267)
    raw=torch.randn(7,13,generator=g,dtype=DT)
    a=simplex(raw); again=simplex(a)
    assert float(a.min())>=0 and float((a.sum(1)-1).abs().max())<1e-12
    idem=float((a-again).abs().max());assert idem<1e-12
    support=F.normalize(torch.randn(13,6,generator=g,dtype=DT),dim=1)
    idx=torch.tensor([0,4,9]);one=F.one_hot(idx,13).to(DT)
    assert torch.equal(one@support,support[idx])
    perm=torch.randperm(13,generator=g)
    perm_error=float((a@support-a[:,perm]@support[perm]).abs().max());assert perm_error<1e-12
    coeff=torch.randn(3,13,generator=g,dtype=DT).softmax(1).requires_grad_()
    p=family(6,2,4,267);weights=torch.zeros(1,3,dtype=DT)
    def objective(v):return memory_losses((v@support)[None],weights,p).square().mean()
    grad=torch.autograd.grad(objective(coeff),coeff)[0]
    direction=torch.randn(3,13,generator=g,dtype=DT);direction-=direction.mean(1,keepdim=True);direction/=direction.norm()
    eps=1e-6;fd=(objective(coeff.detach()+eps*direction)-objective(coeff.detach()-eps*direction))/(2*eps)
    err=abs(float(fd-(grad*direction).sum()));assert err<1e-6
    # Projection KKT: active coordinates share a threshold; inactive coordinates lie below it.
    residual=raw-a
    errors=[]
    for row,proj,res in zip(raw,a,residual):
        threshold=res[proj>0].mean()
        errors.append(float((res[proj>0]-threshold).abs().max()))
        if bool((proj==0).any()):assert float((row[proj==0]-threshold).max())<=1e-12
    assert max(errors)<1e-12
    return dict(status='PASS',simplex_idempotence_error=idem,projection_KKT_error=max(errors),
        barycentric_gradient_fd_error=err,permutation_error=perm_error,same_initial_nodes_exact=True,
        projected_norm_max=float((a@support).norm(dim=1).max()),
        warning='Derivative checked inside simplex before projected optimizer step; no implicit gradient claim.')


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
    resource=[]; trace=[]
    def emit(phase,step):
        journal(dict(phase=phase,step=step,elapsed_seconds=time.monotonic()-begin,
                     completed_memory_modes=list(memories),trace=trace,mode_costs=resource))
    emit('fixture',0)
    for mode in ('moment','decision','barycentric'):
        started=time.monotonic();nodes=init.clone().requires_grad_();logits=zero.clone().requires_grad_()
        coeff=[F.one_hot(indices[j],counts[j]).to(DT).requires_grad_() for j in range(c)] if mode=='barycentric' else []
        opt=torch.optim.Adam(coeff+[logits] if coeff else [nodes,logits],lr=lock['memory_lr'])
        params=[p.requires_grad_() for p in family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+2)]
        adv=torch.optim.Adam(params,lr=lock['query_lr'])
        for step in range(lock['outer_steps']):
            if mode=='barycentric':nodes=torch.stack([a@v for a,v in zip(coeff,support)])
            if mode in ('decision','barycentric'):
                for _ in range(lock['adversary_steps']):
                    adv.zero_grad()
                    error=gap(x,y,nodes.detach(),logits.detach(),params)
                    objective=-error.square().mean();objective.backward();adv.step();constrain_family(params)
                frozen=[p.detach() for p in params]
                with torch.no_grad(): target=grouped(losses(x,frozen),y,c)
                objective=(target-memory_losses(nodes,logits,frozen)).square().mean()
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
        memories[mode]=(nodes.detach().clone(),logits.detach().clone())
        resource.append(dict(mode=mode,wall_seconds=time.monotonic()-started,
                             memory_optimizer_steps=lock['outer_steps'],
                             query_optimizer_steps=lock['outer_steps']*lock['adversary_steps'] if mode in ('decision','barycentric') else 0))
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
    passed=all(row('barycentric')['max_abs_error'] <= .9*row(b)['max_abs_error'] and
               row('barycentric')['mean_abs_error'] <= row(b)['mean_abs_error'] for b in ('moment','random'))
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,all_support_rows_used=len(x),
                real_medical_fit_allocation=False,construction_coefficient_elements=m*len(x),
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
