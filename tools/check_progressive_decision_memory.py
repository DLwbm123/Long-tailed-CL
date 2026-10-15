"""CPU-only falsification of synthetic loss memory; no medical data or production imports."""
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DT = torch.float64


def scalar(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def project_ball(x, radius=1.):
    return x * (radius / x.norm(dim=-1, keepdim=True).clamp_min(radius))


def family(d, rank, queries, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(d, rank, generator=gen, dtype=DT)
    b = torch.randn(d, rank, generator=gen, dtype=DT)
    a = a / a.norm(); b = b / b.norm()
    v = F.normalize(torch.randn(queries, d, generator=gen, dtype=DT), dim=1)
    return [a, b, v]


def features(x, params):
    a, b, _ = params
    return F.normalize(x + torch.tanh(3 * (x @ b)) @ a.T, dim=1, eps=1e-8)


def losses(x, params):
    # v represents a bounded pairwise direction (u-v); beta is fixed, not val-selected.
    return F.softplus(8 * (features(x, params) @ params[2].T))


def grouped(values, labels, classes):
    sums = torch.zeros(classes, values.shape[1], dtype=DT)
    sums.index_add_(0, labels, values)
    count = torch.bincount(labels, minlength=classes).to(DT)
    return sums / count[:, None]


def memory_losses(nodes, logits, params):
    c, m, d = nodes.shape
    val = losses(nodes.reshape(c*m, d), params).reshape(c, m, -1)
    return (val * logits.softmax(1)[:, :, None]).sum(1)


def gap(x, y, nodes, logits, params):
    return grouped(losses(x, params), y, nodes.shape[0]) - memory_losses(nodes, logits, params)


def constrain_family(params):
    with torch.no_grad():
        for p in params[:2]: p.mul_(min(1., 2./max(float(p.norm()), 1e-12)))
        params[2].copy_(project_ball(params[2], 2.))


def unit_checks():
    gen = torch.Generator().manual_seed(263)
    # Two unit-norm distributions have identical first/second moments but different nonlinear losses.
    a = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=DT)
    b = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], dtype=DT)/math.sqrt(2.)
    moment_error = max(float((a.mean(0)-b.mean(0)).abs().max()), float((a.T@a/4-b.T@b/4).abs().max()))
    loss_gap = abs(float(F.softplus(8*a[:, 0]).mean()-F.softplus(8*b[:, 0]).mean()))
    assert moment_error < 1e-12 and loss_gap > .01
    # ETF expansion leaves rescaled old-old logits unchanged at FIXED representation only.
    u = torch.linalg.qr(torch.randn(32, 12, generator=gen, dtype=DT)).Q
    h = F.normalize(torch.randn(19, 32, generator=gen, dtype=DT), dim=1)
    def scores(k):
        alpha = math.sqrt(k/(k-1)); e = alpha*(u[:, :k]-u[:, :k].mean(1, keepdim=True))
        return h@e/alpha
    s0, s1 = scores(4), scores(12)
    err = float(((s0[:, 0]-s0[:, 1])-(s1[:, 0]-s1[:, 1])).abs().max())
    assert err < 1e-12
    # Memory-position gradient, including the nonlinear adapter and normalization.
    nodes = torch.randn(2, 3, 6, generator=gen, dtype=DT)*.1; nodes.requires_grad_()
    params = family(6, 2, 4, 265); logits = torch.zeros(2, 3, dtype=DT)
    f = memory_losses(nodes, logits, params).square().mean()
    grad = torch.autograd.grad(f, nodes)[0]
    direction = F.normalize(torch.randn(nodes.numel(), generator=gen, dtype=DT), dim=0).reshape_as(nodes)
    eps=1e-6
    fp = memory_losses(nodes.detach()+eps*direction, logits, params).square().mean()
    fm = memory_losses(nodes.detach()-eps*direction, logits, params).square().mean()
    fd_error = abs(float((fp-fm)/(2*eps)-(grad*direction).sum()))
    assert fd_error < 1e-6
    # Permuting support rows has no effect on the empirical loss; exact memory is exact.
    x = nodes.detach().reshape(6, 6); y = torch.tensor([0,0,0,1,1,1]); perm=torch.randperm(6,generator=gen)
    target=grouped(losses(x,params),y,2)
    perm_error=float((target-grouped(losses(x[perm],params),y[perm],2)).abs().max())
    exact_error=float((target-memory_losses(nodes.detach(),logits,params)).abs().max())
    assert max(perm_error,exact_error)<1e-12
    return dict(status='PASS',moment_match_error=moment_error,matched_moment_nonlinear_loss_gap=loss_gap,
                progressive_fixed_feature_logit_error=err,memory_gradient_fd_error=fd_error,
                permutation_error=perm_error,exact_memory_error=exact_error,
                warning='geometry property does not protect encoder updates or new-class competition')


def fixture(cfg):
    g=torch.Generator().manual_seed(cfg['seed']); d=cfg['dimension']; c=cfg['classes']; n=cfg['n']
    raw=torch.tensor(cfg['count_weights'],dtype=DT)
    allocation=(n-c)*raw/raw.sum(); counts=allocation.floor().long()+1
    remain=n-int(counts.sum()); ids=torch.argsort(allocation-allocation.floor(),descending=True,stable=True)
    counts[ids[:remain]]+=1
    xs=[]; ys=[]
    for j,count in enumerate(counts.tolist()):
        noise=F.normalize(torch.randn(count,d,generator=g,dtype=DT),dim=1)
        center=torch.zeros(d,dtype=DT); center[j]=.45
        # A fixed asymmetric mixture retains higher-order structure beyond mean/covariance.
        mixture=torch.where(torch.arange(count)%5==0, .8, -.2).to(DT)
        xx=.85*noise+center; xx[:,c+j]+=.6*mixture
        xs.append(F.normalize(xx,dim=1));ys.append(torch.full((count,),j,dtype=torch.long))
    return torch.cat(xs),torch.cat(ys),counts.tolist()


def moments_objective(nodes, logits, means, second):
    w=logits.softmax(1); mu=(nodes*w[:,:,None]).sum(1)
    q=torch.einsum('cm,cmd,cme->cde',w,nodes,nodes)
    return (mu-means).square().sum(1).mean()+(q-second).square().sum((1,2)).mean()


def run_scale(cfg, lock, journal):
    begin=time.monotonic(); x,y,counts=fixture(cfg); c,d=cfg['classes'],cfg['dimension']; m=lock['nodes_per_class']
    means=torch.stack([x[y==j].mean(0) for j in range(c)])
    second=torch.stack([x[y==j].T@x[y==j]/counts[j] for j in range(c)])
    gen=torch.Generator().manual_seed(cfg['seed']+1)
    init=torch.stack([x[y==j][torch.randint(counts[j],(m,),generator=gen)] for j in range(c)])
    zero=torch.zeros(c,m,dtype=DT)
    memories={'mean':(means[:,None].expand(-1,m,-1).clone(),zero.clone()),'random':(init.clone(),zero.clone())}
    resource=[]; trace=[]
    def emit(phase,step):
        journal(dict(phase=phase,step=step,elapsed_seconds=time.monotonic()-begin,
                     completed_memory_modes=list(memories),trace=trace,mode_costs=resource))
    emit('fixture',0)
    for mode in ('moment','decision'):
        started=time.monotonic();nodes=init.clone().requires_grad_();logits=zero.clone().requires_grad_()
        opt=torch.optim.Adam([nodes,logits],lr=lock['memory_lr'])
        params=[p.requires_grad_() for p in family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+2)]
        adv=torch.optim.Adam(params,lr=lock['query_lr'])
        for step in range(lock['outer_steps']):
            if mode=='decision':
                for _ in range(lock['adversary_steps']):
                    adv.zero_grad()
                    error=gap(x,y,nodes.detach(),logits.detach(),params)
                    objective=-error.square().mean();objective.backward();adv.step();constrain_family(params)
                frozen=[p.detach() for p in params]
                with torch.no_grad(): target=grouped(losses(x,frozen),y,c)
                objective=(target-memory_losses(nodes,logits,frozen)).square().mean()
            else: objective=moments_objective(nodes,logits,means,second)
            opt.zero_grad();objective.backward();opt.step()
            with torch.no_grad():nodes.copy_(project_ball(nodes));logits.clamp_(-8,8)
            if step%8==0 or step+1==lock['outer_steps']:
                trace.append(dict(mode=mode,step=step+1,objective=scalar(objective)))
                emit(mode,step+1)
        memories[mode]=(nodes.detach().clone(),logits.detach().clone())
        resource.append(dict(mode=mode,wall_seconds=time.monotonic()-started,
                             memory_optimizer_steps=lock['outer_steps'],
                             query_optimizer_steps=lock['outer_steps']*lock['adversary_steps'] if mode=='decision' else 0))
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
    passed=all(row('decision')['max_abs_error'] <= .9*row(b)['max_abs_error'] and
               row('decision')['mean_abs_error'] <= row(b)['mean_abs_error'] for b in ('moment','random'))
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,all_support_rows_used=len(x),
                real_medical_fit_allocation=False,rows=rows,mode_costs=resource,trace=trace,
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
