"""Synthetic sequential memory audit. Oracle support is evaluation-only after its task."""
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from check_progressive_decision_memory import (
    DT, constrain_family, family, fixture, grouped, losses, memory_losses,
    moments_objective, project_ball,
)


def sphere(x):
    norm=x.norm(dim=-1,keepdim=True)
    assert bool((norm>1e-12).all()), 'Undefined sphere projection'
    return x/norm


def validate_memory(memory, expected_seen, dimension):
    assert memory['schema']=='sequence_sphere_v1'
    assert memory['frame']=='stable_synthetic_input_v1'
    assert memory['seen']==expected_seen and memory['dimension']==dimension
    for n,w in memory['modes'].values():
        assert n.shape==(expected_seen,16,dimension) and w.shape==(expected_seen,16)
        assert torch.isfinite(n).all() and torch.isfinite(w).all()


def construct_current(x,y,start,stop,cfg,lock):
    # The compressor receives only the current task, never the oracle or old support.
    assert set(y.tolist())==set(range(start,stop)) and len(x)==len(y)
    count=torch.bincount(y-start,minlength=stop-start)
    means=torch.stack([x[y==j].mean(0) for j in range(start,stop)])
    second=torch.stack([x[y==j].T@x[y==j]/count[j-start] for j in range(start,stop)])
    init=[]
    for j in range(start,stop):
        g=torch.Generator().manual_seed(cfg['seed']+1000+j)
        samples=x[y==j]
        init.append(samples[torch.randint(len(samples),(lock['nodes_per_class'],),generator=g)])
    init=torch.stack(init);zero=torch.zeros(len(init),lock['nodes_per_class'],dtype=DT)
    output={'random':(init.clone(),zero.clone())};costs=[]
    for mode in ('moment','sphere'):
        begun=time.monotonic();n=init.clone().requires_grad_();w=zero.clone().requires_grad_()
        opt=torch.optim.Adam([n,w],lr=lock['memory_lr'])
        for _ in range(lock['outer_steps']):
            opt.zero_grad();moments_objective(n,w,means,second).backward();opt.step()
            with torch.no_grad():
                n.copy_(sphere(n) if mode=='sphere' else project_ball(n));w.clamp_(-8,8)
        output[mode]=(n.detach(),w.detach())
        costs.append(dict(mode=mode,wall_seconds=time.monotonic()-begun,steps=lock['outer_steps']))
    return output,costs


def evaluate(oracle_x,oracle_y,memories,old,cfg,lock):
    seen=next(iter(memories.values()))[0].shape[0]
    assert int(oracle_y.max())==seen-1
    params=[family(cfg['dimension'],lock['adapter_rank'],lock['queries'],cfg['seed']+100+i) for i in range(4)]
    for i in range(2):
        p=[v.clone().requires_grad_() for v in params[i]]
        opt=torch.optim.Adam(p,lr=lock['query_lr'])
        for _ in range(lock['evaluation_adversary_steps']):
            opt.zero_grad();target=grouped(losses(oracle_x,p),oracle_y,seen)
            objective=-torch.stack([(target-memory_losses(n,w,p)).square().mean() for n,w in memories.values()]).mean()
            objective.backward();opt.step();constrain_family(p)
        params.append([v.detach() for v in p])
    rows=[]
    with torch.no_grad():
        targets=[grouped(losses(oracle_x,p),oracle_y,seen) for p in params]
        for mode,(n,w) in memories.items():
            err=torch.cat([(target-memory_losses(n,w,p)).abs() for target,p in zip(targets,params)],1)
            r=dict(mode=mode,mean_abs_error=float(err.mean()),max_abs_error=float(err.max()),
                random_max=float(err[:,:32].max()),stress_max=float(err[:,32:].max()),
                old_mean_abs_error=float(err[:old].mean()) if old else None,
                old_max_abs_error=float(err[:old].max()) if old else None,
                new_mean_abs_error=float(err[old:].mean()),new_max_abs_error=float(err[old:].max()),
                logical_memory_bytes=8*(n.numel()+w.numel()),
                per_class_mean=[float(a) for a in err.mean(1)],per_class_max=[float(a) for a in err.max(1).values],
                second_trace=[float(a) for a in (n.square().sum(2)*w.softmax(1)).sum(1)],
                min_effective_nodes=float((1/w.softmax(1).square().sum(1)).min()))
            rows.append(r)
    by={r['mode']:r for r in rows}
    passed=all(by['sphere']['max_abs_error']<=.9*by[b]['max_abs_error'] and
               by['sphere']['mean_abs_error']<=by[b]['mean_abs_error'] for b in ('moment','random'))
    return rows,passed


def geometry_error(basis,probe,old,seen):
    def score(k):
        alpha=math.sqrt(k/(k-1))
        e=alpha*(basis[:,:k]-basis[:,:k].mean(1,keepdim=True))
        return probe@e/alpha
    if not old:return 0.
    a,b=score(old),score(seen)[:,:old]
    return float(((a-a[:,:1])-(b-b[:,:1])).abs().max())


def run_scale(cfg,lock,journal):
    begun=time.monotonic();oracle_x,oracle_y,counts=fixture(cfg)
    g=torch.Generator().manual_seed(cfg['seed']+2000)
    basis=torch.linalg.qr(torch.randn(cfg['dimension'],cfg['classes'],dtype=DT,generator=g)).Q
    probe=F.normalize(torch.randn(19,cfg['dimension'],dtype=DT,generator=g),dim=1)
    memories={};stages=[];old=0
    for task,added in enumerate(cfg['task_sizes'],1):
        seen=old+added;journal(dict(phase='current_task_compression',task=task,seen=seen,completed_stages=len(stages),completed_stage_results=stages))
        previous={k:(n.clone(),w.clone()) for k,(n,w) in memories.items()}
        mask=(oracle_y>=old)&(oracle_y<seen)
        current_x,current_y=oracle_x[mask].clone(),oracle_y[mask].clone()
        local,costs=construct_current(current_x,current_y,old,seen,cfg,lock)
        current_rows=len(current_x);del current_x,current_y
        for mode,(n,w) in local.items():
            memories[mode]=(torch.cat([memories[mode][0],n]),torch.cat([memories[mode][1],w])) if old else (n,w)
        assert all(torch.equal(memories[k][0][:old],n) and torch.equal(memories[k][1][:old],w) for k,(n,w) in previous.items())
        fixed_query_error=0.
        with torch.no_grad():
            for k,(n,w) in previous.items():
                for i in range(4):
                    p=family(cfg['dimension'],lock['adapter_rank'],lock['queries'],cfg['seed']+100+i)
                    fixed_query_error=max(fixed_query_error,float((memory_losses(n,w,p)-memory_losses(memories[k][0][:old],memories[k][1][:old],p)).abs().max()))
        assert fixed_query_error==0.
        geo=geometry_error(basis,probe,old,seen);assert geo<1e-12
        checkpoint=dict(schema='sequence_sphere_v1',frame='stable_synthetic_input_v1',seen=seen,
                        dimension=cfg['dimension'],source_commit=os.environ['Q283_SHA'],task=task,modes=memories)
        validate_memory(checkpoint,seen,cfg['dimension'])
        path=Path(os.environ['Q283_ROOT'])/(cfg['id']+'_checkpoint.private.pt')
        torch.save(checkpoint,path);loaded=torch.load(path,map_location='cpu',weights_only=True)
        validate_memory(loaded,seen,cfg['dimension']);assert loaded['source_commit']==os.environ['Q283_SHA'] and loaded['task']==task
        assert all(torch.equal(memories[k][i],loaded['modes'][k][i]) for k in memories for i in (0,1))
        journal(dict(phase='evaluation_only_oracle',task=task,seen=seen,completed_stages=len(stages),completed_stage_results=stages))
        evaluation_start=time.monotonic();mask=oracle_y<seen
        rows,passed=evaluate(oracle_x[mask],oracle_y[mask],memories,old,cfg,lock)
        stages.append(dict(task=task,old=old,seen=seen,current_support_rows=current_rows,
            construction_old_support_rows=0,old_memory_exact=True,old_fixed_query_error=fixed_query_error,
            rescaled_ETF_old_logit_error=geo,checkpoint_exact=True,checkpoint_bytes=path.stat().st_size,
            construction_costs=costs,rows=rows,finite_query_screen_pass=passed,
            evaluation_seconds=time.monotonic()-evaluation_start))
        old=seen
        journal(dict(phase='stage_complete',task=task,seen=seen,completed_stages=len(stages),completed_stage_results=stages))
    assert old==cfg['classes']
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,stages=stages,
        finite_query_screen_pass=all(s['finite_query_screen_pass'] for s in stages),
        oracle_is_synthetic_evaluation_only=True,recompression_performed=False,
        classification_training_performed=False,CPU_pipeline_seconds=time.monotonic()-begun)


def unit_checks():
    n=F.normalize(torch.arange(1.,97.,dtype=DT).reshape(2,16,3),dim=2);w=torch.zeros(2,16,dtype=DT)
    m=dict(schema='sequence_sphere_v1',frame='stable_synthetic_input_v1',seen=2,dimension=3,modes={'sphere':(n,w)})
    validate_memory(m,2,3);rejected=0
    for key,value in [('schema','bad'),('frame','bad'),('seen',3),('dimension',4)]:
        bad=dict(m);bad[key]=value
        try:validate_memory(bad,2,3)
        except AssertionError:rejected+=1
        else:raise AssertionError('Invalid memory accepted')
    assert rejected==4 and float((sphere(n).norm(dim=2)-1).abs().max())<1e-12
    try:sphere(torch.zeros(2,3,dtype=DT))
    except AssertionError:pass
    else:raise AssertionError('Zero direction accepted')
    try:construct_current(n[:,0],torch.tensor([0,2]),1,3,{}, {})
    except AssertionError:pass
    else:raise AssertionError('Old support accepted by current-task compressor')
    g=torch.Generator().manual_seed(283);basis=torch.linalg.qr(torch.randn(32,8,dtype=DT,generator=g)).Q
    probe=torch.randn(11,32,dtype=DT,generator=g);err=geometry_error(basis,probe,4,8);assert err<1e-12
    return dict(status='PASS',invalid_schema_frame_seen_dimension_rejected=rejected,
                zero_direction_rejected=True,old_support_rejected=True,progressive_fixed_input_error=err)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,
        medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not result['GPU_initialized']
    return result
