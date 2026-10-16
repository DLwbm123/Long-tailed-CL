"""CPU synthetic sequential linear classification with a shared zero-anchor loss memory."""
import math
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import DT,fixture,moments_objective,project_ball

BETA=8.
RADIUS=1/BETA
MODES=('oracle_wide','oracle_local','moment_local','quadratic_local')


def exact(x,w,label,weights=None):
    p=torch.full((len(x),),1/len(x),dtype=DT) if weights is None else weights
    t=BETA*(x@w.T);value=(p[:,None]*F.softplus(t)).sum()-BETA*(p[:,None]*x).sum(0)@w[label]
    grad=BETA*(t.sigmoid().T@(p[:,None]*x));grad[label]-=BETA*(p[:,None]*x).sum(0)
    return value,grad


def quadratic(mu,u,ev,rho,w,label):
    assert bool((w.norm(dim=1)<=RADIUS+1e-12).all()), 'Shared head outside local domain'
    proj=w@u;value=len(w)*math.log(2)+(BETA/2)*(w@mu).sum()+(BETA**2/8)*(proj.square()*ev).sum()-BETA*w[label]@mu
    grad=(BETA/2)*mu.expand_as(w)+(BETA**2/4)*(proj*ev)@u.T;grad=grad.clone();grad[label]-=BETA*mu
    norms=w.norm(dim=1);loss_bound=((BETA**2/8)*rho*norms.square()+(BETA**4/192)*norms.pow(4)).sum()
    gradient_bounds=(BETA**2/4)*rho*norms+(BETA**4/48)*norms.pow(3)
    return value,grad,loss_bound,gradient_bounds


def compress_current(current,seed):
    labels=sorted(current);d=current[labels[0]].shape[1];means=torch.stack([current[j].mean(0) for j in labels]);second=torch.stack([current[j].T@current[j]/len(current[j]) for j in labels])
    init=[]
    for j in labels:
        g=torch.Generator().manual_seed(seed+1000+j);init.append(current[j][torch.randint(len(current[j]),(16,),generator=g)])
    n=torch.stack(init).requires_grad_();p=torch.zeros(len(labels),16,dtype=DT,requires_grad=True);opt=torch.optim.Adam([n,p],lr=.02);beg=time.monotonic()
    for _ in range(48):
        opt.zero_grad();moments_objective(n,p,means,second).backward();opt.step()
        with torch.no_grad():n.copy_(project_ball(n));p.clamp_(-8,8)
    baseline_seconds=time.monotonic()-beg;beg=time.monotonic();memory={};geometry=[]
    for i,j in enumerate(labels):
        ev,u=torch.linalg.eigh(second[i]);assert float(ev.min())>=-1e-10;ev=ev.clamp_min(0);k=min(15,d)
        basis=torch.zeros(d,15,dtype=DT);values=torch.zeros(15,dtype=DT);basis[:,:k]=u[:,-k:];values[:k]=ev[-k:];rho=ev[-k-1].clone() if d>k else torch.zeros((),dtype=DT)
        state=(means[i].clone(),basis,values,rho);assert sum(v.numel() for v in state)==16*d+16
        memory[j]=dict(moment=(F.normalize(n[i].detach(),dim=1),p[i].detach().softmax(0)),quadratic=state)
        geometry.append(dict(class_index=j,count=len(current[j]),omitted_second_eigenvalue=float(rho),retained_second_trace=float(values.sum()),second_trace=float(second[i].trace())))
    return memory,geometry,dict(baseline_seconds=baseline_seconds,spectral_seconds=time.monotonic()-beg,baseline_memory_updates=48,candidate_memory_updates=0)


def train_head(w,current,old,mode,steps,oracle=None):
    assert mode in MODES
    assert (oracle is not None)==mode.startswith('oracle')
    begun=time.monotonic();seen=len(w);start=w.clone();radius=2. if mode=='oracle_wide' else RADIUS
    for _ in range(steps):
        gradient=torch.zeros_like(w)
        if oracle is not None:
            for j,x in oracle.items():gradient+=exact(x,w,j)[1]/seen
        else:
            assert set(current).isdisjoint(old) and set(current)|set(old)==set(range(seen))
            for j,x in current.items():gradient+=exact(x,w,j)[1]/seen
            for j,m in old.items():
                if mode=='moment_local':z,p=m['moment'];gradient+=exact(z,w,j,p)[1]/seen
                else:gradient+=quadratic(*m['quadratic'],w,j)[1]/seen
        w=project_ball(w-gradient/16,radius)
        assert bool(torch.isfinite(w).all()) and float(w.norm(dim=1).max())<=radius+1e-12
    return w,dict(training_seconds=time.monotonic()-begun,steps=steps,max_head_norm=float(w.norm(dim=1).max()),boundary_heads=int((w.norm(dim=1)>=radius-1e-10).sum()),head_change_norm=float((w-start).norm()))


def unit_checks():
    g=torch.Generator().manual_seed(331);x=F.normalize(torch.randn(19,6,generator=g,dtype=DT),dim=1);w=project_ball(torch.randn(3,6,generator=g,dtype=DT)*.03,RADIUS/2)
    q=x.T@x/len(x);ev,u=torch.linalg.eigh(q);ev=ev.clamp_min(0);mu=x.mean(0);rho=torch.zeros((),dtype=DT)
    val,grad,bound,gb=quadratic(mu,u,ev,rho,w,1);tv,tg=exact(x,w,1)
    assert abs(float(val-tv))<=float(bound)+1e-10 and bool(((grad-tg).norm(dim=1)<=gb+1e-10).all())
    param=w.clone().requires_grad_();t=BETA*x@param.T;loss=F.softplus(t).mean(0).sum()-BETA*(x@param[1]).mean();autograd=torch.autograd.grad(loss,param)[0]
    assert float((autograd-tg).abs().max())<1e-12
    direction=F.normalize(torch.randn(w.numel(),generator=g,dtype=DT),dim=0).reshape_as(w);eps=1e-6
    fd=abs(float((quadratic(mu,u,ev,rho,w+eps*direction,1)[0]-quadratic(mu,u,ev,rho,w-eps*direction,1)[0])/(2*eps)-(grad*direction).sum()));assert fd<1e-7
    current={0:x[:9],1:x[9:]};initial=torch.zeros(2,6,dtype=DT)
    a,_=train_head(initial,current,{},'oracle_local',3,oracle=current);b,_=train_head(initial,current,{},'quadratic_local',3)
    assert torch.equal(a,b)
    reject=False
    try:quadratic(mu,u,ev,rho,torch.ones_like(w),1)
    except AssertionError:reject=True
    assert reject
    return dict(status='PASS',exact_risk_gradient_error=float((autograd-tg).abs().max()),quadratic_gradient_fd_error=fd,loss_and_gradient_bounds_pass=True,first_task_no_old_memory_exact=True,outside_domain_rejected=True,synthetic_head_updates=6)


def run_scale(cfg,lock,journal):
    begin=time.monotonic();x,y,counts=fixture(cfg);c,d=cfg['classes'],cfg['dimension'];memory={};heads={m:torch.zeros(0,d,dtype=DT) for m in MODES};snapshots=[];geometry=[];construction=[];seen=0;training=[]
    for task,add in enumerate(cfg['task_sizes'],1):
        old_seen=seen;seen+=add;current={j:x[y==j] for j in range(old_seen,seen)}
        frozen={j:[a.clone() for a in (*m['moment'],*m['quadratic'])] for j,m in memory.items()}
        for mode in MODES:
            journal(dict(phase='synthetic_sequence_train',task=task,mode=mode));initial=torch.cat((heads[mode],torch.zeros(add,d,dtype=DT)),0)
            oracle={j:x[y==j] for j in range(seen)} if mode.startswith('oracle') else None
            heads[mode],row=train_head(initial,current,memory,mode,48,oracle=oracle);training.append(dict(task=task,mode=mode,**row))
        for j,m in memory.items():assert all(torch.equal(a,b) for a,b in zip((*m['moment'],*m['quadratic']),frozen[j]))
        journal(dict(phase='current_task_compression',task=task));new,geo,cost=compress_current(current,cfg['seed']);memory.update(new);geometry+=geo;construction.append(dict(task=task,**cost))
        snapshots.append((task,old_seen,seen,{m:w.clone() for m,w in heads.items()}))
    assert seen==c
    # Evaluation oracle is isolated from candidate optimization; opened only after every arm/stage is sealed.
    evaluation_cfg=dict(cfg,seed=cfg['seed']+10000);ex,ey,eval_counts=fixture(evaluation_cfg);rows=[];class_rows=[];diagnostics=[];stages=[]
    for task,old_seen,seen,saved in snapshots:
        journal(dict(phase='sealed_synthetic_evaluation',task=task));mask=ey<seen;xx=ex[mask];yy=ey[mask]
        for mode,w in saved.items():
            pred=(xx@w.T).argmax(1);recalls=[]
            for j in range(seen):
                cm=yy==j;rec=float((pred[cm]==j).to(DT).mean());recalls.append(rec);class_rows.append(dict(task=task,mode=mode,class_index=j,count=int(cm.sum()),recall=rec))
            rows.append(dict(task=task,seen_classes=seen,mode=mode,accuracy=float((pred==yy).to(DT).mean()),balanced_accuracy=sum(recalls)/seen,old_balanced_accuracy=sum(recalls[:old_seen])/old_seen if old_seen else None,new_balanced_accuracy=sum(recalls[old_seen:])/(seen-old_seen)))
        w=saved['quadratic_local'];actual_loss=0.;surrogate_loss=0.;bound=0.;ag=torch.zeros_like(w);sg=torch.zeros_like(w);gb=torch.zeros(seen,dtype=DT)
        for j in range(old_seen):
            a,b=exact(x[y==j],w,j);q,g,bb,bg=quadratic(*memory[j]['quadratic'],w,j);actual_loss+=float(a)/seen;surrogate_loss+=float(q)/seen;ag+=b/seen;sg+=g/seen;bound+=float(bb)/seen;gb+=bg/seen
        observed=abs(actual_loss-surrogate_loss);grad_error=(ag-sg).norm(dim=1)
        assert observed<=bound+1e-8 and bool((grad_error<=gb+1e-8).all())
        diagnostics.append(dict(task=task,old_classes=old_seen,old_loss_absolute_error=observed,old_loss_bound=bound,max_old_gradient_error=float(grad_error.max()),max_old_gradient_bound=float(gb.max()),within_shared_domain=True,old_memory_unchanged=True))
        by={a['mode']:a for a in rows if a['task']==task};can=by['quadratic_local'];base=by['moment_local'];keys=('balanced_accuracy','new_balanced_accuracy')+(('old_balanced_accuracy',) if old_seen else ())
        stages.append(dict(task=task,no_harm_to_moment=all(can[k]>=base[k] for k in keys),strict_gain_over_moment=any(can[k]>base[k] for k in keys),candidate_minus_moment_BA=can['balanced_accuracy']-base['balanced_accuracy'],candidate_minus_local_oracle_BA=can['balanced_accuracy']-by['oracle_local']['balanced_accuracy'],local_minus_wide_oracle_BA=by['oracle_local']['balanced_accuracy']-by['oracle_wide']['balanced_accuracy']))
    signal=all(a['no_harm_to_moment'] for a in stages) and any(a['strict_gain_over_moment'] for a in stages)
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,synthetic_evaluation_counts=eval_counts,rows=rows,class_rows=class_rows,training=training,curvature=geometry,construction_costs=construction,bound_diagnostics=diagnostics,stage_contributions=stages,synthetic_sequence_signal_pass=signal,all_training_sealed_before_evaluation=True,candidate_old_support_reads_during_training=0,oracle_old_support_reads='SYNTHETIC_CONTROL_ONLY',candidate_memory_bytes=c*(16*d+16)*8,moment_memory_bytes=c*(16*d+16)*8,head_updates_per_arm=48*len(cfg['task_sizes']),total_head_updates=4*48*len(cfg['task_sizes']),baseline_compression_updates=48*len(cfg['task_sizes']),candidate_compression_updates=0,CPU_pipeline_seconds=time.monotonic()-begin)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    a.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not a['GPU_initialized'];return a
