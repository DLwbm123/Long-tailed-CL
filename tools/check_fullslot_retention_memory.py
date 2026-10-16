"""One frozen CPU factorial: retention scope x memory; original science dependencies immutable."""
import os
import time
import torch
import torch.nn.functional as F
from check_reserved_competition_memory import (DT, BETA, initial_model, clone_model, frame, scores, response, sliced, risk_gradient, assign, training_batch, initialization, moments_objective, project_ball, lowrank_fixture)
import check_reserved_competition_memory as reference
MODES=('old_moment','old_decision','full_moment','full_decision')


def centered(x):
    return x-x.mean(1,keepdim=True)


def train(model,current,old,q,slots,steps=48,full=False):
    start=time.monotonic();x,y,p=training_batch(current,old);params=clone_model(model,True)
    old_n=len(old);mask=y<old_n;coverage=[];movement=[]
    with torch.no_grad():
        teacher_all=scores(x[mask],model,q) if old_n else None
        correct=teacher_all[:,slots[:old_n]].argmax(1)==y[mask] if old_n else None
        indices=list(range(q.shape[1])) if full else slots[:old_n]
        teacher=centered(teacher_all[:,indices]) if old_n else None
        if old_n:
            for c in range(old_n):
                cm=y[mask]==c;cp=p[mask][cm]*len(slots)
                full_correct=teacher_all[cm].argmax(1)==slots[c]
                coverage.append(dict(class_index=c,weighted_old_correct=float((cp*correct[cm]).sum()),weighted_full_correct=float((cp*full_correct).sum()),effective_retention_mass=float((cp*correct[cm]).sum())/old_n,unweighted_old_correct=float(correct[cm].to(DT).mean())))
    for step in range(steps):
        all_s=scores(x,params,q);ss=all_s[:,slots]
        classification=-(ss.log_softmax(1)[torch.arange(len(x)),y]*p).sum();retain=classification*0
        if old_n and bool(correct.any()):
            now=centered(all_s[mask][:,indices])
            retain=(((now-teacher).square().mean(1)*correct)*p[mask]*len(slots)/old_n).sum()
        if old_n and step in (0,steps-1):
            with torch.no_grad():
                current=all_s[mask].detach();weights=p[mask]*len(slots)/old_n
                old_delta=centered(current[:,slots[:old_n]])-centered(teacher_all[:,slots[:old_n]])
                full_delta=centered(current)-centered(teacher_all)
                row=dict(preupdate_step=step,old_centered_MSE=float((weights*old_delta.square().mean(1)).sum()),full_centered_MSE=float((weights*full_delta.square().mean(1)).sum()),retention=float(retain.detach()),coverage_mean=sum(c['weighted_old_correct'] for c in coverage)/old_n)
                for name,other in [('new',slots[old_n:]),('reserved',[j for j in range(q.shape[1]) if j not in slots])]:
                    delta=(current[:,slots[:old_n]].mean(1)-current[:,other].mean(1))-(teacher_all[:,slots[:old_n]].mean(1)-teacher_all[:,other].mean(1)) if other else None
                    row['old_vs_'+name+'_mean_shift']=float((weights*delta).sum()) if other else None
                    row['old_vs_'+name+'_shift_MSE']=float((weights*delta.square()).sum()) if other else None
                movement.append(row)
        loss=classification+retain;gg=torch.autograd.grad(loss,params)
        with torch.no_grad():
            for a,b in zip(params,gg):
                a.sub_(b/16);assert bool(torch.isfinite(a).all())
    return clone_model(params),dict(seconds=time.monotonic()-start,updates=steps,retention_scope='full' if full else 'old',teacher_coverage=coverage,coverage_mean=sum(c['weighted_old_correct'] for c in coverage)/old_n if old_n else None,movement=movement,last_preupdate_classification=float(classification),last_preupdate_retention=float(retain),last_preupdate_gradient_norm=float(torch.cat([g.flatten() for g in gg]).norm()))


def unit_checks():
    g=torch.Generator().manual_seed(347);q=frame(12,4,347);model=initial_model(12,347)
    x=F.normalize(torch.randn(12,12,generator=g,dtype=DT),dim=1);old={0:(x[:3],torch.ones(3,dtype=DT)/3),1:(x[3:6],torch.ones(3,dtype=DT)/3)};current={2:x[6:]};slots=[0,1,2]
    a,info=train(model,current,old,q,slots,1,False);b,_=reference.train(model,current,old,q,slots,1)
    assert all(torch.equal(u,v) for u,v in zip(a,b))
    assert len(info['teacher_coverage'])==2 and all(0<=v['weighted_old_correct']<=1+1e-12 for v in info['teacher_coverage'])
    teacher=torch.tensor([[2.,1.,0.,-1.]],dtype=DT);student=torch.tensor([[1.,0.,0.,-1.]],dtype=DT)
    old_error=float((centered(student[:,:2])-centered(teacher[:,:2])).square().mean());full_error=float((centered(student)-centered(teacher)).square().mean())
    assert old_error==0 and full_error>0
    equal=float((centered(student+3)-centered(student)).abs().max());assert equal<1e-12
    w=student.clone().requires_grad_();loss=(centered(w)-centered(teacher)).square().mean();grad=torch.autograd.grad(loss,w)[0];eps=1e-6
    direction=torch.tensor([[.2,-.4,.1,.3]],dtype=DT)
    fun=lambda z:(centered(z)-centered(teacher)).square().mean()
    fd=abs(float((fun(student+eps*direction)-fun(student-eps*direction))/(2*eps)-(grad*direction).sum()));assert fd<1e-8
    return dict(status='PASS',old_training_reference_exact=True,teacher_coverage_checked=True,old_blind_shift_error=old_error,full_detected_shift_error=full_error,common_shift_invariance_error=equal,full_retention_FD_error=fd,synthetic_updates=2)


def compress(current, seed, mode, model_start=None, model_end=None, q=None, slots=None):
    start = time.monotonic()
    labels = sorted(current)
    n, w = initialization(current, seed)
    n.requires_grad_(); w.requires_grad_()
    opt = torch.optim.Adam([n, w], lr=.02)
    targets = []
    if mode == 'moment':
        means = torch.stack([current[c].mean(0) for c in labels])
        second = torch.stack([current[c].T @ current[c] / len(current[c]) for c in labels])
    else:
        models = [clone_model(m, True) for m in (model_start, model_end)]
        for c in labels:
            with torch.no_grad():
                resp = torch.cat([response(current[c], c, m, q, slots) for m in models], 1)
            gradients = [risk_gradient(current[c], c, m, q, slots)[1].detach() for m in models]
            targets.append((resp.detach(), gradients))
    first = None; components=[]
    target_norms=[dict(class_index=c,snapshot=j,gradient_norm_squared=float(g.square().sum()),denominator=float(g.square().sum().clamp_min(1e-12))) for c,(_,gs) in zip(labels,targets) for j,g in enumerate(gs)]
    for step in range(48):
        if mode == 'moment':
            objective = moments_objective(n, w, means, second)
        else:
            objective = n.sum() * 0; response_part=0.; gradient_parts=[0.,0.]
            for i, c in enumerate(labels):
                z, p = F.normalize(n[i], dim=1), w[i].softmax(0)
                response_now = torch.cat([response(z, c, m, q, slots) for m in models], 1)
                term = sliced(targets[i][0], response_now, p) / BETA
                response_part += float(term.detach()) / len(labels)
                for j,(m, reference) in enumerate(zip(models, targets[i][1])):
                    grad = risk_gradient(z, c, m, q, slots, p, True)[1]
                    part = .5 * (grad-reference).square().sum() / reference.square().sum().clamp_min(1e-12)
                    gradient_parts[j] += float(part.detach()) / len(labels)
                    term = term + part
                objective = objective + term / len(labels)
        if mode!='moment' and step in (0,47): components.append(dict(preupdate_step=step,response=response_part,gradient_pre=gradient_parts[0],gradient_post=gradient_parts[1],total=float(objective.detach())))
        assert bool(torch.isfinite(objective))
        if first is None:
            first = float(objective.detach())
        opt.zero_grad()
        n.grad, w.grad = torch.autograd.grad(objective, (n, w))
        opt.step()
        with torch.no_grad():
            n.copy_(project_ball(n)); w.clamp_(-8, 8)
            assert float(n.norm(dim=2).min()) > 1e-12
    memory = {c: (F.normalize(n[i].detach(), dim=1), w[i].detach().softmax(0)) for i, c in enumerate(labels)}
    return memory, dict(seconds=time.monotonic()-start, updates=48, method=mode,
                        objective_components=components,target_gradient_norms=target_norms,initial_objective=first, last_preupdate_objective=float(objective.detach()),
                        min_effective_nodes=min(float(1/p.square().sum()) for _, p in memory.values()))


def run_scale(cfg, lock, journal):
    begun=time.monotonic(); x,y,counts=lowrank_fixture(cfg); d,k=cfg['dimension'],cfg['classes']
    q=frame(d,k,cfg['seed']); q_copy=q.clone(); source_copy=x.clone()
    models={m:initial_model(d,cfg['seed']) for m in MODES}
    memories={m:{} for m in MODES}; slots={m:[] for m in MODES}
    snapshots=[]; training=[]; construction=[]; allocation=[]; seen=0
    for task,add in enumerate(cfg['task_sizes'],1):
        old_seen=seen; seen+=add; current={c:x[y==c] for c in range(old_seen,seen)}
        before_memory={m:{c:(z.clone(),p.clone()) for c,(z,p) in memories[m].items()} for m in MODES}
        starts={}
        for mode in MODES:
            journal(dict(phase='synthetic_nonlinear_train',task=task,mode=mode,completed_stages=len(snapshots)))
            tm=time.monotonic(); previous=slots[mode].copy()
            slots[mode]=assign(current,memories[mode],models[mode],q,previous,False)
            assert slots[mode][:old_seen]==previous
            allocation.append(dict(task=task,mode=mode,seconds=time.monotonic()-tm,old_slots_unchanged=True,
                                   nondefault_new_assignments=sum(a!=b for a,b in zip(slots[mode][old_seen:],range(old_seen,seen)))))
            starts[mode]=clone_model(models[mode])
            models[mode],cost=train(models[mode],current,memories[mode],q,slots[mode],full=mode.startswith('full'))
            training.append(dict(task=task,mode=mode,**cost))
        if task==1:
            assert all(all(torch.equal(a,b) for a,b in zip(models['old_moment'],models[m])) for m in MODES)
        journal(dict(phase='current_memory_compression',task=task))
        baseline,cost=compress(current,cfg['seed'],'moment')
        construction.append(dict(task=task,mode='shared_moment_construction',**cost))
        for mode in MODES:
            if mode.endswith('moment'):
                new=baseline
            else:
                journal(dict(phase='decision_memory_compression',task=task,mode=mode))
                new,cost=compress(current,cfg['seed'],'decision',starts[mode],models[mode],q,slots[mode])
                construction.append(dict(task=task,mode=mode,**cost))
            for c,(z,p) in before_memory[mode].items():
                assert torch.equal(z,memories[mode][c][0]) and torch.equal(p,memories[mode][c][1])
            memories[mode].update(new)
        assert torch.equal(x,source_copy) and torch.equal(q,q_copy)
        snapshots.append((task,old_seen,seen,{m:clone_model(models[m]) for m in MODES},{m:slots[m].copy() for m in MODES}))
    # Only now create independent synthetic evaluation; all four trajectories and stages are sealed.
    ex,ey,eval_counts=lowrank_fixture(dict(cfg,seed=cfg['seed']+10000))
    rows=[]; class_rows=[]; pairs=[]; diagnostic=[]; contributions=[]
    comparisons=[('old_moment','old_decision'),('full_moment','full_decision'),
                 ('old_moment','full_moment'),('old_decision','full_decision'),('old_moment','full_decision')]
    for task,old_seen,seen,mm,ss in snapshots:
        journal(dict(phase='sealed_synthetic_evaluation',task=task)); mask=ey<seen; xx,yy=ex[mask],ey[mask]; predictions={}
        with torch.no_grad():
            for mode in MODES:
                s=scores(xx,mm[mode],q)[:,ss[mode]]; pred=s.argmax(1); predictions[mode]=pred
                recalls=[]
                for c in range(seen):
                    cm=yy==c; recall=float((pred[cm]==c).to(DT).mean()); recalls.append(recall)
                    class_rows.append(dict(task=task,mode=mode,class_index=c,count=int(cm.sum()),recall=recall))
                truth=s[torch.arange(len(yy)),yy]; wrong=s.clone(); wrong[torch.arange(len(yy)),yy]=-torch.inf
                margins=truth-wrong.max(1).values
                rows.append(dict(task=task,mode=mode,seen_classes=seen,accuracy=float((pred==yy).to(DT).mean()),
                    balanced_accuracy=sum(recalls)/seen,old_balanced_accuracy=sum(recalls[:old_seen])/old_seen if old_seen else None,
                    new_balanced_accuracy=sum(recalls[old_seen:])/(seen-old_seen),margin_min=float(margins.min()),
                    margin_q10=float(torch.quantile(margins,.1)),margin_median=float(margins.median())))
            for base,can in comparisons:
                a,b=predictions[base],predictions[can]
                for group,gm in (('all',torch.ones_like(yy,dtype=torch.bool)),('old',yy<old_seen),('new',yy>=old_seen)):
                    ac,bc=a==yy,b==yy
                    pairs.append(dict(task=task,baseline=base,candidate=can,group=group,count=int(gm.sum()),
                        rescued=int((gm&~ac&bc).sum()),harmed=int((gm&ac&~bc).sum()),
                        wrong_to_different_wrong=int((gm&~ac&~bc&(a!=b)).sum()),
                        baseline_errors_to_old=int((gm&~ac&(a<old_seen)).sum()),baseline_errors_to_new=int((gm&~ac&(a>=old_seen)).sum()),
                        candidate_errors_to_old=int((gm&~bc&(b<old_seen)).sum()),candidate_errors_to_new=int((gm&~bc&(b>=old_seen)).sum())))
        by={r['mode']:r for r in rows if r['task']==task}
        keys=['balanced_accuracy','new_balanced_accuracy']+(['old_balanced_accuracy'] if old_seen else [])
        passes={base:all(by['full_decision'][j]>=by[base][j] for j in keys) for base in ('old_moment','full_moment','old_decision')}
        contributions.append(dict(task=task,no_harm_vs_controls=passes,
            D_minus_A_BA=by['full_decision']['balanced_accuracy']-by['old_moment']['balanced_accuracy'],
            C_minus_A_BA=by['old_decision']['balanced_accuracy']-by['old_moment']['balanced_accuracy'],
            D_minus_B_BA=by['full_decision']['balanced_accuracy']-by['full_moment']['balanced_accuracy'],
            B_minus_A_BA=by['full_moment']['balanced_accuracy']-by['old_moment']['balanced_accuracy'],
            D_minus_C_BA=by['full_decision']['balanced_accuracy']-by['old_decision']['balanced_accuracy'],
            interaction_BA=by['full_decision']['balanced_accuracy']-by['full_moment']['balanced_accuracy']-by['old_decision']['balanced_accuracy']+by['old_moment']['balanced_accuracy']))
        # Same-state real-vs-memory diagnostic uses old synthetic support only AFTER sealing.
        for mode in MODES:
            model=clone_model(mm[mode],True); losses=[]; errors=[]; gradients=[]
            for c in range(old_seen):
                real=x[y==c]; z,p=memories[mode][c]
                a,ag=risk_gradient(real,c,model,q,ss[mode]); b,bg=risk_gradient(z,c,model,q,ss[mode],p)
                losses.append(abs(float(a-b))); gradients.append(float((ag-bg).norm()))
                with torch.no_grad(): errors.append(float(sliced(response(real,c,model,q,ss[mode]),response(z,c,model,q,ss[mode]),p)))
            diagnostic.append(dict(task=task,mode=mode,old_classes=old_seen,
                mean_old_CE_error=sum(losses)/len(losses) if losses else None,max_old_CE_error=max(losses) if losses else None,
                max_old_gradient_error=max(gradients) if gradients else None,max_old_margin_sliced_W1=max(errors) if errors else None))
    full_signal=all(all(c['no_harm_vs_controls'].values()) for c in contributions)
    for control,field in (('old_moment','D_minus_A_BA'),('full_moment','D_minus_B_BA'),('old_decision','D_minus_C_BA')):
        full_signal &= any(c[field]>0 for c in contributions)
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,synthetic_evaluation_counts=eval_counts,
        rows=rows,class_rows=class_rows,paired_predictions=pairs,training=training,construction=construction,
        assignment_costs=allocation,diagnostics=diagnostic,contributions=contributions,
        joint_mechanism_signal_pass=bool(full_signal),all_training_sealed_before_evaluation=True,
        old_memory_unchanged=True,stable_input_and_frame_exact=True,old_slot_ids_unchanged=True,
        no_old_support_during_candidate_training=True,candidate_memory_bytes=k*16*(d+1)*8,
        adapter_parameters=2*d*8,frame_bytes=d*k*8,total_adapter_updates=4*48*len(cfg['task_sizes']),
        actual_memory_updates=3*48*len(cfg['task_sizes']),shared_moment_construction_counted_once=True,
        CPU_pipeline_seconds=time.monotonic()-begun)


def run(cfg, lock, journal):
    torch.set_num_threads(lock['threads']); torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,
                  medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not result['GPU_initialized']
    return result
