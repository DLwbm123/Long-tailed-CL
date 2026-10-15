"""Fixed-frame local loss jets; explicit radius, no claim for arbitrary future adapters."""
import math
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import DT,fixture,moments_objective,project_ball

BETA=8.
SLOTS=14
THIRD=BETA**3/(6*math.sqrt(3))


def oracle(x,v):
    t=BETA*(x@v.T)
    return F.softplus(t).mean(0), BETA*(t.sigmoid().T@x)/len(x)


def fit(x,anchor):
    assert x.ndim==2 and bool((x.norm(dim=1)<=1+1e-12).all())
    t=BETA*(x@anchor);s=t.sigmoid()
    loss=F.softplus(t).mean();grad=BETA*(s[:,None]*x).mean(0)
    h=(x.T@(x*(BETA**2*s*(1-s))[:,None]))/len(x);h=(h+h.T)/2
    ev,u=torch.linalg.eigh(h);assert float(ev.min())>=-1e-10
    ev=ev.clamp_min(0);k=min(SLOTS,x.shape[1]);vectors=torch.zeros(x.shape[1],SLOTS,dtype=DT);values=torch.zeros(SLOTS,dtype=DT)
    vectors[:,:k]=u[:,-k:];values[:k]=ev[-k:]
    omitted=ev[-k-1] if len(ev)>k else torch.zeros((),dtype=DT)
    state=(anchor.clone(),loss,grad,vectors,values,omitted)
    assert sum(a.numel() for a in state)==16*x.shape[1]+16
    return state,dict(retained_rank=k,omitted_spectral_norm=float(omitted),hessian_trace=float(ev.sum()),retained_trace=float(values.sum()))


def predict(state,v):
    anchor,l,g,u,ev,res=state;delta=v-anchor;r=delta.norm(dim=1)
    assert bool((r<=1/BETA+1e-12).all()), 'Query outside frozen local radius'
    proj=delta@u;value=l+delta@g+.5*(proj.square()*ev).sum(1)
    grad=g+(proj*ev)@u.T
    value_bound=.5*res*r.square()+THIRD*r.pow(3)/6
    gradient_bound=res*r+THIRD*r.square()/2
    return value,grad,value_bound,gradient_bound


def queries(anchor,seed):
    d=len(anchor);gen=torch.Generator().manual_seed(seed);z=torch.randn(47,d,generator=gen,dtype=DT)
    e=anchor/2;z=z-(z@e)[:,None]*e
    assert bool((z.norm(dim=1)>1e-12).all());z=F.normalize(z,dim=1)
    # Chord length 1/beta on the original ||v||=2 sphere, not a tuned radius grid.
    angle=2*math.asin((1/BETA)/4)
    return torch.cat((anchor[None],2*(math.cos(angle)*e+math.sin(angle)*z)),0)


def unit_checks():
    gen=torch.Generator().manual_seed(323);x=F.normalize(torch.randn(11,6,generator=gen,dtype=DT),dim=1);anchor=2*F.normalize(x.mean(0),dim=0)
    state,_=fit(x,anchor);v=queries(anchor,324);true,grad=oracle(x,v);value,g,vb,gb=predict(state,v)
    assert float((value-true).abs().sub(vb).max())<1e-10
    assert float((g-grad).norm(dim=1).sub(gb).max())<1e-10
    assert abs(float(value[0]-true[0]))<1e-12 and float((g[0]-grad[0]).abs().max())<1e-12
    eps=1e-6;direction=F.normalize(torch.randn(6,generator=gen,dtype=DT),dim=0)
    a=anchor.detach().requires_grad_();f=F.softplus(BETA*(x@a)).mean();ag=torch.autograd.grad(f,a,create_graph=True)[0];hv=torch.autograd.grad(ag@direction,a)[0]
    hdir=(state[3]*state[4])@(state[3].T@direction)
    fd=abs(float((oracle(x,(anchor+eps*direction)[None])[0]-oracle(x,(anchor-eps*direction)[None])[0])/(2*eps)-state[2]@direction))
    he=float((hv-hdir).abs().max());assert fd<1e-7 and he<1e-10
    shuffled,_=fit(x.flip(0),anchor);val2,grad2,_,_=predict(shuffled,v)
    assert float((value-val2).abs().max())<1e-10 and float((g-grad2).abs().max())<1e-10
    rejected=False
    try:predict(state,(anchor+.25*direction)[None])
    except AssertionError:rejected=True
    assert rejected
    # A positive omitted spectrum must enter both bounds, even with an exact low-rank eigensolve.
    tail_state,_=fit(torch.eye(20,dtype=DT),torch.zeros(20,dtype=DT))
    assert abs(float(tail_state[-1])-.8)<1e-12
    tv=torch.full((1,20),1/(BETA*math.sqrt(20)),dtype=DT)
    tp,tg,tb,tgb=predict(tail_state,tv);truth,truthg=oracle(torch.eye(20,dtype=DT),tv)
    assert float((tp-truth).abs().max())<=float(tb.max())+1e-12
    assert float((tg-truthg).norm())<=float(tgb.max())+1e-12
    assert THIRD/BETA**3==1/(6*math.sqrt(3))
    return dict(status='PASS',anchor_loss_gradient_exact=True,loss_gradient_fd_error=fd,hessian_vector_error=he,permutation_prediction_pass=True,outside_radius_rejected=True,optimizer_updates=0)


def run_scale(cfg,lock,journal):
    start=time.monotonic();x,y,counts=fixture(cfg);c,d=cfg['classes'],cfg['dimension'];m=16
    means=torch.stack([x[y==j].mean(0) for j in range(c)]);second=torch.stack([x[y==j].T@x[y==j]/counts[j] for j in range(c)])
    gen=torch.Generator().manual_seed(cfg['seed']+1);init=torch.stack([x[y==j][torch.randint(counts[j],(m,),generator=gen)] for j in range(c)])
    n=init.clone().requires_grad_();w=torch.zeros(c,m,dtype=DT,requires_grad=True);opt=torch.optim.Adam([n,w],lr=.02);beg=time.monotonic()
    initial=float(moments_objective(n,w,means,second).detach())
    for step in range(48):
        opt.zero_grad();moments_objective(n,w,means,second).backward();opt.step()
        with torch.no_grad():n.copy_(project_ball(n));w.clamp_(-8,8)
        if step%8==0:journal(dict(phase='unchanged_moment_baseline',step=step+1))
    base_seconds=time.monotonic()-beg;final=float(moments_objective(n,w,means,second).detach());n=n.detach();w=w.detach()
    rows=[];geometry=[];fit_seconds=0.;eval_seconds=0.;all_errors={a:[] for a in ('random','moment','linear','local_jet')};all_grad={a:[] for a in all_errors}
    max_value_violation=0.;max_grad_violation=0.;max_vb=0.;max_gb=0.
    for j in range(c):
        journal(dict(phase='class_loss_jet',class_index=j));a=x[y==j];assert float(means[j].norm())>1e-12
        anchor=2*F.normalize(means[j],dim=0);beg=time.monotonic();state,diag=fit(a,anchor);fit_seconds+=time.monotonic()-beg
        frozen=[t.clone() for t in state];beg=time.monotonic();v=queries(anchor,cfg['seed']+1000+j);target,tg=oracle(a,v);jet,jg,vb,gb=predict(state,v)
        value_violation=float(((target-jet).abs()-vb).max());grad_violation=float(((tg-jg).norm(dim=1)-gb).max())
        assert value_violation<1e-9 and grad_violation<1e-9
        max_value_violation=max(max_value_violation,value_violation);max_grad_violation=max(max_grad_violation,grad_violation);max_vb=max(max_vb,float(vb.max()));max_gb=max(max_gb,float(gb.max()))
        predictions={'linear':(state[1]+(v-anchor)@state[2],state[2].expand_as(v)),'local_jet':(jet,jg)}
        for name,z,p in (('random',init[j],torch.full((m,),1/m,dtype=DT)),('moment',n[j],w[j].softmax(0))):
            z=F.normalize(z,dim=1);t=BETA*(z@v.T);predictions[name]=((p[:,None]*F.softplus(t)).sum(0),BETA*(t.sigmoid().T@(p[:,None]*z)))
        for name,(pred,pg) in predictions.items():
            e=(pred-target).abs();ge=(pg-tg).norm(dim=1);all_errors[name].append(e);all_grad[name].append(ge)
            rows.append(dict(class_index=j,count=counts[j],mode=name,mean_abs_error=float(e.mean()),max_abs_error=float(e.max()),mean_gradient_l2_error=float(ge.mean()),max_gradient_l2_error=float(ge.max()),anchor_abs_error=float(e[0]),anchor_gradient_error=float(ge[0])))
        geometry.append(dict(class_index=j,**diag,max_loss_bound=float(vb.max()),max_gradient_bound=float(gb.max()),mean_oracle_loss=float(target.mean())))
        assert all(torch.equal(a,b) for a,b in zip(state,frozen));eval_seconds+=time.monotonic()-beg
    summary=[]
    for mode,e in all_errors.items():
        e=torch.stack(e);ge=torch.stack(all_grad[mode]);summary.append(dict(mode=mode,mean_abs_error=float(e.mean()),max_abs_error=float(e.max()),mean_gradient_l2_error=float(ge.mean()),max_gradient_l2_error=float(ge.max())))
    by={r['mode']:r for r in summary};screen=all(by['local_jet']['max_abs_error']<=.9*by[b]['max_abs_error'] and by['local_jet']['mean_abs_error']<=by[b]['mean_abs_error'] for b in ('random','moment'))
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,rows=summary,class_rows=rows,geometry=geometry,local_finite_query_screen_pass=screen,original_full_query_gate_pass=None,local_bound_checks_pass=True,max_value_bound_violation=max_value_violation,max_gradient_bound_violation=max_grad_violation,max_value_bound=max_vb,max_gradient_bound=max_gb,candidate_memory_bytes=c*(16*d+16)*8,baseline_memory_bytes=c*(16*d+16)*8,queries_per_class=48,query_radius=1/BETA,query_norm=2.,candidate_optimizer_updates=0,baseline_optimizer_updates=48,baseline_initial_objective=initial,baseline_final_objective=final,baseline_construction_seconds=base_seconds,jet_construction_seconds=fit_seconds,evaluation_seconds=eval_seconds,memory_unchanged=True,CPU_pipeline_seconds=time.monotonic()-start)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not result['GPU_initialized'];return result
