"""Accepted-point L-BFGS with a fixed, bounded Newton-CG tail; same CE objective."""
import json,time
from pathlib import Path
import numpy as np
import torch

class HeadBudget(Exception):pass
class StageBudget(Exception):pass

class SafeLBFGS(torch.optim.LBFGS):
    @torch.no_grad()
    def _directional_evaluate(self,closure,x,t,d):
        try:
            self._add_grad(t,d)
            loss=float(closure());gradient=self._gather_flat_grad()
            return loss,gradient
        finally:self._set_param(x)

def write(path,value):
    if path is None:return
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    def clean(v):
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        return v
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False));temp.replace(path)

def loss_grad(w,x,y,p,lam):
    s=x@w;prob=s.softmax(1);ce=(p*(s.logsumexp(1)-s[torch.arange(len(y),device=y.device),y])).sum()
    loss=ce+lam/2*w[:-1].square().sum()
    hot=torch.nn.functional.one_hot(y,w.shape[1]).to(x.dtype)
    grad=x.T@((prob-hot)*p[:,None]);grad[:-1]+=lam*w[:-1]
    return loss,grad

def hessian_vector(x,p,prob,v,lam):
    a=x@v;u=prob*(a-(prob*a).sum(1,keepdim=True))
    answer=x.T@(p[:,None]*u);answer[:-1]+=lam*v[:-1]
    return answer

def project_bias(v):
    v=v.clone();v[-1]-=v[-1].mean();return v

def objective_delta(w,d,alpha,x,y,p,prob,scores,lam):
    a=alpha*(x@d);small=a.abs().amax(1)<=.5;normal=(scores+a).logsumexp(1)-scores.logsumexp(1)
    # log1p/expm1 avoids subtracting nearly equal total objectives in the tail.
    safe_a=torch.where(small[:,None],a,torch.zeros_like(a))
    stable=torch.log1p((prob*torch.expm1(safe_a)).sum(1));logratio=torch.where(small,stable,normal)
    ce=(p*(logratio-a[torch.arange(len(y),device=y.device),y])).sum()
    return ce+lam*(alpha*(w[:-1]*d[:-1]).sum()+alpha*alpha/2*d[:-1].square().sum())

def _core(x,y,p,k,lam,device,check,limit,lbfgs_limit=1000):
    start=time.monotonic();x=x.to(device);y=y.to(device);p=p.to(device)
    diagonal=.5*(x.square()*p[:,None]).sum(0);diagonal[:-1]+=lam
    scale=diagonal.sqrt().clamp_min(1e-12)[:,None]
    theta=torch.zeros((x.shape[1],k),dtype=torch.float64,device=device,requires_grad=True)
    optimizer=SafeLBFGS([theta],lr=1,max_iter=1,max_eval=1500,tolerance_grad=1e-9,tolerance_change=1e-14,history_size=20,line_search_fn='strong_wolfe')
    trace=[];kinds=[];accepted_trace=[];newton_log=[];counts=dict(value_gradient=0,HVP=0,objective_difference=0,curvature_prepare=0)
    lb_calls=0;lb_iters=0;newton_steps=0;phase='LBFGS';reason='';accepted_w=None;accepted_theta=theta.detach().clone();accepted_loss=None;accepted_gradient=None
    def count(kind):
        if time.monotonic()-start>limit:raise HeadBudget('HEAD_WALL_LIMIT')
        if sum(counts.values())>=3000:raise HeadBudget('TOTAL_OPERATOR_LIMIT')
        check();counts[kind]+=1
    def evaluate(w,kind):
        nonlocal lb_calls
        if phase=='LBFGS' and lb_calls>=1500:raise StageBudget('LBFGS_EVALUATION_ALLOCATION')
        count('value_gradient')
        if phase=='LBFGS':lb_calls+=1
        value,gradient=loss_grad(w,x,y,p,lam)
        assert bool(torch.isfinite(value)) and bool(torch.isfinite(gradient).all()),'nonfinite objective/gradient'
        trace.append((float(value),float(gradient.abs().max()),float((gradient/scale).abs().max()),lb_iters+newton_steps));kinds.append(kind)
        return value.detach(),gradient.detach()
    def checkpoint(w,value,gradient,kind):
        nonlocal accepted_w,accepted_loss,accepted_gradient,accepted_theta
        accepted_w=w.detach().clone();accepted_loss=value.detach().clone();accepted_gradient=gradient.detach().clone()
        if phase=='LBFGS':accepted_theta=theta.detach().clone()
        accepted_trace.append(dict(phase=phase,kind=kind,iteration=lb_iters+newton_steps,objective=float(value),raw_gradient_max=float(gradient.abs().max()),operator_evaluations=sum(counts.values())))
    def certified():
        return accepted_gradient is not None and float(accepted_gradient.abs().max())<=1e-7 and float(accepted_loss)<=initial+1e-10
    def closure():
        optimizer.zero_grad();value,gradient=evaluate(theta.detach()/scale,'trial_or_initial');theta.grad=gradient/scale;return value
    with torch.no_grad():value,gradient=evaluate(theta.detach()/scale,'initial_accepted');checkpoint(theta.detach()/scale,value,gradient,'initial')
    initial=float(value);phase_transition=None
    try:
        while lb_iters<lbfgs_limit and not certified():
            before=theta.detach().clone();previous_loss=float(accepted_loss);old_iters=int(optimizer.state[theta].get('n_iter',0));lb_iters+=1
            try:
                optimizer.step(closure)
                with torch.no_grad():value,gradient=evaluate(theta.detach()/scale,'accepted_certificate')
            except StageBudget as error:
                with torch.no_grad():theta.copy_(before)
                phase_transition=str(error);break
            except BaseException:
                with torch.no_grad():theta.copy_(before)
                raise
            with torch.no_grad():checkpoint(theta.detach()/scale,value,gradient,'accepted_after_complete_step')
            if certified():break
            if int(optimizer.state[theta].get('n_iter',0))==old_iters or bool(torch.equal(before,theta.detach())):
                phase_transition='INTERNAL_STOP_WITHOUT_MOVEMENT';break
            if abs(float(value)-previous_loss)<1e-14:
                phase_transition='INTERNAL_OBJECTIVE_STAGNATION';break
        if not certified():
            phase_transition=phase_transition or 'LBFGS_ITERATION_ALLOCATION';phase='NEWTON_CG'
            w=project_bias(accepted_w)
            with torch.no_grad():value,gradient=evaluate(w,'gauge_certificate');checkpoint(w,value,gradient,'bias_gauge_only')
            for step in range(5):
                if certified():break
                newton_steps+=1;w=accepted_w;g=accepted_gradient;gmax=float(g.abs().max());count('curvature_prepare');scores=x@w;prob=scores.softmax(1)
                hv_before=counts['HVP'];cg_iterations=0
                def hv(v):
                    if counts['HVP']>=1000:raise HeadBudget('HVP_ALLOCATION_LIMIT')
                    count('HVP');result=project_bias(hessian_vector(x,p,prob,v,lam));assert bool(torch.isfinite(result).all());return result
                direction=torch.zeros_like(w);residual=project_bias(-g);z=project_bias(residual/diagonal[:,None]);q=z.clone();rz=(residual*z).sum();cg_ok=False;bad_curvature=False
                for j in range(199):
                    cg_iterations=j+1;hq=hv(q);den=(q*hq).sum()
                    if not bool(torch.isfinite(den)) or float(den)<=0:bad_curvature=True;break
                    alpha=rz/den;direction=project_bias(direction+alpha*q);residual=project_bias(residual-alpha*hq)
                    if float(residual.abs().max())<=.1*gmax:cg_ok=True;break
                    z=project_bias(residual/diagonal[:,None]);new_rz=(residual*z).sum();q=project_bias(z+(new_rz/rz)*q);rz=new_rz
                log=dict(step=step+1,cg_iterations=cg_iterations,HVP_count=counts['HVP']-hv_before,raw_gradient_before=gmax)
                if bad_curvature:reason='UNRESOLVED_CG_CURVATURE';newton_log.append(log);break
                if not cg_ok:reason='UNRESOLVED_CG_ITERATION_LIMIT';newton_log.append(log);break
                actual=float((hv(direction)+g).abs().max());log.update(HVP_count=counts['HVP']-hv_before,linear_residual_max=actual)
                if actual>.1*gmax:reason='UNRESOLVED_CG_RESIDUAL';newton_log.append(log);break
                gtd=float((g*direction).sum())
                if not np.isfinite(gtd) or gtd>=0:reason='UNRESOLVED_NEWTON_DIRECTION';newton_log.append(log);break
                found=False
                for j in range(20):
                    alpha=2.**(-j);count('objective_difference');delta=float(objective_delta(w,direction,alpha,x,y,p,prob,scores,lam))
                    if not np.isfinite(delta) or delta>1e-4*alpha*gtd:continue
                    candidate=project_bias(w+alpha*direction);value,gradient=evaluate(candidate,'newton_trial')
                    if float(value)>float(accepted_loss)+1e-10:continue
                    checkpoint(candidate,value,gradient,'accepted_newton');log.update(line_search_halvings=j,step_size=alpha,objective_delta=delta,raw_gradient_after=float(gradient.abs().max()));found=True;break
                newton_log.append(log)
                if not found:reason='UNRESOLVED_NEWTON_LINE_SEARCH';break
            if not reason and not certified():reason='UNRESOLVED_NEWTON_STEP_LIMIT'
    except HeadBudget as error:
        reason=str(error)
        with torch.no_grad():theta.copy_(accepted_theta)
    finally:
        with torch.no_grad():theta.copy_(accepted_theta)
    success=certified() and reason not in ['HEAD_WALL_LIMIT','TOTAL_OPERATOR_LIMIT','HVP_ALLOCATION_LIMIT']
    if success:reason='RAW_W_ACCEPTED_CERTIFICATE'
    info=dict(status='CONVERGED' if success else 'UNRESOLVED',stop_reason=reason or 'UNRESOLVED',lambda_value=lam,iterations=lb_iters+newton_steps,lbfgs_iterations=lb_iters,newton_steps=newton_steps,closure_evaluations=counts['value_gradient'],operator_evaluations=sum(counts.values()),operator_counts=counts,initial_objective=initial,final_objective=float(accepted_loss),gradient_max_abs=float(accepted_gradient.abs().max()),theta_gradient_max_abs=float((accepted_gradient/scale).abs().max()),wall_seconds=time.monotonic()-start,optimizer_trace_columns=['objective','raw_W_gradient_max','theta_gradient_max','outer_iteration'],optimizer_trace=trace,trace_point_kinds=kinds,accepted_trace=accepted_trace,newton_log=newton_log,phase_transition=phase_transition,point_kind='accepted',trial_parameters_always_restored=True,torch_version=torch.__version__,reused=False)
    return accepted_w.cpu(),info

def fit(x,y,p,k,lam,device,check,limit=60,receipt=None):
    tick=time.monotonic();record=dict(state='STARTED',classes=k,lambda_value=lam,point_kind='accepted',per_head_limit_seconds=limit)
    write(receipt,record)
    try:
        w,info=_core(x,y,p,k,lam,device,check,limit);record.update(info,state='FINISHED');return w,info
    except BaseException as error:
        record.update(state='FAILED',error=type(error).__name__)
        tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code is _core.__code__:
                v=tb.tb_frame.f_locals
                record.update(optimizer_trace=v.get('trace',[]),trace_point_kinds=v.get('kinds',[]),accepted_trace=v.get('accepted_trace',[]),operator_counts=v.get('counts',{}),newton_log=v.get('newton_log',[]))
            tb=tb.tb_next
        raise
    finally:record['receipt_wall_seconds']=time.monotonic()-tick;write(receipt,record)

def selfcheck(receipt=None):
    tick=time.monotonic();result=dict(state='STARTED',real_data_access=False)
    write(receipt,result)
    try:
        rng=torch.Generator().manual_seed(74149);x=torch.randn(18,5,generator=rng,dtype=torch.float64);x[:,-1]=1;y=torch.arange(18)%3;p=torch.full((18,),1/18,dtype=x.dtype);w=torch.randn(5,3,generator=rng,dtype=x.dtype,requires_grad=True);lam=.01
        s=x@w;reference=(p*(-s.log_softmax(1)[torch.arange(18),y])).sum()+lam/2*w[:-1].square().sum();value,g=loss_grad(w.detach(),x,y,p,lam)
        assert torch.allclose(value,reference,atol=1e-12) and torch.allclose(g,torch.autograd.grad(reference,w)[0],atol=1e-12);result['analytic_gradient']=True
        direction=project_bias(torch.randn(w.shape,generator=rng,dtype=x.dtype));prob=(x@w.detach()).softmax(1);actual=hessian_vector(x,p,prob,direction,lam)
        def fun(v):return loss_grad(v,x,y,p,lam)[1]
        _,autograd_hvp=torch.autograd.functional.jvp(fun,w.detach(),direction)
        error=float((actual-autograd_hvp).abs().max());assert error<1e-10;result['HVP_autograd_error']=error
        epsilon=1e-5;fd=(fun(w.detach()+epsilon*direction)-fun(w.detach()-epsilon*direction))/(2*epsilon);assert float((fd-actual).abs().max())<1e-8;result['HVP_finite_difference']=True
        shifted=w.detach().clone();shifted[-1]+=3;assert abs(float(loss_grad(shifted,x,y,p,lam)[0]-value))<1e-12;result['bias_gauge_equivalence']=True
        for alpha in [1.,1e-5]:
            delta=objective_delta(w.detach(),direction,alpha,x,y,p,prob,x@w.detach(),lam);direct=loss_grad(w.detach()+alpha*direction,x,y,p,lam)[0]-value;assert abs(float(delta-direct))<1e-12
        result['stable_objective_difference']=True
        raw=torch.tensor([2e-7,5e-8],dtype=x.dtype);scale=torch.tensor([1000.,.01],dtype=x.dtype);assert raw[0]>1e-7 and (raw/scale)[0]<1e-9 and raw[1]<1e-7 and (raw/scale)[1]>1e-9;result['raw_vs_scaled_certificate_counterexample']=True
        theta=torch.zeros((5,3),dtype=x.dtype,requires_grad=True);opt=SafeLBFGS([theta],max_iter=1,line_search_fn='strong_wolfe');before=theta.detach().clone()
        def failure():raise HeadBudget('synthetic trial interruption')
        try:opt._directional_evaluate(failure,opt._clone_param(),1.,torch.ones(theta.numel(),dtype=x.dtype))
        except HeadBudget:pass
        assert torch.equal(theta.detach(),before);result['trial_exception_restores_exact_parameters']=True
        f,info=fit(x,y,p,3,lam,'cpu',lambda:None,10,receipt.with_name('SELFCHECK_FIT.json') if receipt else None);result['fit']=info;write(receipt,result)
        assert info['status']=='CONVERGED' and info['point_kind']=='accepted' and info['gradient_max_abs']<=1e-7
        # Validate a Newton direction independently of the L-BFGS path.
        z=project_bias(w.detach());gz=loss_grad(z,x,y,p,lam)[1];n=z.numel();basis=torch.eye(n,dtype=x.dtype);H=torch.stack([hessian_vector(x,p,(x@z).softmax(1),basis[i].reshape_as(z),lam).flatten() for i in range(n)],dim=1)
        d=torch.linalg.pinv(H,hermitian=True)@(-gz.flatten());d=project_bias(d.reshape_as(z));assert float((hessian_vector(x,p,(x@z).softmax(1),d,lam)+gz).abs().max())<1e-9;result['Newton_gauge_linear_system']=True
        xx=x.clone();xx[:,:-1]*=.1
        nf,ni=_core(xx,y,p,3,1.,'cpu',lambda:None,10,lbfgs_limit=0)
        result['newton_only_fit']=ni;write(receipt,result)
        assert ni['status']=='CONVERGED' and ni['newton_steps']>0 and ni['newton_log'] and ni['gradient_max_abs']<=1e-7
        result['Newton_CG_actual_tail_certified']=True
        calls=[0]
        def interrupt_trial():
            calls[0]+=1
            if calls[0]==3:raise HeadBudget('TOTAL_OPERATOR_LIMIT')
        restored,stopped=fit(x,y,p,3,lam,'cpu',interrupt_trial,10,receipt.with_name('INTERRUPTED_FIT.json') if receipt else None)
        assert stopped['status']=='UNRESOLVED' and stopped['point_kind']=='accepted' and torch.equal(restored,torch.zeros_like(restored))
        result['mid_trial_budget_returns_last_accepted']=True
        result.update(state='PASSED',raw_gradient_certificate=True)
        return result
    except BaseException as error:result.update(state='FAILED',error=type(error).__name__);raise
    finally:result['wall_seconds']=time.monotonic()-tick;write(receipt,result)
