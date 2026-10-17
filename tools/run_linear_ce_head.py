"""Fixed-budget class-balanced linear cross-entropy on legal retained support."""
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import prototype_analytic as analytic
from check_reserved_competition_memory import BETA, DT, features
from run_affine_pair_diagnostic import paired
from run_head_competition_diagnostic import competition, stats
from run_raw_ridge_head import REGULARIZATION
from run_real_query_node import load, save


def protection(w,z,y,p,center=0.,response_scale=None):
    delta=w-center
    if response_scale is None:return delta.square().sum()
    change=z@delta
    gaps=change.gather(1,y[:,None])-change
    return response_scale*(p[:,None]*gaps.square()).sum()


def objective(w,z,y,p,penalty=REGULARIZATION,center=0.,response_scale=None):
    ce=-(p*(BETA*z@w).log_softmax(1)[torch.arange(len(y)),y]).sum()
    return ce+penalty*protection(w,z,y,p,center,response_scale)/2


def hessian_vector(v,z,y,p,prob,penalty,response_scale):
    change=z@v
    out=BETA**2*z.T@(p[:,None]*prob*(change-(prob*change).sum(1,keepdim=True)))
    if response_scale is None:return out+penalty*v
    gap=change.gather(1,y[:,None])-change
    adj=-gap;adj.scatter_add_(1,y[:,None],gap.sum(1,keepdim=True))
    return out+penalty*response_scale*z.T@(p[:,None]*adj)


def newton_steps(w,z,y,p,penalty,center,response_scale,gradient_map,trace):
    counts=dict(hessian_vector_evaluations=0,line_search_objective_evaluations=0,curvature_probability_evaluations=0)
    work=lambda:len(trace)+sum(counts.values())
    reason='iteration_cap';iterations=0
    for _ in range(200):
        if work()>=300:reason='evaluation_cap';break
        loss=objective(w,z,y,p,penalty,center,response_scale);g=torch.autograd.grad(loss,w)[0].detach()
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(g).all())
        original_gradient=gradient_map@g if gradient_map is not None else g
        trace.append(dict(evaluation=len(trace)+1,objective=float(loss.detach()),gradient_max_abs=float(original_gradient.abs().max()),optimizer_gradient_max_abs=float(g.abs().max())))
        if float(g.abs().max())<=1e-8:reason='optimizer_gradient';break
        if work()>297:reason='evaluation_cap';break
        iterations+=1;counts['curvature_probability_evaluations']+=1
        with torch.no_grad():
            prob=(BETA*z@w).softmax(1);direction=torch.zeros_like(g);residual=-g.clone();cg_direction=residual.clone();rr=float(residual.square().sum())
            target=min(.5,math.sqrt(math.sqrt(rr)))*math.sqrt(rr)
            for _ in range(50):
                if work()>=299:break  # Reserve one objective evaluation for accepting a step.
                hd=hessian_vector(cg_direction,z,y,p,prob,penalty,response_scale);counts['hessian_vector_evaluations']+=1
                assert bool(torch.isfinite(hd).all())
                curvature=float((cg_direction*hd).sum())
                if curvature<=0:break
                alpha=rr/curvature;direction.add_(cg_direction,alpha=alpha);residual.add_(hd,alpha=-alpha)
                next_rr=float(residual.square().sum())
                if math.sqrt(next_rr)<=target:break
                cg_direction=residual+(next_rr/rr)*cg_direction;rr=next_rr
            slope=float((g*direction).sum())
            if slope>=0:direction=-g;slope=-float(g.square().sum())
            accepted=False;step=1.
            for _ in range(20):
                if work()>=300:break
                candidate=w+step*direction;trial=objective(candidate,z,y,p,penalty,center,response_scale);counts['line_search_objective_evaluations']+=1
                if bool(torch.isfinite(trial)) and float(trial)<=float(loss)+1e-4*step*slope:
                    w.copy_(candidate);accepted=True;break
                step*=.5
            if not accepted:reason='evaluation_cap' if work()>=300 else 'line_search_stop';break
    assert work()<=300
    return dict(optimizer_iterations=iterations,optimizer_stop_reason=reason,**counts,combined_optimizer_evaluations=work())


def fit(z,y,p,initial,penalty=REGULARIZATION,center=0.,response_scale=None,gradient_map=None,solver='lbfgs'):
    if gradient_map is not None:
        assert isinstance(center,torch.Tensor) and gradient_map.shape==(z.shape[1],z.shape[1]) and gradient_map.dtype==z.dtype and bool(torch.isfinite(gradient_map).all())
    assert response_scale is None or (math.isfinite(response_scale) and response_scale>0)
    if isinstance(center,torch.Tensor):
        assert center.shape==initial.shape and center.dtype==initial.dtype and bool(torch.isfinite(center).all()) and not center.requires_grad
    w=initial.detach().clone().requires_grad_();trace=[]
    opt=torch.optim.LBFGS([w],lr=1.,max_iter=200,max_eval=250,tolerance_grad=1e-8,tolerance_change=1e-14,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        assert len(trace)<300,'Gradient evaluation cap'
        opt.zero_grad();loss=objective(w,z,y,p,penalty,center,response_scale);assert bool(torch.isfinite(loss))
        loss.backward();assert bool(torch.isfinite(w.grad).all())
        original_gradient=gradient_map@w.grad if gradient_map is not None else w.grad
        trace.append(dict(evaluation=len(trace)+1,objective=float(loss.detach()),gradient_max_abs=float(original_gradient.abs().max()),optimizer_gradient_max_abs=float(w.grad.abs().max())))
        return loss
    assert solver in ('lbfgs','newton_cg')
    if solver=='newton_cg':solver_info=newton_steps(w,z,y,p,penalty,center,response_scale,gradient_map,trace)
    else:
        opt.step(closure);solver_info=dict(optimizer_iterations=int(opt.state[w]['n_iter']),optimizer_stop_reason='lbfgs_return',hessian_vector_evaluations=0,line_search_objective_evaluations=0,curvature_probability_evaluations=0,combined_optimizer_evaluations=len(trace))
    final=objective(w,z,y,p,penalty,center,response_scale);g=torch.autograd.grad(final,w)[0]
    end=float(final.detach());optimizer_grad=float(g.abs().max());original_gradient=gradient_map@g if gradient_map is not None else g;grad=float(original_gradient.abs().max());assert end<=trace[0]['objective']+1e-10 and bool(torch.isfinite(w).all())
    initial_protection=float(protection(initial,z,y,p,center,response_scale));final_protection=float(protection(w.detach(),z,y,p,center,response_scale))
    original_w=w.detach();original_initial=initial;original_center=center
    if gradient_map is not None:
        original_w=torch.linalg.solve_triangular(gradient_map.T,original_w,upper=True)
        original_initial=torch.linalg.solve_triangular(gradient_map.T,initial,upper=True)
        original_center=torch.linalg.solve_triangular(gradient_map.T,center,upper=True)
    return original_w,dict(initial_protection_squared=initial_protection,final_protection_squared=final_protection,initial_objective=trace[0]['objective'],final_objective=end,initial_support_CE=trace[0]['objective']-penalty*initial_protection/2,final_support_CE=end-penalty*final_protection/2,initial_head_norm=float(original_initial.norm()),final_head_norm=float(original_w.norm()),initial_center_distance=float((original_initial-original_center).norm()),final_center_distance=float((original_w-original_center).norm()),final_optimizer_gradient_max_abs=optimizer_grad,final_gradient_max_abs=grad,converged=grad<=1e-6,**solver_info,closure_gradient_evaluations=len(trace),final_gradient_evaluations=1,trace=trace)


def unit(penalty=REGULARIZATION,centered=False,response=False,preconditioned=False,solver='lbfgs'):
    a0=.1 if centered else 0.;center=torch.tensor([[a0,-a0],[-a0,a0]],dtype=DT)
    z=torch.eye(2,dtype=DT);y=torch.arange(2);p=torch.full((2,),.5,dtype=DT);w=torch.tensor([[.2,-.1],[.3,.4]],dtype=DT,requires_grad=True)
    scale=float(z.shape[1]/(2*(p[:,None]*z.square()).sum())) if response else None
    g=torch.autograd.grad(objective(w,z,y,p,penalty,center,scale),w,create_graph=solver=='newton_cg')[0]
    pg=w-center
    if response:
        pg=torch.zeros_like(w);explicit=w.new_zeros(());eye=torch.eye(2,dtype=DT)
        for i in range(len(y)):
            for j in range(2):
                direction=eye[y[i]]-eye[j];gap=z[i]@(w-center)@direction
                explicit=explicit+scale*p[i]*gap.square()
                pg=pg+scale*p[i]*gap*z[i,:,None]*direction[None,:]
        assert torch.allclose(protection(w,z,y,p,center,scale),explicit,atol=1e-12,rtol=0)
        shift=torch.tensor([[.7],[-.2]],dtype=DT)
        assert torch.allclose(objective(w+shift,z,y,p,penalty,center,scale),objective(w,z,y,p,penalty,center,scale),atol=1e-12,rtol=0)
        # Equal average curvature on the class-contrast subspace fixes scale without queries.
        trace=sum(float(protection(torch.nn.functional.one_hot(torch.tensor(i),4).to(DT).reshape(2,2),z,y,p,0.,scale)) for i in range(4))
        assert abs(trace-2*(2-1))<1e-12
        z3=torch.tensor([[1.,0.],[0.,1.],[.6,.8]],dtype=DT);y3=torch.arange(3);p3=torch.full((3,),1/3,dtype=DT)
        scale3=float(2/(2*(p3[:,None]*z3.square()).sum()));v3=torch.tensor([[.2,-.3,.1],[.5,.4,-.2]],dtype=DT)
        explicit3=sum(float(p3[i]*(z3[i]@(v3[:,int(y3[i])]-v3[:,j])).square()) for i in range(3) for j in range(3))*scale3
        assert abs(float(protection(v3,z3,y3,p3,0.,scale3))-explicit3)<1e-12
        trace3=sum(float(protection(torch.nn.functional.one_hot(torch.tensor(i),6).to(DT).reshape(2,3),z3,y3,p3,0.,scale3)) for i in range(6))
        assert abs(trace3-2*(3-1))<1e-12
    expected=BETA*z.T@(p[:,None]*((BETA*z@w).softmax(1)-torch.eye(2,dtype=DT)))+penalty*pg
    assert torch.allclose(g,expected,atol=1e-12,rtol=0)
    u=BETA*w
    logit_objective=-(p*(z@u).log_softmax(1)[torch.arange(len(y)),y]).sum()+penalty/BETA**2*protection(u,z,y,p,BETA*center,scale)/2
    assert torch.allclose(objective(w,z,y,p,penalty,center,scale),logit_objective,atol=1e-12,rtol=0)
    gu=z.T@(p[:,None]*((z@u).softmax(1)-torch.eye(2,dtype=DT)))+penalty/BETA**2*(BETA*pg)
    assert torch.allclose(g,BETA*gu,atol=1e-12,rtol=0)
    original=-(p*(BETA*z@w).log_softmax(1)[torch.arange(len(y)),y]).sum()+REGULARIZATION*w.square().sum()/2
    assert torch.allclose(objective(w,z,y,p),original,atol=1e-12,rtol=0)
    zd=z[[0,0,1]];yd=torch.tensor([0,0,1]);pd=torch.tensor([.25,.25,.5],dtype=DT)
    assert scale is None or abs(float(zd.shape[1]/(2*(pd[:,None]*zd.square()).sum()))-scale)<1e-12
    assert torch.allclose(objective(w,zd,yd,pd,penalty,center,scale),objective(w,z,y,p,penalty,center,scale),atol=1e-12,rtol=0)
    if preconditioned:
        factor=torch.tensor([[2.,0.],[.3,1.5]],dtype=DT);zt=torch.linalg.solve_triangular(factor,z.T,upper=False).T
        v=(factor.T@w.detach()).requires_grad_();ct=factor.T@center
        transformed=objective(v,zt,y,p,penalty,ct,scale)
        assert torch.allclose(transformed,objective(w,z,y,p,penalty,center,scale),atol=1e-12,rtol=0)
        if solver=='newton_cg':
            direction=torch.tensor([[.2,-.4],[.1,.5]],dtype=DT)
            exact=torch.autograd.grad((g*direction).sum(),w)[0]
            native=hessian_vector(direction,z,y,p,(BETA*z@w).softmax(1),penalty,scale)
            transformed_hv=factor@hessian_vector(factor.T@direction,zt,y,p,(BETA*zt@v).softmax(1),penalty,scale)
            assert torch.allclose(native,exact,atol=1e-12,rtol=0) and torch.allclose(transformed_hv,exact,atol=1e-12,rtol=0)
        else:
            gv=torch.autograd.grad(transformed,v)[0];assert torch.allclose(factor@gv,g,atol=1e-12,rtol=0)
        fitted,info=fit(zt,y,p,torch.zeros((2,2),dtype=DT),penalty,ct,scale,factor,solver)
    else:
        gd=torch.autograd.grad(objective(w,zd,yd,pd,penalty,center,scale),w)[0];assert torch.allclose(g,gd,atol=1e-12,rtol=0)
        fitted,info=fit(z,y,p,torch.zeros((2,2),dtype=DT),penalty,center,scale)
    lo,hi=0.,1.
    for _ in range(80):
        a=(lo+hi)/2
        if penalty*(a-a0)-BETA*.5/(1+math.exp(2*BETA*a))>0:hi=a
        else:lo=a
    expected=torch.tensor([[a,-a],[-a,a]],dtype=DT)
    assert info['converged'] and torch.allclose(fitted,expected,atol=1e-5,rtol=0)
    assert info['combined_optimizer_evaluations']==info['closure_gradient_evaluations']+sum(info[k] for k in ('hessian_vector_evaluations','line_search_objective_evaluations','curvature_probability_evaluations'))<=300
    return info


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['regularization']==REGULARIZATION and lock['logit_scale']==BETA
    coordinates=lock.get('ce_penalty_coordinates','head');assert coordinates in ('head','logits')
    penalty=REGULARIZATION*(BETA**2 if coordinates=='logits' else 1);assert penalty==lock.get('ce_head_penalty',REGULARIZATION)
    center_kind=lock.get('ce_penalty_center','zero');assert center_kind in ('zero','ridge')
    geometry=lock.get('ce_protection_geometry','parameter');assert geometry in ('parameter','response')
    response=geometry=='response';assert not response or center_kind=='ridge'
    solver_coordinates=lock.get('solver_coordinates','native');assert solver_coordinates in ('native','ridge_cholesky')
    preconditioned=solver_coordinates=='ridge_cholesky';assert not preconditioned or response
    solver=lock.get('solver','lbfgs');assert solver in ('lbfgs','newton_cg') and (solver=='lbfgs' or preconditioned)
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit_info=unit(penalty,center_kind=='ridge',response,preconditioned,solver)
    case=next(v for v in lock['datasets'] if v['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    prior=Path(cfg['prior_root']);cr=Path(cfg['centroid_root']);rr=Path(cfg['ridge_root']);ar=Path(cfg['affine_root'])
    par=Path(cfg['parameter_root']) if response else None
    allowed={cr:{'DELIVERY.private.json',name+'.centroid.private.pt',name+'.herding.private.pt'},rr:{'DELIVERY.private.json',name+'.ridge.private.pt'}}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name==name+'.S_new.features.private.pt' or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Fit/query boundary'
        for base,names in allowed.items():
            if p.is_relative_to(base):assert p.name in names,'Sealed source boundary'
        if p.is_relative_to(ar):assert p.name=='DELIVERY.private.json' or (sealed and p.name==name+'_affine.json'),'Reference before seal'
        if par is not None and p.is_relative_to(par):assert p.name=='DELIVERY.private.json' or (sealed and p.name in {name+'.ce.private.pt',name+'_ce.json'}),'Parameter reference before seal'
    sys.addaudithook(audit)
    for p in [prior/(name+'.'+s) for s in ('S_old.features.private.pt','teacher.private.pt','E.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt')]+[ar/(name+'_affine.json')]+([par/(name+'.ce.private.pt'),par/(name+'_ce.json')] if response else []):
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    for k,base in [('centroid',cr),('ridge',rr),('affine',ar)]+([('parameter',par)] if response else []):
        rec=json.loads((base/'DELIVERY.private.json').read_text());assert rec['design_source_commit']==lock['inputs'][k]['source'] and rec['public_commit']==lock['inputs'][k]['delivery']
    centroid=load(cr/(name+'.centroid.private.pt'));ridge=load(rr/(name+'.ridge.private.pt'));bank=load(cr/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    for obj in (centroid,ridge):
        assert obj['frame'].shape==(1536,seen) and obj['frame'].dtype==DT and bool(torch.isfinite(obj['frame']).all()) and 'bias' not in obj
        assert len(obj['model'])==2 and all(v.ndim==2 and v.shape[0]==1536 and v.dtype==DT and bool(torch.isfinite(v).all()) and not bool(v.any()) for v in obj['model'])
    assert all(torch.equal(a,b) for a,b in zip(centroid['model'],ridge['model']))
    assert new['x'].dtype==DT and new['x'].ndim==2 and new['x'].shape[1]==1536 and bool(torch.isfinite(new['x']).all()) and new['y'].dtype==torch.int64 and bool(((new['y']>=old)&(new['y']<seen)).all())
    zz=[];counts=[]
    with torch.no_grad():
        for c in range(seen):
            if c<old:
                x,p=bank[c];assert p.dtype==DT and p.shape==(len(x),) and len(x)>0
                assert torch.allclose(p,torch.full_like(p,1/len(x)),atol=1e-12,rtol=0)
            else:x=new['x'][new['y']==c]
            n=len(x);assert n==lock['support_counts'][name][c] and x.shape==(n,1536) and x.dtype==DT and bool(torch.isfinite(x).all())
            z=torch.cat([features(x[i:i+256],centroid['model']) for i in range(0,n,256)]).numpy();zz.append(z)
            counts.append(dict(dataset=name,class_index=c,group='old' if c<old else 'new',count=n))
    def moments(parts):
        return analytic.append(analytic.empty(1536),np.concatenate(parts),np.concatenate([np.full(len(z),c,dtype=np.int64) for c,z in enumerate(parts)]),list(range(seen)))
    full=moments(zz);mu=full['mu'];norm=np.linalg.norm(mu,axis=1);assert np.all(norm>1e-12)
    mean_error=float(np.max(np.abs((mu/norm[:,None]).T-centroid['frame'].numpy())))
    G=full['S']/seen;B=mu.T/seen;normal_error=float(np.linalg.norm((G+REGULARIZATION*np.eye(1536))@ridge['frame'].numpy()-B)/max(np.linalg.norm(B),1e-15))
    assert mean_error<1e-12 and normal_error<=1e-7,'Full support/reference mismatch'
    z=torch.from_numpy(np.concatenate(zz));y=torch.cat([torch.full((len(v),),c,dtype=torch.int64) for c,v in enumerate(zz)])
    mass=torch.cat([torch.full((len(v),),1/(seen*len(v)),dtype=DT) for v in zz]);assert abs(float(mass.sum())-1)<1e-12
    for c in range(seen):assert abs(float(mass[y==c].sum())-1/seen)<1e-12
    center=ridge['frame'] if center_kind=='ridge' else 0.
    response_scale=float(z.shape[1]/(2*(mass[:,None]*z.square()).sum())) if response else None
    conditioning={}
    if preconditioned:
        metric=torch.from_numpy(G)+REGULARIZATION*torch.eye(1536,dtype=DT);factor=torch.linalg.cholesky(metric)
        zt=torch.linalg.solve_triangular(factor,z.T,upper=False).T;initial=factor.T@ridge['frame'];ct=factor.T@center
        conditioning=dict(cholesky_relative_residual=float((factor@factor.T-metric).norm()/metric.norm()),feature_transform_relative_residual=float((zt@factor.T-z).norm()/z.norm()),ridge_floor=REGULARIZATION)
        before=float(objective(initial,zt,y,mass,penalty,ct,response_scale));native_before=float(objective(ridge['frame'],z,y,mass,penalty,center,response_scale))
        assert conditioning['cholesky_relative_residual']<1e-10 and conditioning['feature_transform_relative_residual']<1e-10 and abs(before-native_before)<1e-10
        w,fit_info=fit(zt,y,mass,initial,penalty,ct,response_scale,factor,solver)
        native_after=float(objective(w,z,y,mass,penalty,center,response_scale));conditioning['objective_equivalence_max_abs']=max(abs(before-native_before),abs(native_after-fit_info['final_objective']))
        assert conditioning['objective_equivalence_max_abs']<1e-10
    else:w,fit_info=fit(z,y,mass,ridge['frame'],penalty,center,response_scale)
    head_stats=[dict(dataset=name,class_index=c,head_norm=float(w[:,c].norm()),head_change_from_ridge_norm=float((w[:,c]-ridge['frame'][:,c]).norm())) for c in range(seen)]
    torch.save(dict(model=centroid['model'],frame=w),root/(name+'.ce.private.pt'))
    sealed=True;sealed_at=time.time();save(root/(job['id']+'.progress.json'),dict(phase='linear_ce_head_sealed',fit={k:v for k,v in fit_info.items() if k!='trace'}))
    ref=json.loads((ar/(name+'_affine.json')).read_text());classes=[];pairs=[];margin_rows=[];competitors=[];queries=[];summaries=[];gates={}
    heads={'raw_centroid':centroid['frame'],'raw_ridge':ridge['frame'],'linear_ce':w}
    parameter_ref=None;response_rows=[]
    if response:
        parameter=load(par/(name+'.ce.private.pt'));parameter_ref=json.loads((par/(name+'_ce.json')).read_text())
        assert parameter_ref['source_commit']==lock['inputs']['parameter']['source'] and parameter_ref['fit']['converged']
        assert parameter['frame'].shape==w.shape and parameter['frame'].dtype==DT and bool(torch.isfinite(parameter['frame']).all()) and 'bias' not in parameter
        assert len(parameter['model'])==2 and all(torch.equal(a,b) for a,b in zip(parameter['model'],centroid['model']))
        heads={'raw_ridge':ridge['frame'],'parameter_ce':parameter['frame'],'linear_ce':w}
        with torch.no_grad():
            for model in ('parameter_ce','linear_ce'):
                change=z@(heads[model]-center);gap=change.gather(1,y[:,None])-change
                for c in range(seen):
                    mask=y==c;raw=float(gap[mask].square().sum(1).mean()/(seen-1))
                    response_rows.append(dict(dataset=name,model=model,class_index=c,group='old' if c<old else 'new',count=int(mask.sum()),raw_pair_response_squared=raw,normalized_response_squared=raw*(seen-1)*response_scale))
                average=sum(v['normalized_response_squared'] for v in response_rows if v['model']==model)/seen
                assert abs(average-float(protection(heads[model],z,y,mass,center,response_scale)))<1e-8
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));x=data['x'];y=data['y'];n=len(y)
        assert x.shape==(n,1536) and x.dtype==DT and bool(torch.isfinite(x).all()) and y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        with torch.no_grad():zq=torch.cat([features(x[i:i+256],centroid['model']) for i in range(0,n,256)]);scores={k:BETA*zq@h for k,h in heads.items()}
        pred={k:s.argmax(1) for k,s in scores.items()};queries.append(dict(query=query,count=n))
        for model,s in scores.items():
            pr=pred[model];rv,margin,rank,ties=competition(s,y);ce=-s.log_softmax(1)[torch.arange(n),y]
            for c in range(seen):
                mask=y==c;count=int(mask.sum());assert count>0;prc=pr[mask]
                row=dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',count=count,correct=int((prc==c).sum()),recall=float((prc==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((prc!=c)&(prc<old)).sum()),errors_to_new=int(((prc!=c)&(prc>=old)).sum()));classes.append(row)
                if model!='linear_ce':
                    reference=parameter_ref if model=='parameter_ce' else ref;reference_model='linear_ce' if model=='parameter_ce' else model
                    r=next(v for v in reference['class_rows'] if v['query']==query and v['model']==reference_model and v['class_index']==c)
                    assert all(row[t]==r[t] for t in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-r['CE'])<1e-10
                margin_rows.append(dict(dataset=name,query=query,model=model,class_index=c,count=count,margin=stats(margin[mask]),rank=stats(rank[mask]),top2=int((rank[mask]<=2).sum()),top3=int((rank[mask]<=3).sum()),true_score_tie_samples=int((ties[mask]>0).sum())))
                for j in sorted(set(rv[mask].tolist())):
                    v=mask&(rv==j);competitors.append(dict(dataset=name,query=query,model=model,class_index=c,competitor=j,count=int(v.sum()),wrong_count=int((v&(pr!=y)).sum())))
            for group in ('all','old','new'):
                rr=[v for v in classes if v['query']==query and v['model']==model and (group=='all' or v['group']==group)]
                summaries.append(dict(dataset=name,query=query,model=model,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
        for model in (k for k in heads if k!='linear_ce'):
            for c in range(seen):
                m=y==c;p=paired(pred[model][m],pred['linear_ce'][m],y[m]);pairs.append(dict(dataset=name,query=query,comparison=model+'__to__linear_ce',class_index=c,group='old' if c<old else 'new',**p))
        get=lambda m:{v['group']:v['BA'] for v in summaries if v['query']==query and v['model']==m}
        from query_node_weights import select
        gates[query]=select(get('raw_ridge'),get('linear_ce'))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,solver=solver,engineering_explicit_hessian_vector_checks=2*int(solver=='newton_cg'),solver_coordinates=solver_coordinates,conditioning=conditioning,formal_cholesky_factorizations=int(preconditioned),formal_triangular_solves=4*int(preconditioned),ce_protection_geometry=geometry,response_scale=response_scale,support_response_rows=response_rows,support_response_diagnostic_sample_evaluations=2*sum(v['count'] for v in counts) if response else 0,ce_penalty_center=center_kind,ce_penalty_coordinates=coordinates,ce_head_penalty=penalty,ce_logit_penalty=penalty/BETA**2,fit=fit_info,unit_fit=unit_info,class_rows=classes,summaries=summaries,gates=gates,primary_both_queries_pass=fit_info['converged'] and all(v['selected']=='candidate' for v in gates.values()),primary_reference='raw_ridge',paired_rows=pairs,margin_rows=margin_rows,competitor_rows=competitors,query_counts=queries,support_counts=counts,head_stats=head_stats,full_centroid_max_abs_reconstruction_error=mean_error,full_ridge_normal_relative_residual=normal_error,published_aggregates_matched=True,engineering_checks_passed=True,support_feature_sample_forwards=sum(v['count'] for v in counts),shared_query_feature_sample_forwards=sum(v['count'] for v in queries),model_logit_sample_evaluations=3*sum(v['count'] for v in queries),formal_head_fits=1,engineering_head_fits=1,production_linear_solves=4*int(preconditioned),full_reference_head_fits=0,full_support_moment_reconstructions=1,head_elements=w.numel(),head_bytes=w.numel()*w.element_size(),formal_gradient_evaluations=fit_info['closure_gradient_evaluations']+1,engineering_gradient_evaluations=unit_info['closure_gradient_evaluations']+3,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Standard linear class-balanced CE baseline on the same legal H/new-S and same feature/parameter class. Fixed beta8; head/logit penalty, zero/ridge center and parameter/response geometry and optimizer coordinates explicitly recorded, no bias or column normalization; loss scale interacts with regularization. No optimal-hyperparameter, full-history, compression-loss, novel-method or independent-validation claim.'))


if __name__=='__main__':main()
