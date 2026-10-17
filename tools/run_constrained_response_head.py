"""Fixed-budget response CE optimized inside the old-H update span."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA, DT, features
from run_affine_pair_diagnostic import paired
from run_head_competition_diagnostic import competition, stats
from run_real_query_node import load, save
from query_node_weights import select
from run_linear_ce_head import objective, hessian_vector, newton_steps


def constrained_fit(z,y,p,base,q,penalty=.064):
    assert z.dtype==p.dtype==base.dtype==q.dtype==DT and z.shape[1]==base.shape[0]==q.shape[0] and y.shape==p.shape==(len(z),)
    assert all(bool(torch.isfinite(v).all()) for v in (z,p,base,q)) and bool((p>0).all())
    scale=float(z.shape[1]/(2*(p[:,None]*z.square()).sum()))
    reduced=z@q;gram=reduced.T@(p[:,None]*reduced)+.001*torch.eye(q.shape[1],dtype=DT)
    factor=torch.linalg.cholesky(gram);zt=torch.linalg.solve_triangular(factor,reduced.T,upper=False).T
    offset=BETA*z@base;v=torch.zeros((q.shape[1],base.shape[1]),dtype=DT,requires_grad=True);trace=[]
    chol_error=float((factor@factor.T-gram).abs().max());feature_error=float((zt@factor.T-reduced).abs().max())
    assert chol_error<1e-10 and feature_error<1e-10
    info=newton_steps(v,zt,y,p,penalty,0.,scale,q@factor,trace,offset)
    final=objective(v,zt,y,p,penalty,0.,scale,offset);gv=torch.autograd.grad(final,v)[0]
    delta=torch.linalg.solve_triangular(factor.T,v.detach(),upper=True);w=base+q@delta
    native=w.detach().requires_grad_();native_loss=objective(native,z,y,p,penalty,base,scale);full_g=torch.autograd.grad(native_loss,native)[0]
    tangent=q@(q.T@full_g);mapped=q@(factor@gv);gradient_error=float((tangent-mapped).abs().max());objective_error=abs(float(native_loss.detach())-float(final.detach()))
    assert gradient_error<1e-10 and objective_error<1e-10 and bool(torch.isfinite(w).all()) and float(final)<=trace[0]['objective']+1e-10
    change=z@(w-base);gaps=change.gather(1,y[:,None])-change;protection=float(scale*(p[:,None]*gaps.square()).sum())
    grad=float(tangent.abs().max());info.update(initial_objective=trace[0]['objective'],final_objective=float(final.detach()),initial_support_CE=trace[0]['objective'],final_support_CE=float(final.detach())-penalty*protection/2,final_protection_squared=protection,response_scale=scale,final_tangent_gradient_max_abs=grad,final_full_gradient_max_abs=float(full_g.abs().max()),final_optimizer_gradient_max_abs=float(gv.abs().max()),constrained_converged=grad<=1e-6,full_space_convergence_claim=False,closure_gradient_evaluations=len(trace),final_gradient_evaluations=2,chol_max_abs=chol_error,feature_max_abs=feature_error,gradient_chain_max_abs=gradient_error,objective_equivalence_max_abs=objective_error,trace=trace)
    return w,info


def unit():
    z=torch.tensor([[1.,0.],[1.,1.]],dtype=DT);q=torch.tensor([[1.],[0.]],dtype=DT);base=torch.tensor([[0.,0.],[.1,-.1]],dtype=DT)
    y=torch.tensor([0,1]);p=torch.full((2,),.5,dtype=DT);a=torch.tensor([[.2,-.2]],dtype=DT,requires_grad=True);scale=2/3;offset=BETA*z@base
    loss=objective(a,z@q,y,p,.064,0.,scale,offset);g=torch.autograd.grad(loss,a,create_graph=True)[0];direction=torch.tensor([[.3,-.1]],dtype=DT)
    exact=torch.autograd.grad((g*direction).sum(),a)[0];analytic=hessian_vector(direction,z@q,y,p,(BETA*(z@q)@a+offset).softmax(1),.064,scale)
    assert torch.allclose(exact,analytic,atol=1e-12,rtol=0)
    native=objective(base+q@a,z,y,p,.064,base,scale);assert torch.allclose(native,loss,atol=1e-12,rtol=0)
    zero=objective(a,z@q,y,p,.064,0.,scale);explicit=-(p*(BETA*(z@q)@a).log_softmax(1)[torch.arange(2),y]).sum()+.064*scale*(a[0,0]-a[0,1]).square()/2
    assert torch.allclose(zero,explicit,atol=1e-12,rtol=0)
    w,info=constrained_fit(z,y,p,base,q);lo,hi=-1.,1.
    import math
    for _ in range(80):
        mid=(lo+hi)/2;derivative=8*(1/(1+math.exp(-16*mid))-1+1/(1+math.exp(-16*(mid+.1))))+4*.064*scale*mid
        if derivative>0:hi=mid
        else:lo=mid
    expected=base+q@torch.tensor([[mid,-mid]],dtype=DT)
    assert info['constrained_converged'] and torch.allclose(w,expected,atol=1e-5,rtol=0) and torch.equal(w[1],base[1])
    assert info['combined_optimizer_evaluations']==info['closure_gradient_evaluations']+sum(info[k] for k in ('hessian_vector_evaluations','line_search_objective_evaluations','curvature_probability_evaluations'))<=300
    return dict(**info,explicit_HVP_checks=1,additional_autograd_calls=2)


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    case=next(v for v in lock['datasets'] if v['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['logit_scale']==BETA and lock['ce_head_penalty']==.064 and lock['regularization']==.001
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit_info=unit()
    prior=Path(cfg['prior_root']);bases={k:Path(cfg[k+'_root']) for k in ('centroid','ridge','projection')}
    names={'centroid':name+'.centroid.private.pt','ridge':name+'.ridge.private.pt','projection':name+'.projected.private.pt'}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name==name+'.S_new.features.private.pt' or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Query before basis or forbidden data'
        for k,base in bases.items():
            if p.is_relative_to(base):assert p.name=='DELIVERY.private.json' or (k!='projection' and p.name==names[k]) or (k=='centroid' and p.name==name+'.herding.private.pt') or (k=='projection' and sealed and p.name in {names[k],name+'_projection.json'}),'Frozen artifact boundary'
    sys.addaudithook(audit)
    for p in [prior/(name+'.'+v) for v in ('S_old.features.private.pt','teacher.private.pt','E.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt')]+[bases['projection']/(name+'.projected.private.pt'),bases['projection']/(name+'_projection.json')]:
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    objects={}
    for k,base in bases.items():
        rec=json.loads((base/'DELIVERY.private.json').read_text());assert rec['design_source_commit']==lock['inputs'][k]['source'] and rec['public_commit']==lock['inputs'][k]['delivery']
        if k=='projection':continue
        v=load(base/names[k]);assert v['frame'].shape==(1536,seen) and v['frame'].dtype==DT and bool(torch.isfinite(v['frame']).all()) and 'bias' not in v
        assert len(v['model'])==2 and all(a.ndim==2 and a.shape[0]==1536 and a.dtype==DT and bool(torch.isfinite(a).all()) and not bool(a.any()) for a in v['model']);objects[k]=v
    assert all(all(torch.equal(a,b) for a,b in zip(v['model'],objects['ridge']['model'])) for v in objects.values())
    bank=load(bases['centroid']/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    assert new['x'].dtype==DT and new['x'].ndim==2 and new['x'].shape[1]==1536 and bool(torch.isfinite(new['x']).all()) and new['y'].dtype==torch.int64 and bool(((new['y']>=old)&(new['y']<seen)).all())
    parts=[];counts=[];support_rows=[];classes=[];pairs=[];summaries=[];transitions=[];margins=[];competitors=[];gates={}
    def transform(x):return torch.cat([features(x[i:i+256],objects['ridge']['model']) for i in range(0,len(x),256)])
    with torch.no_grad():
        for c in range(seen):
            if c<old:
                x,p=bank[c];assert p.dtype==DT and p.shape==(len(x),) and len(x)>0 and torch.allclose(p,torch.full_like(p,1/len(x)),atol=1e-12,rtol=0)
            else:x=new['x'][new['y']==c]
            assert x.shape==(lock['support_counts'][name][c],1536) and x.dtype==DT and bool(torch.isfinite(x).all())
            parts.append(transform(x));counts.append(dict(dataset=name,class_index=c,group='old' if c<old else 'new',count=len(x)))
        h=torch.cat(parts[:old]);_,singular,vh=torch.linalg.svd(h,full_matrices=False)
        cutoff=max(h.shape)*torch.finfo(DT).eps*float(singular[0]);rank=int((singular>cutoff).sum());assert rank>0
        q=vh[:rank].T;orth=float((q.T@q-torch.eye(rank,dtype=DT)).abs().max());reconstruction=float((h-(h@q)@q.T).norm()/h.norm());assert orth<1e-10 and reconstruction<1e-10
        basis=dict(rank=rank,old_support_count=len(h),feature_dimension=1536,rank_cutoff=cutoff,largest_singular_value=float(singular[0]),smallest_retained_singular_value=float(singular[rank-1]),orthogonality_max_abs=orth,old_support_relative_residual=reconstruction)
        base=objects['ridge']['frame'];z=torch.cat(parts);y=torch.cat([torch.full((len(v),),c,dtype=torch.int64) for c,v in enumerate(parts)]);mass=torch.cat([torch.full((len(v),),1/(seen*len(v)),dtype=DT) for v in parts])
        assert abs(float(mass.sum())-1)<1e-12
        for c in range(seen):assert abs(float(mass[y==c].sum())-1/seen)<1e-12
        with torch.enable_grad():projected,fit_info=constrained_fit(z,y,mass,base,q)
        span_error=float(((projected-base)-q@(q.T@(projected-base))).abs().max())
        assert span_error<1e-10 and bool(torch.isfinite(projected).all())
        head_rows=[dict(dataset=name,class_index=c,head_norm=float(projected[:,c].norm()),change_from_ridge_norm=float((projected[:,c]-base[:,c]).norm())) for c in range(seen)]
        torch.save(dict(model=objects['ridge']['model'],frame=projected),root/(name+'.constrained.private.pt'))
        sealed=True;sealed_at=time.time();save(root/(job['id']+'.progress.json'),dict(phase='constrained_head_sealed',basis=basis,update_span_max_abs=span_error,constrained_converged=fit_info['constrained_converged'],final_tangent_gradient_max_abs=fit_info['final_tangent_gradient_max_abs'],combined_optimizer_evaluations=fit_info['combined_optimizer_evaluations']))
        ref=json.loads((bases['projection']/(name+'_projection.json')).read_text());assert ref['source_commit']==lock['inputs']['projection']['source']
        reference=load(bases['projection']/names['projection']);full=reference['frame'];assert full.shape==base.shape and full.dtype==DT and bool(torch.isfinite(full).all()) and 'bias' not in reference and len(reference['model'])==len(objects['ridge']['model']) and all(torch.equal(a,b) for a,b in zip(reference['model'],objects['ridge']['model']))
        heads={'raw_ridge':base,'frozen_projection':full,'constrained_response':projected};query_counts=[]
        for model,w in heads.items():
            for c,z in enumerate(parts):
                scores=BETA*z@w;ce=-scores.log_softmax(1)[:,c]
                support_rows.append(dict(**counts[c],model=model,correct=int((scores.argmax(1)==c).sum()),CE=float(ce.mean())))
        for row in support_rows:
            if row['model']=='constrained_response':continue
            a=next(a for a in ref['support_class_rows'] if a['class_index']==row['class_index'] and a['model']==('projected_response' if row['model']=='frozen_projection' else 'raw_ridge'))
            assert row['count']==a['count'] and row['correct']==a['correct'] and abs(row['CE']-a['CE'])<1e-10
        assert abs(sum(v['CE'] for v in support_rows if v['model']=='constrained_response')/seen-fit_info['final_support_CE'])<1e-10
        for query in ('Qfit','Qgate'):
            data=load(prior/(name+'.'+query+'.features.private.pt'));x=data['x'];y=data['y'];n=len(y)
            assert x.shape==(n,1536) and x.dtype==DT and bool(torch.isfinite(x).all()) and y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
            z=transform(x);scores={k:BETA*z@w for k,w in heads.items()};pred={k:v.argmax(1) for k,v in scores.items()};query_counts.append(dict(query=query,count=n))
            rivals={}
            for model,scores_m in scores.items():
                rival,margin,rank,ties=competition(scores_m,y);rivals[model]=rival;ce=-scores_m.log_softmax(1)[torch.arange(n),y]
                for c in range(seen):
                    mask=y==c;count=int(mask.sum());assert count>0;pr=pred[model][mask]
                    row=dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',count=count,correct=int((pr==c).sum()),recall=float((pr==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((pr!=c)&(pr<old)).sum()),errors_to_new=int(((pr!=c)&(pr>=old)).sum()));classes.append(row)
                    if model!='constrained_response':
                        a=next(a for a in ref['class_rows'] if a['query']==query and a['model']==('projected_response' if model=='frozen_projection' else model) and a['class_index']==c)
                        assert all(row[k]==a[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-a['CE'])<1e-10
                    margins.append(dict(dataset=name,query=query,model=model,class_index=c,count=count,margin=stats(margin[mask]),rank=stats(rank[mask]),top2=int((rank[mask]<=2).sum()),top3=int((rank[mask]<=3).sum()),true_score_tie_samples=int((ties[mask]>0).sum())))
                    for j in sorted(set(rival[mask].tolist())):
                        m=mask&(rival==j);competitors.append(dict(dataset=name,query=query,model=model,class_index=c,competitor=j,count=int(m.sum()),wrong_count=int((m&(pred[model]!=y)).sum())))
                for group in ('all','old','new'):
                    rr=[v for v in classes if v['query']==query and v['model']==model and (group=='all' or v['group']==group)]
                    summaries.append(dict(dataset=name,query=query,model=model,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
            for model in ('raw_ridge','frozen_projection'):
                for c in range(seen):
                    mask=y==c;p=paired(pred[model][mask],pred['constrained_response'][mask],y[mask]);pairs.append(dict(dataset=name,query=query,comparison=model+'__to__constrained_response',class_index=c,group='old' if c<old else 'new',**p))
            bc=pred['raw_ridge']==y;fc=pred['frozen_projection']==y;pc=pred['constrained_response']==y
            for c in range(seen):
                m=y==c;harm=m&bc&~fc;rescue=m&~bc&fc
                transitions.append(dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',count=int(m.sum()),ridge_to_candidate_rival_switches=int((m&(rivals['raw_ridge']!=rivals['constrained_response'])).sum()),reference_to_candidate_rival_switches=int((m&(rivals['frozen_projection']!=rivals['constrained_response'])).sum()),reference_harmed=int(harm.sum()),reference_harmed_repaired=int((harm&pc).sum()),reference_harmed_remaining=int((harm&~pc).sum()),reference_rescued=int(rescue.sum()),reference_rescues_retained=int((rescue&pc).sum()),reference_rescues_lost=int((rescue&~pc).sum()),previously_both_correct_harmed=int((m&bc&fc&~pc).sum()),previously_both_wrong_rescued=int((m&~bc&~fc&pc).sum())))
            get=lambda model:{v['group']:v['BA'] for v in summaries if v['query']==query and v['model']==model}
            gates[query]=select(get('raw_ridge'),get('constrained_response'))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,basis=basis,update_span_max_abs=span_error,fit=fit_info,unit_fit=unit_info,engineering_checks_passed=True,published_aggregates_matched=True,support_counts=counts,support_class_rows=support_rows,head_stats=head_rows,class_rows=classes,paired_rows=pairs,transition_rows=transitions,margin_rows=margins,competitor_rows=competitors,summaries=summaries,metric_gates=gates,primary_reference='raw_ridge',full_method_success_evaluated=False,method_success_claim=False,primary_both_queries_pass=fit_info['constrained_converged'] and all(g['selected']=='candidate' for g in gates.values()),convergence_scope='Original-W tangent projection Q Q^T grad, not full unconstrained gradient',reference_projection_not_fitted=True,query_counts=query_counts,formal_head_fits=1,engineering_head_fits=1,gradient_evaluations=fit_info['closure_gradient_evaluations']+2,hessian_vector_evaluations=fit_info['hessian_vector_evaluations'],engineering_autograd_calls=unit_info['closure_gradient_evaluations']+4,engineering_HVP_evaluations=unit_info['hessian_vector_evaluations']+1,formal_cholesky_factorizations=1,formal_triangular_solves=2,engineering_cholesky_factorizations=1,engineering_triangular_solves=2,formal_thin_SVDs=1,derived_head_builds=1,new_head_elements=projected.numel(),new_head_bytes=projected.numel()*projected.element_size(),support_feature_sample_forwards=sum(v['count'] for v in counts),support_head_logit_sample_evaluations=3*sum(v['count'] for v in counts),shared_query_feature_sample_forwards=sum(v['count'] for v in query_counts),query_head_logit_sample_evaluations=3*sum(v['count'] for v in query_counts),adapter_updates=0,encoder_updates=0,teacher_updates=0,memory_updates=0,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='One constrained response-CE fit from ridge with update in old-H span. Same original objective scale and bounded Newton-CG; no coefficient/rank search or historical continuation. Convergence refers only to original-W tangent projection. Exposed development queries, not method-success or deployment evidence.'))


if __name__=='__main__':main()
