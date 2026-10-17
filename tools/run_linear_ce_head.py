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


def objective(w,z,y,p,penalty=REGULARIZATION):
    ce=-(p*(BETA*z@w).log_softmax(1)[torch.arange(len(y)),y]).sum()
    return ce+penalty*w.square().sum()/2


def fit(z,y,p,initial,penalty=REGULARIZATION):
    w=initial.detach().clone().requires_grad_();trace=[]
    opt=torch.optim.LBFGS([w],lr=1.,max_iter=200,max_eval=250,tolerance_grad=1e-8,tolerance_change=1e-14,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        assert len(trace)<300,'Gradient evaluation cap'
        opt.zero_grad();loss=objective(w,z,y,p,penalty);assert bool(torch.isfinite(loss))
        loss.backward();assert bool(torch.isfinite(w.grad).all())
        trace.append(dict(evaluation=len(trace)+1,objective=float(loss.detach()),gradient_max_abs=float(w.grad.abs().max())))
        return loss
    opt.step(closure)
    final=objective(w,z,y,p,penalty);g=torch.autograd.grad(final,w)[0]
    end=float(final.detach());grad=float(g.abs().max());assert end<=trace[0]['objective']+1e-10 and bool(torch.isfinite(w).all())
    return w.detach(),dict(initial_objective=trace[0]['objective'],final_objective=end,initial_support_CE=trace[0]['objective']-penalty*float(initial.square().sum())/2,final_support_CE=end-penalty*float(w.detach().square().sum())/2,initial_head_norm=float(initial.norm()),final_head_norm=float(w.detach().norm()),final_gradient_max_abs=grad,converged=grad<=1e-6,optimizer_iterations=int(opt.state[w]['n_iter']),closure_gradient_evaluations=len(trace),final_gradient_evaluations=1,trace=trace)


def unit(penalty=REGULARIZATION):
    z=torch.eye(2,dtype=DT);y=torch.arange(2);p=torch.full((2,),.5,dtype=DT);w=torch.tensor([[.2,-.1],[.3,.4]],dtype=DT,requires_grad=True)
    g=torch.autograd.grad(objective(w,z,y,p,penalty),w)[0]
    expected=BETA*z.T@(p[:,None]*((BETA*z@w).softmax(1)-torch.eye(2,dtype=DT)))+penalty*w
    assert torch.allclose(g,expected,atol=1e-12,rtol=0)
    u=BETA*w
    logit_objective=-(p*(z@u).log_softmax(1)[torch.arange(len(y)),y]).sum()+penalty/BETA**2*u.square().sum()/2
    assert torch.allclose(objective(w,z,y,p,penalty),logit_objective,atol=1e-12,rtol=0)
    gu=z.T@(p[:,None]*((z@u).softmax(1)-torch.eye(2,dtype=DT)))+penalty/BETA**2*u
    assert torch.allclose(g,BETA*gu,atol=1e-12,rtol=0)
    original=-(p*(BETA*z@w).log_softmax(1)[torch.arange(len(y)),y]).sum()+REGULARIZATION*w.square().sum()/2
    assert torch.allclose(objective(w,z,y,p),original,atol=1e-12,rtol=0)
    zd=z[[0,0,1]];yd=torch.tensor([0,0,1]);pd=torch.tensor([.25,.25,.5],dtype=DT)
    assert torch.allclose(objective(w,zd,yd,pd,penalty),objective(w,z,y,p,penalty),atol=1e-12,rtol=0)
    gd=torch.autograd.grad(objective(w,zd,yd,pd,penalty),w)[0];assert torch.allclose(g,gd,atol=1e-12,rtol=0)
    fitted,info=fit(z,y,p,torch.zeros((2,2),dtype=DT),penalty)
    lo,hi=0.,1.
    for _ in range(80):
        a=(lo+hi)/2
        if penalty*a-BETA*.5/(1+math.exp(2*BETA*a))>0:hi=a
        else:lo=a
    expected=torch.tensor([[a,-a],[-a,a]],dtype=DT)
    assert info['converged'] and torch.allclose(fitted,expected,atol=1e-5,rtol=0)
    return info


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['regularization']==REGULARIZATION and lock['logit_scale']==BETA
    coordinates=lock.get('ce_penalty_coordinates','head');assert coordinates in ('head','logits')
    penalty=REGULARIZATION*(BETA**2 if coordinates=='logits' else 1);assert penalty==lock.get('ce_head_penalty',REGULARIZATION)
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit_info=unit(penalty)
    case=next(v for v in lock['datasets'] if v['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    prior=Path(cfg['prior_root']);cr=Path(cfg['centroid_root']);rr=Path(cfg['ridge_root']);ar=Path(cfg['affine_root'])
    allowed={cr:{'DELIVERY.private.json',name+'.centroid.private.pt',name+'.herding.private.pt'},rr:{'DELIVERY.private.json',name+'.ridge.private.pt'}}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name==name+'.S_new.features.private.pt' or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Fit/query boundary'
        for base,names in allowed.items():
            if p.is_relative_to(base):assert p.name in names,'Sealed source boundary'
        if p.is_relative_to(ar):assert p.name=='DELIVERY.private.json' or (sealed and p.name==name+'_affine.json'),'Reference before seal'
    sys.addaudithook(audit)
    for p in [prior/(name+'.'+s) for s in ('S_old.features.private.pt','teacher.private.pt','E.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt')]+[ar/(name+'_affine.json')]:
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    for k,base in (('centroid',cr),('ridge',rr),('affine',ar)):
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
    w,fit_info=fit(z,y,mass,ridge['frame'],penalty)
    head_stats=[dict(dataset=name,class_index=c,head_norm=float(w[:,c].norm()),head_change_from_ridge_norm=float((w[:,c]-ridge['frame'][:,c]).norm())) for c in range(seen)]
    torch.save(dict(model=centroid['model'],frame=w),root/(name+'.ce.private.pt'))
    sealed=True;sealed_at=time.time();save(root/(job['id']+'.progress.json'),dict(phase='linear_ce_head_sealed',fit={k:v for k,v in fit_info.items() if k!='trace'}))
    ref=json.loads((ar/(name+'_affine.json')).read_text());classes=[];pairs=[];margin_rows=[];competitors=[];queries=[];summaries=[];gates={}
    heads={'raw_centroid':centroid['frame'],'raw_ridge':ridge['frame'],'linear_ce':w}
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
                    r=next(v for v in ref['class_rows'] if v['query']==query and v['model']==model and v['class_index']==c)
                    assert all(row[t]==r[t] for t in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-r['CE'])<1e-10
                margin_rows.append(dict(dataset=name,query=query,model=model,class_index=c,count=count,margin=stats(margin[mask]),rank=stats(rank[mask]),top2=int((rank[mask]<=2).sum()),top3=int((rank[mask]<=3).sum()),true_score_tie_samples=int((ties[mask]>0).sum())))
                for j in sorted(set(rv[mask].tolist())):
                    v=mask&(rv==j);competitors.append(dict(dataset=name,query=query,model=model,class_index=c,competitor=j,count=int(v.sum()),wrong_count=int((v&(pr!=y)).sum())))
            for group in ('all','old','new'):
                rr=[v for v in classes if v['query']==query and v['model']==model and (group=='all' or v['group']==group)]
                summaries.append(dict(dataset=name,query=query,model=model,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
        for model in ('raw_ridge','raw_centroid'):
            for c in range(seen):
                m=y==c;p=paired(pred[model][m],pred['linear_ce'][m],y[m]);pairs.append(dict(dataset=name,query=query,comparison=model+'__to__linear_ce',class_index=c,group='old' if c<old else 'new',**p))
        get=lambda m:{v['group']:v['BA'] for v in summaries if v['query']==query and v['model']==m}
        from query_node_weights import select
        gates[query]=select(get('raw_ridge'),get('linear_ce'))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,ce_penalty_coordinates=coordinates,ce_head_penalty=penalty,ce_logit_penalty=penalty/BETA**2,fit=fit_info,unit_fit=unit_info,class_rows=classes,summaries=summaries,gates=gates,primary_both_queries_pass=fit_info['converged'] and all(v['selected']=='candidate' for v in gates.values()),primary_reference='raw_ridge',paired_rows=pairs,margin_rows=margin_rows,competitor_rows=competitors,query_counts=queries,support_counts=counts,head_stats=head_stats,full_centroid_max_abs_reconstruction_error=mean_error,full_ridge_normal_relative_residual=normal_error,published_aggregates_matched=True,engineering_checks_passed=True,support_feature_sample_forwards=sum(v['count'] for v in counts),shared_query_feature_sample_forwards=sum(v['count'] for v in queries),model_logit_sample_evaluations=3*sum(v['count'] for v in queries),formal_head_fits=1,engineering_head_fits=1,production_linear_solves=0,full_reference_head_fits=0,full_support_moment_reconstructions=1,head_elements=w.numel(),head_bytes=w.numel()*w.element_size(),formal_gradient_evaluations=fit_info['closure_gradient_evaluations']+1,engineering_gradient_evaluations=unit_info['closure_gradient_evaluations']+3,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Standard linear class-balanced CE baseline on the same legal H/new-S and same feature/parameter class. Fixed beta8; head/logit penalty explicitly recorded, no bias or column normalization; loss scale interacts with regularization. No optimal-hyperparameter, full-history, compression-loss, novel-method or independent-validation claim.'))


if __name__=='__main__':main()
