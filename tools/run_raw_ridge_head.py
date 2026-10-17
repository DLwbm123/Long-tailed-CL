"""Existing class-balanced ridge on the same frozen raw support features."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from check_reserved_competition_memory import BETA,DT,features
from query_node_weights import select
from run_real_query_node import load,metrics,save
import prototype_analytic as analytic


REGULARIZATION=.001


def diagonal_ridge(bank,regularization):
    k=len(bank['n']);diagonal=np.diag(bank['S'])/k+regularization;rhs=bank['mu'].T/k
    assert np.isfinite(diagonal).all() and (diagonal>0).all() and np.isfinite(rhs).all()
    w=rhs/diagonal[:,None];residual=float(np.linalg.norm(diagonal[:,None]*w-rhs)/max(np.linalg.norm(rhs),1e-15))
    assert np.isfinite(w).all() and residual<=1e-7
    return w,residual


def prototype_ridge(bank,regularization):
    return analytic.ridge(dict(bank,S=bank['mu'].T@bank['mu']),regularization)


def isotropic_ridge(bank,regularization):
    between=bank['mu'].T@bank['mu'];trace=float(np.trace(bank['S'])-np.trace(between))
    assert np.isfinite(trace) and trace>=0
    isotropic=between+np.eye(len(between))*(trace/len(between))
    assert np.isclose(np.trace(isotropic),np.trace(bank['S']),atol=1e-12,rtol=1e-12)
    return analytic.ridge(dict(bank,S=isotropic),regularization)


def within_diagonal_ridge(bank,regularization):
    between=bank['mu'].T@bank['mu'];variances=np.diag(bank['S'])-np.diag(between)
    assert np.isfinite(variances).all() and (variances>=0).all()
    np.fill_diagonal(between,np.diag(bank['S']))
    return analytic.ridge(dict(bank,S=between),regularization)


def new_covariance_ridge(bank,new_bank,regularization):
    combined=bank['mu'].T@bank['mu']+new_bank['S']-new_bank['mu'].T@new_bank['mu']
    old_variances=np.diag(bank['S'])-np.diag(combined)
    assert np.isfinite(old_variances).all() and (old_variances>=0).all()
    np.fill_diagonal(combined,np.diag(bank['S']))
    return analytic.ridge(dict(bank,S=combined),regularization)


def unit(diagonal=False,prototype=False,isotropic=False,within_diagonal=False,new_covariance=False):
    if new_covariance:
        x=np.eye(2);y=np.array([0,1]);bank=analytic.append(analytic.empty(2),x,y,[0,1]);nb=analytic.append(analytic.empty(2),x[1:],y[1:],[1])
        w,res=new_covariance_ridge(bank,nb,REGULARIZATION)
        assert np.allclose(w,np.eye(2)/(1+2*REGULARIZATION),atol=1e-12,rtol=0) and res<1e-12
        xx=np.vstack((x,x[1:]));yy=np.array([0,1,1]);bb=analytic.append(analytic.empty(2),xx,yy,[0,1]);nn=analytic.append(analytic.empty(2),xx[1:],yy[1:],[1]);ww,_=new_covariance_ridge(bb,nn,REGULARIZATION)
        assert np.allclose(ww,w,atol=1e-12,rtol=0)
        correlated=dict(S=np.array([[3.,.75],[.75,3.]]),mu=np.eye(2),n=np.array([2,2]));nb=dict(S=np.array([[1.,.25],[.25,2.]]),mu=np.array([[0.,1.]]),n=np.array([2]))
        wc,_=new_covariance_ridge(correlated,nb,REGULARIZATION);a=1.5+REGULARIZATION;b=.125
        assert np.allclose(wc,.5*np.array([[a,-b],[-b,a]])/(a*a-b*b),atol=1e-12,rtol=0)
        assert np.array_equal(correlated['S'],np.array([[3.,.75],[.75,3.]]))
        assert np.linalg.norm((correlated['S']/2+REGULARIZATION*np.eye(2))@wc-np.eye(2)/2)>.05
        return
    solve=within_diagonal_ridge if within_diagonal else isotropic_ridge if isotropic else prototype_ridge if prototype else diagonal_ridge if diagonal else analytic.ridge
    x=np.eye(2,dtype=np.float64);bank=analytic.append(analytic.empty(2),x,np.array([0,1]),[0,1]);w,res=solve(bank,REGULARIZATION)
    assert np.allclose(w,np.eye(2)/(1+2*REGULARIZATION),atol=1e-12,rtol=0) and res<1e-12
    xx=np.vstack((x,x[1:]));bb=analytic.append(analytic.empty(2),xx,np.array([0,1,1]),[0,1]);ww,_=solve(bb,REGULARIZATION)
    assert np.allclose(ww,w,atol=1e-12,rtol=0),'Class-balanced duplication changed solution'
    if diagonal:
        correlated=dict(S=np.array([[2.,1.],[1.,2.]]),mu=np.eye(2),n=np.array([1,1]))
        wd,_=diagonal_ridge(correlated,REGULARIZATION)
        assert np.allclose(wd,np.eye(2)*.5/(1+REGULARIZATION),atol=1e-12,rtol=0)
        assert np.linalg.norm((correlated['S']/2+REGULARIZATION*np.eye(2))@wd-np.eye(2)/2)>.1
    if prototype:
        correlated=dict(S=np.array([[2.,1.],[1.,2.]]),mu=np.eye(2),n=np.array([1,1]))
        wp,_=prototype_ridge(correlated,REGULARIZATION)
        assert np.allclose(wp,w,atol=1e-12,rtol=0),'Within-class scatter changed prototype-only fit'
        assert np.linalg.norm((correlated['S']/2+REGULARIZATION*np.eye(2))@wp-np.eye(2)/2)>.1
    if isotropic:
        correlated=dict(S=np.array([[3.,1.],[1.,2.]]),mu=np.eye(2),n=np.array([1,1]))
        wi,_=isotropic_ridge(correlated,REGULARIZATION)
        assert np.allclose(wi,np.eye(2)*.5/(1.25+REGULARIZATION),atol=1e-12,rtol=0)
        assert np.array_equal(correlated['S'],np.array([[3.,1.],[1.,2.]]))
        assert np.linalg.norm((correlated['S']/2+REGULARIZATION*np.eye(2))@wi-np.eye(2)/2)>.1
    if within_diagonal:
        correlated=dict(S=np.array([[3.,1.5],[1.5,3.]]),mu=np.array([[1.,0.],[1.,1.]]),n=np.array([1,1]))
        wd,_=within_diagonal_ridge(correlated,REGULARIZATION);a=1.5+REGULARIZATION;b=.5
        expected=.5*np.array([[a,a-b],[-b,a-b]])/(a*a-b*b)
        assert np.allclose(wd,expected,atol=1e-12,rtol=0)
        assert np.array_equal(correlated['S'],np.array([[3.,1.5],[1.5,3.]]))
        assert np.linalg.norm((correlated['S']/2+REGULARIZATION*np.eye(2))@wd-correlated['mu'].T/2)>.05



def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['regularization']==REGULARIZATION
    diagonal=lock.get('diagonal_second_moment',False);prototype=lock.get('prototype_only_second_moment',False);isotropic=lock.get('isotropic_within_second_moment',False);within_diagonal=lock.get('diagonal_within_second_moment',False);new_covariance=lock.get('new_class_cross_covariance_only',False);assert sum((diagonal,prototype,isotropic,within_diagonal,new_covariance))<=1
    label='raw_new_covariance_ridge' if new_covariance else 'raw_within_diagonal_ridge' if within_diagonal else 'raw_isotropic_ridge' if isotropic else 'raw_prototype_ridge' if prototype else 'raw_diagonal_ridge' if diagonal else 'raw_ridge'
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit(diagonal,prototype,isotropic,within_diagonal,new_covariance)
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];prior=Path(cfg['prior_root']);refroot=Path(cfg['within_diagonal_root'] if new_covariance else cfg['isotropic_root'] if within_diagonal else cfg['prototype_root'] if isotropic else cfg['diagonal_root'] if prototype else cfg['ridge_root'] if diagonal else cfg['centered_root']);refname=name+('_within_diagonal.json' if new_covariance else '_isotropic.json' if within_diagonal else '_prototype.json' if isotropic else '_diagonal.json' if prototype else '_ridge.json' if diagonal else '_centered.json');sealed=False
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in {name+'.teacher.private.pt',name+'.S_new.features.private.pt'} or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Ridge fit/query boundary'
        if p.is_relative_to(refroot):assert sealed and p.name==refname,'Historical aggregates before seal'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    state=load(prior/(name+'.teacher.private.pt'));bank=load(root/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'));model=tuple(torch.zeros_like(v) for v in state['teacher'])
    zz=[];yy=[];stats=[];save(root/(job['id']+'.progress.json'),dict(phase='fixed_class_balanced_ridge',dataset=name))
    with torch.no_grad():
        for c in range(seen):
            if c<old:x,p=bank[c]
            else:x=new['x'][new['y']==c];p=torch.full((len(x),),1/len(x),dtype=DT)
            assert len(x)>0 and torch.allclose(p,torch.full_like(p,1/len(x)),atol=1e-12,rtol=0),'Existing helper requires uniform class weights'
            z=torch.cat([features(x[i:i+256],model) for i in range(0,len(x),256)]);assert z.shape==(len(x),1536) and bool(torch.isfinite(z).all())
            zz.append(z.numpy());yy.append(np.full(len(x),c,dtype=np.int64));stats.append(dict(class_index=c,count=len(x)))
    z=np.concatenate(zz);y=np.concatenate(yy);moments=analytic.append(analytic.empty(1536),z,y,list(range(seen)))
    if new_covariance:
        new_moments=analytic.append(analytic.empty(1536),z[y>=old],y[y>=old],list(range(old,seen)))
        w,residual=new_covariance_ridge(moments,new_moments,REGULARIZATION)
    else:
        w,residual=(within_diagonal_ridge if within_diagonal else isotropic_ridge if isotropic else prototype_ridge if prototype else diagonal_ridge if diagonal else analytic.ridge)(moments,REGULARIZATION)
    head=torch.from_numpy(w);assert head.shape==(1536,seen) and bool(torch.isfinite(head).all())
    obj=dict(model=model,frame=head);torch.save(obj,root/(name+'.ridge.private.pt'));sealed=True;sealed_at=time.time()
    for c in range(seen):stats[c]['head_norm']=float(head[:,c].norm())
    ref=json.loads((refroot/refname).read_text());rows=list(ref['class_rows']);references=['raw_centroid','MM','HH','identity_ETF','teacher_centroid','teacher_ETF','coordinate_median','centered_centroid']+(['raw_ridge'] if diagonal or prototype or isotropic or within_diagonal or new_covariance else [])+(['raw_diagonal_ridge'] if prototype or isotropic or within_diagonal or new_covariance else [])+(['raw_prototype_ridge'] if isotropic or within_diagonal or new_covariance else [])+(['raw_isotropic_ridge'] if within_diagonal or new_covariance else [])+(['raw_within_diagonal_ridge'] if new_covariance else []);summaries=[];gates={};counts=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));assert data['y'].dtype==torch.int64 and bool(((data['y']>=0)&(data['y']<seen)).all())
        counts.append(dict(query=query,count=len(data['y'])));_,rr,_=metrics(obj,data,old,seen);rows.extend(dict(dataset=name,query=query,model=label,**r) for r in rr)
        for ref_label in references+[label]:
            for group in ('all','old','new'):
                cc=[r for r in rows if r['query']==query and r['model']==ref_label and (group=='all' or r['group']==group)]
                assert len(cc)==(seen if group=='all' else old if group=='old' else seen-old)
                summaries.append(dict(dataset=name,query=query,model=ref_label,group=group,BA=sum(r['recall'] for r in cc)/len(cc),CE=sum(r['CE'] for r in cc)/len(cc)))
        get=lambda label:{r['group']:r['BA'] for r in summaries if r['query']==query and r['model']==label}
        gates[query]={ref_label:select(get(ref_label),get(label)) for ref_label in references}
    within_stats={}
    if within_diagonal:
        within=moments['S']-moments['mu'].T@moments['mu'];variances=np.diag(within)/seen
        within_stats=dict(within_diagonal_variance_min=float(variances.min()),within_diagonal_variance_max=float(variances.max()),within_offdiagonal_frobenius_fraction=float(np.linalg.norm(within-np.diag(np.diag(within)))/max(np.linalg.norm(within),1e-15)))
    if new_covariance:
        within=moments['S']-moments['mu'].T@moments['mu'];new_within=new_moments['S']-new_moments['mu'].T@new_moments['mu'];old_within=within-new_within
        within_stats=dict(old_within_trace=float(np.trace(old_within)/seen),new_within_trace=float(np.trace(new_within)/seen),old_within_frobenius_fraction=float(np.linalg.norm(old_within)/max(np.linalg.norm(within),1e-15)),new_within_frobenius_fraction=float(np.linalg.norm(new_within)/max(np.linalg.norm(within),1e-15)),old_within_offdiagonal_frobenius_fraction=float(np.linalg.norm(old_within-np.diag(np.diag(old_within)))/max(np.linalg.norm(within),1e-15)),additional_support_rows_for_moments=int((y>=old).sum()))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,class_rows=rows,summaries=summaries,gates=gates,primary_reference='raw_ridge' if diagonal or prototype or isotropic or within_diagonal or new_covariance else 'raw_centroid',primary_both_queries_pass=all(g['raw_ridge' if diagonal or prototype or isotropic or within_diagonal or new_covariance else 'raw_centroid']['selected']=='candidate' for g in gates.values()),head_stats=stats,head_shape=list(head.shape),head_elements=head.numel(),head_bytes=head.numel()*head.element_size(),regularization=REGULARIZATION,solve_relative_residual=residual,diagonal_second_moment=diagonal,prototype_only_second_moment=prototype,isotropic_within_second_moment=isotropic,diagonal_within_second_moment=within_diagonal,new_class_cross_covariance_only=new_covariance,**within_stats,isotropic_within_variance=float((np.trace(moments['S'])-np.square(moments['mu']).sum())/(seen*1536)) if isotropic else None,within_class_trace_fraction=float((np.trace(moments['S'])-np.square(moments['mu']).sum())/np.trace(moments['S'])),within_class_frobenius_fraction=float(np.linalg.norm(moments['S']-moments['mu'].T@moments['mu'])/np.linalg.norm(moments['S'])),second_moment_offdiagonal_fraction=float(np.linalg.norm(moments['S']-np.diag(np.diag(moments['S'])))/np.linalg.norm(moments['S'])),production_linear_solves=0 if diagonal else 1,production_elementwise_solutions=1 if diagonal else 0,engineering_small_linear_solves=0 if diagonal else 3 if prototype or isotropic or within_diagonal or new_covariance else 2,engineering_small_elementwise_solutions=3 if diagonal else 0,logit_scale=BETA,head_constructions=1,head_column_normalization=False,derived_bias_elements=0,support_feature_forwards=len(z),query_sample_forwards=sum(v['count'] for v in counts),query_counts=counts,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,query_used_for_fitting=False,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Full, diagonal, prototype-only, isotropic-within, diagonal-within or new-class-cross-covariance class-balanced second-moment readout on fixed H/new-S features; variant recorded explicitly. Removing cross terms, removing within-class scatter, isotropizing it or removing all/old-class cross-coordinate terms changes the objective; not an equivalent full least-squares solve. No temperature/regularization selection; CE depends on unnormalized head scale.'))


if __name__=='__main__':main()
