"""Frozen-head margin decomposition against the span of legal old exemplars."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA, DT, features
from run_head_competition_diagnostic import competition, stats
from run_real_query_node import load, save


def decompose(z,q,baseline,candidate,y,s0,s1):
    parallel=(z@q)@q.T;outside=z-parallel
    change_parallel=BETA*parallel@(candidate-baseline);change_outside=BETA*outside@(candidate-baseline)
    error=float((change_parallel+change_outside-(s1-s0)).abs().max());assert error<1e-10
    r0,m0,_,_=competition(s0,y);r1,m1,_,_=competition(s1,y);i=torch.arange(len(y))
    gap=lambda s,r:s[i,y]-s[i,r]
    values=dict(actual_margin_change=m1-m0,baseline_rival_parallel=gap(change_parallel,r0),baseline_rival_outside=gap(change_outside,r0),candidate_switch_adjustment=m1-gap(s1,r0),candidate_rival_parallel=gap(change_parallel,r1),candidate_rival_outside=gap(change_outside,r1),baseline_switch_adjustment=gap(s0,r1)-m0)
    a=values['baseline_rival_parallel']+values['baseline_rival_outside']+values['candidate_switch_adjustment']
    b=values['candidate_rival_parallel']+values['candidate_rival_outside']+values['baseline_switch_adjustment']
    error=max(error,float((a-values['actual_margin_change']).abs().max()),float((b-values['actual_margin_change']).abs().max()))
    assert error<1e-10 and float(values['candidate_switch_adjustment'].max())<1e-10 and float(values['baseline_switch_adjustment'].min())>-1e-10
    values.update({k+'_abs':values[k].abs() for k in ('baseline_rival_parallel','baseline_rival_outside','candidate_rival_parallel','candidate_rival_outside')})
    return values,r0!=r1,error


def unit():
    z=torch.tensor([[1.,2.],[2.,-1.]],dtype=DT);q=torch.tensor([[1.],[0.]],dtype=DT)
    w=torch.tensor([[.2,.5,-.4],[-.3,.1,.6]],dtype=DT);dw=torch.tensor([[.4,-.8,.2],[.3,-.4,-.2]],dtype=DT);y=torch.tensor([0,2])
    v,sw,e=decompose(z,q,w,w+dw,y,BETA*z@w,BETA*z@(w+dw));assert bool(sw.any()) and e<1e-12
    explicit=BETA*(z[:,0,None]*dw[0]);r0=competition(BETA*z@w,y)[0];i=torch.arange(2)
    assert torch.allclose(v['baseline_rival_parallel'],explicit[i,y]-explicit[i,r0],atol=1e-12,rtol=0)
    zero,sw,e=decompose(z,q,w,w,y,BETA*z@w,BETA*z@w);assert not bool(sw.any()) and all(not bool(a.any()) for a in zero.values())


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    case=next(v for v in lock['datasets'] if v['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['logit_scale']==BETA
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    prior=Path(cfg['prior_root']);bases={k:Path(cfg[k+'_root']) for k in ('centroid','ridge','parameter','response')}
    names={'centroid':name+'.centroid.private.pt','ridge':name+'.ridge.private.pt','parameter':name+'.ce.private.pt','response':name+'.ce.private.pt'}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name==name+'.S_new.features.private.pt' or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Query before basis or forbidden data'
        for k,base in bases.items():
            if p.is_relative_to(base):assert p.name in {'DELIVERY.private.json',names[k]} or (k=='centroid' and p.name==name+'.herding.private.pt') or (k=='response' and sealed and p.name==name+'_ce.json'),'Frozen artifact boundary'
    sys.addaudithook(audit)
    for p in [prior/(name+'.'+v) for v in ('S_old.features.private.pt','teacher.private.pt','E.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt')]+[bases['response']/(name+'_ce.json')]:
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    objects={}
    for k,base in bases.items():
        rec=json.loads((base/'DELIVERY.private.json').read_text());assert rec['design_source_commit']==lock['inputs'][k]['source'] and rec['public_commit']==lock['inputs'][k]['delivery']
        v=load(base/names[k]);assert v['frame'].shape==(1536,seen) and v['frame'].dtype==DT and bool(torch.isfinite(v['frame']).all()) and 'bias' not in v
        assert len(v['model'])==2 and all(a.ndim==2 and a.shape[0]==1536 and a.dtype==DT and bool(torch.isfinite(a).all()) and not bool(a.any()) for a in v['model']);objects[k]=v
    assert all(all(torch.equal(a,b) for a,b in zip(v['model'],objects['ridge']['model'])) for v in objects.values())
    bank=load(bases['centroid']/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    assert new['x'].dtype==DT and new['x'].ndim==2 and new['x'].shape[1]==1536 and bool(torch.isfinite(new['x']).all()) and new['y'].dtype==torch.int64 and bool(((new['y']>=old)&(new['y']<seen)).all())
    parts=[];counts=[];support_rows=[];coverage=[];classes=[];rows=[];max_error=0.
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
        def residual_fraction(z):
            f=(z-(z@q)@q.T).square().sum(1)/z.square().sum(1);assert bool(torch.isfinite(f).all()) and float(f.min())>=0 and float(f.max())<=1+1e-10;return f
        for c,z in enumerate(parts):support_rows.append(dict(**counts[c],outside_squared_fraction=stats(residual_fraction(z))))
        sealed=True;sealed_at=time.time();save(root/(job['id']+'.progress.json'),dict(phase='old_memory_basis_sealed',basis=basis))
        ref=json.loads((bases['response']/(name+'_ce.json')).read_text());assert ref['source_commit']==lock['inputs']['response']['source']
        heads={'raw_ridge':objects['ridge']['frame'],'parameter_ce':objects['parameter']['frame'],'linear_ce':objects['response']['frame']};query_counts=[]
        for query in ('Qfit','Qgate'):
            data=load(prior/(name+'.'+query+'.features.private.pt'));x=data['x'];y=data['y'];n=len(y)
            assert x.shape==(n,1536) and x.dtype==DT and bool(torch.isfinite(x).all()) and y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
            z=transform(x);fraction=residual_fraction(z);scores={k:BETA*z@w for k,w in heads.items()};pred={k:s.argmax(1) for k,s in scores.items()};base_correct=pred['raw_ridge']==y;query_counts.append(dict(query=query,count=n))
            for c in range(seen):
                mask=y==c;assert bool(mask.any());coverage.append(dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',count=int(mask.sum()),outside_squared_fraction=stats(fraction[mask])))
            for model,s in scores.items():
                ce=-s.log_softmax(1)[torch.arange(n),y]
                for c in range(seen):
                    mask=y==c;pr=pred[model][mask];row=dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',count=int(mask.sum()),correct=int((pr==c).sum()),CE=float(ce[mask].mean()),errors_to_old=int(((pr!=c)&(pr<old)).sum()),errors_to_new=int(((pr!=c)&(pr>=old)).sum()));classes.append(row)
                    a=next(a for a in ref['class_rows'] if a['query']==query and a['model']==model and a['class_index']==c)
                    assert all(row[k]==a[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-a['CE'])<1e-10
            for model in ('parameter_ce','linear_ce'):
                values,sw,error=decompose(z,q,heads['raw_ridge'],heads[model],y,scores['raw_ridge'],scores[model]);max_error=max(max_error,error);correct=pred[model]==y
                partitions={'all':torch.ones(n,dtype=torch.bool),'harmed':base_correct&~correct,'rescued':~base_correct&correct,'both_correct':base_correct&correct,'both_wrong':~base_correct&~correct}
                assert sum(int(v.sum()) for k,v in partitions.items() if k!='all')==n
                for c in range(seen):
                    for outcome,mask in partitions.items():
                        m=(y==c)&mask;count=int(m.sum())
                        rows.append(dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',outcome=outcome,count=count,competitor_switches=int(sw[m].sum()),outside_dominates_baseline_rival_count=int((values['baseline_rival_outside'][m].abs()>values['baseline_rival_parallel'][m].abs()).sum()),outside_squared_fraction=stats(fraction[m]) if count else None,**{k:stats(v[m]) if count else None for k,v in values.items()}))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,basis=basis,engineering_checks_passed=True,published_aggregates_matched=True,decomposition_max_abs_error=max_error,support_counts=counts,support_coverage_rows=support_rows,query_coverage_rows=coverage,class_rows=classes,decomposition_rows=rows,query_counts=query_counts,formal_head_fits=0,engineering_head_fits=0,gradient_evaluations=0,hessian_vector_evaluations=0,formal_thin_SVDs=1,support_feature_sample_forwards=sum(v['count'] for v in counts),shared_query_feature_sample_forwards=sum(v['count'] for v in query_counts),query_head_logit_sample_evaluations=3*sum(v['count'] for v in query_counts),diagnostic_parallel_outside_logit_sample_evaluations=4*sum(v['count'] for v in query_counts),new_head_bytes=0,adapter_updates=0,encoder_updates=0,teacher_updates=0,memory_updates=0,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Descriptive decomposition of frozen q398/q401 heads relative to original ridge and old-H span. Outside old-H span is not necessarily unconstrained by full H+newS response protection. No fitting, query-based rank choice, deployment, compression-loss or independent validation claim.'))


if __name__=='__main__':main()
