"""One frozen old-H projection ablation; no fitting or model selection."""
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


def unit():
    q=torch.tensor([[1.],[0.]],dtype=DT);h=torch.tensor([[1.,0.],[2.,0.]],dtype=DT)
    base=torch.tensor([[.2,-.1],[.3,.4]],dtype=DT);full=base+torch.tensor([[.1,-.2],[.4,-.5]],dtype=DT)
    projected=base+q@(q.T@(full-base))
    assert torch.allclose(h@projected,h@full,atol=1e-12,rtol=0)
    assert torch.allclose(projected-base,q@(q.T@(projected-base)),atol=1e-12,rtol=0)
    assert torch.equal(projected[1],base[1]) and not torch.equal(projected,full)


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    case=next(v for v in lock['datasets'] if v['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['logit_scale']==BETA
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    prior=Path(cfg['prior_root']);bases={k:Path(cfg[k+'_root']) for k in ('centroid','ridge','response')}
    names={'centroid':name+'.centroid.private.pt','ridge':name+'.ridge.private.pt','response':name+'.ce.private.pt'}
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
        base=objects['ridge']['frame'];full=objects['response']['frame'];projected=base+q@(q.T@(full-base))
        old_error=float((BETA*h@(projected-full)).abs().max());span_error=float(((projected-base)-q@(q.T@(projected-base))).abs().max())
        assert old_error<1e-10 and span_error<1e-10 and bool(torch.isfinite(projected).all())
        head_rows=[dict(dataset=name,class_index=c,head_norm=float(projected[:,c].norm()),change_from_ridge_norm=float((projected[:,c]-base[:,c]).norm()),removed_outside_norm=float((full[:,c]-projected[:,c]).norm())) for c in range(seen)]
        torch.save(dict(model=objects['ridge']['model'],frame=projected),root/(name+'.projected.private.pt'))
        sealed=True;sealed_at=time.time();save(root/(job['id']+'.progress.json'),dict(phase='projected_head_sealed',basis=basis,old_support_logit_preservation_max_abs=old_error,update_span_max_abs=span_error))
        ref=json.loads((bases['response']/(name+'_ce.json')).read_text());assert ref['source_commit']==lock['inputs']['response']['source']
        heads={'raw_ridge':base,'full_response':full,'projected_response':projected};query_counts=[]
        for model,w in heads.items():
            for c,z in enumerate(parts):
                scores=BETA*z@w;ce=-scores.log_softmax(1)[:,c]
                support_rows.append(dict(**counts[c],model=model,correct=int((scores.argmax(1)==c).sum()),CE=float(ce.mean())))
        for model,key in [('raw_ridge','initial_support_CE'),('full_response','final_support_CE')]:
            assert abs(sum(v['CE'] for v in support_rows if v['model']==model)/seen-ref['fit'][key])<1e-10
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
                    if model!='projected_response':
                        a=next(a for a in ref['class_rows'] if a['query']==query and a['model']==('linear_ce' if model=='full_response' else model) and a['class_index']==c)
                        assert all(row[k]==a[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-a['CE'])<1e-10
                    margins.append(dict(dataset=name,query=query,model=model,class_index=c,count=count,margin=stats(margin[mask]),rank=stats(rank[mask]),top2=int((rank[mask]<=2).sum()),top3=int((rank[mask]<=3).sum()),true_score_tie_samples=int((ties[mask]>0).sum())))
                    for j in sorted(set(rival[mask].tolist())):
                        m=mask&(rival==j);competitors.append(dict(dataset=name,query=query,model=model,class_index=c,competitor=j,count=int(m.sum()),wrong_count=int((m&(pred[model]!=y)).sum())))
                for group in ('all','old','new'):
                    rr=[v for v in classes if v['query']==query and v['model']==model and (group=='all' or v['group']==group)]
                    summaries.append(dict(dataset=name,query=query,model=model,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
            for model in ('raw_ridge','full_response'):
                for c in range(seen):
                    mask=y==c;p=paired(pred[model][mask],pred['projected_response'][mask],y[mask]);pairs.append(dict(dataset=name,query=query,comparison=model+'__to__projected_response',class_index=c,group='old' if c<old else 'new',**p))
            bc=pred['raw_ridge']==y;fc=pred['full_response']==y;pc=pred['projected_response']==y
            for c in range(seen):
                m=y==c;harm=m&bc&~fc;rescue=m&~bc&fc
                transitions.append(dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',count=int(m.sum()),ridge_to_projected_rival_switches=int((m&(rivals['raw_ridge']!=rivals['projected_response'])).sum()),full_to_projected_rival_switches=int((m&(rivals['full_response']!=rivals['projected_response'])).sum()),full_harmed=int(harm.sum()),full_harmed_repaired=int((harm&pc).sum()),full_harmed_remaining=int((harm&~pc).sum()),full_rescued=int(rescue.sum()),full_rescues_retained=int((rescue&pc).sum()),full_rescues_lost=int((rescue&~pc).sum()),previously_both_correct_harmed=int((m&bc&fc&~pc).sum()),previously_both_wrong_rescued=int((m&~bc&~fc&pc).sum())))
            get=lambda model:{v['group']:v['BA'] for v in summaries if v['query']==query and v['model']==model}
            gates[query]=select(get('raw_ridge'),get('projected_response'))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,basis=basis,old_support_logit_preservation_max_abs=old_error,update_span_max_abs=span_error,engineering_checks_passed=True,published_aggregates_matched=True,support_counts=counts,support_class_rows=support_rows,head_stats=head_rows,class_rows=classes,paired_rows=pairs,transition_rows=transitions,margin_rows=margins,competitor_rows=competitors,summaries=summaries,metric_gates=gates,primary_reference='raw_ridge',full_method_success_evaluated=False,method_success_claim=False,reference_response_converged=ref['fit']['converged'],reference_response_primary_pass=ref['primary_both_queries_pass'],query_counts=query_counts,formal_head_fits=0,engineering_head_fits=0,gradient_evaluations=0,hessian_vector_evaluations=0,formal_thin_SVDs=1,derived_head_builds=1,new_head_elements=projected.numel(),new_head_bytes=projected.numel()*projected.element_size(),support_feature_sample_forwards=sum(v['count'] for v in counts),support_head_logit_sample_evaluations=3*sum(v['count'] for v in counts),shared_query_feature_sample_forwards=sum(v['count'] for v in query_counts),query_head_logit_sample_evaluations=3*sum(v['count'] for v in query_counts),adapter_updates=0,encoder_updates=0,teacher_updates=0,memory_updates=0,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='One fixed projection of frozen q401 delta into old-H span. No fitting or coefficient/rank search; metric gates are descriptive only, no solver convergence established or method-success/deployment claim. q401 failures unchanged.'))


if __name__=='__main__':main()
