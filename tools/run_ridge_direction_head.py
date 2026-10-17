"""Fixed unit-column or support-intercept control of a sealed full ridge head."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA,DT
from query_node_weights import select
from run_real_query_node import load,metrics,save


def unit_columns(w):
    assert w.ndim==2 and bool(torch.isfinite(w).all())
    norms=w.norm(dim=0)
    assert bool((norms>1e-12).all()),'Zero or near-zero column: stop without fallback'
    out=w/norms
    assert torch.allclose(out.norm(dim=0),torch.ones_like(norms),atol=1e-12,rtol=0)
    return out,norms


def unit():
    w=torch.diag(torch.tensor([2.,1.],dtype=DT));head,_=unit_columns(w)
    assert torch.equal(w,torch.diag(torch.tensor([2.,1.],dtype=DT)))
    assert torch.allclose(unit_columns(7*w)[0],head,atol=1e-12,rtol=0)
    z=torch.tensor([.6,.8],dtype=DT)
    assert int((z@w).argmax())==0 and int((z@head).argmax())==1
    for bad in (torch.zeros(2,2,dtype=DT),torch.full((2,2),float('nan'),dtype=DT)):
        try:unit_columns(bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid head accepted')


def support_bias(w,center):
    assert w.ndim==2 and w.shape[1]>0 and bool(torch.isfinite(w).all())
    assert center.shape==(w.shape[0],) and bool(torch.isfinite(center).all())
    bias=torch.full((w.shape[1],),1/w.shape[1],dtype=w.dtype)-center@w
    residual=float((center@w+bias-1/w.shape[1]).abs().max())
    assert bool(torch.isfinite(bias).all()) and residual<=1e-12
    return bias,residual


def unit_bias():
    z=torch.tensor([[1.,0.],[2.,0.],[0.,1.]],dtype=DT);p=torch.tensor([.25,.25,.5],dtype=DT)
    target=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],dtype=DT);w=torch.tensor([[2.,-.5],[.3,1.]],dtype=DT)
    center=(p[:,None]*z).sum(0);b,_=support_bias(w,center);errors=z@w+b-target
    assert torch.allclose((p[:,None]*errors).sum(0),torch.zeros(2,dtype=DT),atol=1e-12,rtol=0)
    loss=(p[:,None]*errors.square()).sum();delta=torch.tensor([.2,-.4],dtype=DT)
    for change in (delta,-delta):
        assert abs(float((p[:,None]*(errors+change).square()).sum()-loss-change.square().sum()))<1e-12
    shift=torch.tensor([.3,-.8],dtype=DT);shifted,_=support_bias(w,center+shift)
    assert torch.allclose((z+shift)@w+shifted,z@w+b,atol=1e-12,rtol=0)
    zero,_=support_bias(w,torch.zeros(2,dtype=DT));assert torch.equal(zero,torch.full((2,),.5,dtype=DT))


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['logit_scale']==BETA==8
    intercept=lock.get('support_intercept',False)
    torch.set_num_threads(2);torch.set_default_dtype(DT);(unit_bias if intercept else unit)()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    prior=Path(cfg['prior_root']);ridge=Path(cfg['ridge_root']);refroot=Path(cfg['direction_root'] if intercept else cfg['old_covariance_root']);refname=name+('_direction.json' if intercept else '_old_covariance.json');center_root=Path(cfg['centered_root']) if intercept else None
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'},'Query/support access boundary'
        if p.is_relative_to(ridge):assert p.name in {name+'.ridge.private.pt','DELIVERY.private.json'},'Only sealed full ridge head and provenance allowed'
        if center_root is not None and p.is_relative_to(center_root):assert p.name in {name+'.centered.private.pt','DELIVERY.private.json'},'Only frozen center and provenance allowed'
        if p.is_relative_to(refroot):assert sealed and p.name==refname,'Historical aggregates before seal'
    sys.addaudithook(audit)
    for p in [prior/(name+'.'+s) for s in ('S_old.features.private.pt','S_new.features.private.pt','teacher.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt')]+[refroot/refname]:
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    receipt=json.loads((ridge/'DELIVERY.private.json').read_text())
    assert receipt['design_source_commit']==lock['original_head_source_commit'] and receipt['public_commit']==lock['original_head_delivery_commit']
    original=load(ridge/(name+'.ridge.private.pt'));w=original['frame'];model=original['model']
    assert set(original)=={'model','frame'} and w.shape==(1536,seen) and w.dtype==DT
    assert len(model)==2 and all(v.ndim==2 and v.shape[0]==1536 and v.dtype==DT and bool(torch.isfinite(v).all()) and not bool(v.any()) for v in model) and model[0].shape==model[1].shape
    save(root/(job['id']+'.progress.json'),dict(phase='fixed_support_intercept' if intercept else 'fixed_unit_column_transform',dataset=name))
    bias_residual=None;center_norm=None;bias=None
    if intercept:
        center_receipt=json.loads((center_root/'DELIVERY.private.json').read_text())
        assert center_receipt['design_source_commit']==lock['center_source_commit'] and center_receipt['public_commit']==lock['center_delivery_commit']
        centered=load(center_root/(name+'.centered.private.pt'));center=centered['center']
        assert len(centered['model'])==2 and centered['frame'].shape==w.shape and center.dtype==DT and all(torch.equal(a,b) for a,b in zip(centered['model'],model))
        bias,bias_residual=support_bias(w,center);center_norm=float(center.norm());head=w;norms=w.norm(dim=0)
        assert bool((norms>1e-12).all());obj=dict(model=model,frame=head,bias=BETA*bias)
    else:
        head,norms=unit_columns(w);obj=dict(model=model,frame=head)
    torch.save(obj,root/(name+('.intercept.private.pt' if intercept else '.direction.private.pt')));sealed=True;sealed_at=time.time()
    ref=json.loads((refroot/refname).read_text());rows=list(ref['class_rows']);references=lock['references'];label='raw_ridge_support_intercept' if intercept else 'raw_ridge_unit_columns';summaries=[];gates={};counts=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));x=data['x'];y=data['y']
        assert x.shape==(len(y),1536) and x.dtype==DT and bool(torch.isfinite(x).all()) and y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        counts.append(dict(query=query,count=len(y)));_,rr,_=metrics(obj,data,old,seen);rows.extend(dict(dataset=name,query=query,model=label,**r) for r in rr)
        for ref_label in references+[label]:
            for group in ('all','old','new'):
                cc=[r for r in rows if r['query']==query and r['model']==ref_label and (group=='all' or r['group']==group)]
                assert len(cc)==(seen if group=='all' else old if group=='old' else seen-old)
                summaries.append(dict(dataset=name,query=query,model=ref_label,group=group,BA=sum(r['recall'] for r in cc)/len(cc),CE=sum(r['CE'] for r in cc)/len(cc)))
        get=lambda label:{r['group']:r['BA'] for r in summaries if r['query']==query and r['model']==label}
        gates[query]={ref_label:select(get(ref_label),get(label)) for ref_label in references}
    stats=[dict(class_index=c,original_head_norm=float(norms[c]),head_norm=float(head[:,c].norm()),positive_rescale=1. if intercept else float(1/norms[c]),**(dict(unscaled_intercept=float(bias[c]),logit_bias=float(BETA*bias[c])) if intercept else {})) for c in range(seen)]
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,class_rows=rows,summaries=summaries,gates=gates,primary_reference='raw_ridge',primary_both_queries_pass=all(g['raw_ridge']['selected']=='candidate' for g in gates.values()),head_stats=stats,head_shape=list(head.shape),head_elements=head.numel(),head_bytes=head.numel()*head.element_size(),original_head_source_commit=receipt['design_source_commit'],original_head_delivery_commit=receipt['public_commit'],head_column_normalization=not intercept,support_intercept=intercept,conditional_bias_fits=int(intercept),center_source_commit=lock.get('center_source_commit') if intercept else None,center_delivery_commit=lock.get('center_delivery_commit') if intercept else None,center_norm=center_norm,bias_mean_equation_residual=bias_residual,derived_bias_elements=seen if intercept else 0,derived_bias_bytes=seen*head.element_size() if intercept else 0,engineering_bias_constructions=3 if intercept else 0,logit_scale=BETA,derived_head_constructions=0 if intercept else 1,production_linear_solves=0,engineering_small_linear_solves=0,production_elementwise_solutions=0,engineering_small_elementwise_solutions=0,engineering_unit_checks_passed=True,support_feature_forwards=0,query_sample_forwards=sum(v['count'] for v in counts),query_counts=counts,gradient_evaluations=0,engineering_gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,query_used_for_fitting=False,old_S_access=False,new_S_access=False,H_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Support-derived conditional optimal squared-loss intercept for fixed full ridge W; W unchanged, intercept data-fitted from reused support center. Not joint affine ridge optimization or independent calibration evidence.' if intercept else 'Unit-column directions of sealed full raw ridge; no new fit or parameter selection. Fixed BETA8 changes class scales and CE scale. Not equivalent ridge optimum, independent calibration evidence, or a new method.'))


if __name__=='__main__':main()
