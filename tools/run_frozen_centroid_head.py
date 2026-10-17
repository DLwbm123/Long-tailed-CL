"""Fixed teacher features with a support-derived cosine centroid head."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT,BETA,features,scores
from query_node_weights import select
from run_real_query_node import load,metrics,save


def centroid(z,p):
    assert bool(torch.isfinite(z).all()) and bool((p>=0).all()) and abs(float(p.sum())-1)<1e-12
    mean=(z*p[:,None]).sum(0);norm=mean.norm();assert float(norm)>1e-12
    return mean/norm,float(norm)


def unit():
    z=torch.eye(2,dtype=DT);q,n=centroid(z,torch.tensor([.75,.25],dtype=DT))
    assert torch.allclose(q,torch.tensor([3.,1.],dtype=DT)/(10**.5)) and abs(n-(10**.5)/4)<1e-12
    assert torch.allclose(features(z,(torch.zeros(2,1,dtype=DT),torch.zeros(2,1,dtype=DT))),z)
    try:centroid(torch.tensor([[1.,0.],[-1.,0.]],dtype=DT),torch.ones(2,dtype=DT)/2)
    except AssertionError:pass
    else:raise AssertionError('Zero centroid must stop')


def euclidean_head(head,stats):
    means=head*torch.tensor([s['mean_norm'] for s in stats],dtype=DT)
    return means,-BETA/2*means.square().sum(0)


def euclidean_unit():
    z=torch.eye(2,dtype=DT);model=(torch.zeros(2,1,dtype=DT),torch.zeros(2,1,dtype=DT))
    mu,bias=euclidean_head(z,[dict(mean_norm=.5),dict(mean_norm=.8)])
    ss=scores(z,model,mu)+bias
    direct=-BETA/2*(z[:,None,:]-mu.T[None,:,:]).square().sum(2)
    assert torch.allclose(ss-direct,torch.full_like(ss,BETA/2))
    assert torch.equal(ss.argmax(1),direct.argmax(1))
    equal,b=euclidean_head(z,[dict(mean_norm=.7),dict(mean_norm=.7)])
    assert torch.allclose(scores(z,model,equal)+b,.7*scores(z,model,z)-BETA/2*.7**2)


def coordinate_median(z,p):
    assert bool(torch.isfinite(z).all()) and torch.allclose(p,torch.full_like(p,1/len(z)),atol=1e-12,rtol=0)
    center=torch.quantile(z,.5,dim=0);norm=center.norm();assert float(norm)>1e-12
    return center/norm,float(norm)


def median_unit():
    z=torch.eye(2,dtype=DT);q,n=coordinate_median(z,torch.ones(2,dtype=DT)/2)
    assert torch.allclose(q,torch.ones(2,dtype=DT)/2**.5) and abs(n-2**.5/2)<1e-12
    q,_=coordinate_median(torch.tensor([[0.,0.],[2.,0.],[3.,4.]],dtype=DT),torch.ones(3,dtype=DT)/3)
    assert torch.equal(q,torch.tensor([1.,0.],dtype=DT))
    q,_=coordinate_median(torch.tensor([[3.,4.]],dtype=DT),torch.ones(1,dtype=DT));assert torch.allclose(q,torch.tensor([.6,.8],dtype=DT))
    try:coordinate_median(torch.tensor([[1.,0.],[-1.,0.]],dtype=DT),torch.ones(2,dtype=DT)/2)
    except AssertionError:pass
    else:raise AssertionError('Zero median must stop')


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];prior=Path(cfg['prior_root']);refroot=Path(cfg['bank_root']);sealed=False
    raw=lock.get('raw_features',False);euclidean=lock.get('euclidean_head',False)
    median=lock.get('coordinate_median_head',False);compare_raw=euclidean or median
    assert not (euclidean and median) and (not compare_raw or raw)
    if euclidean:euclidean_unit()
    if median:median_unit()
    prior_centroid=Path(cfg['raw_centroid_root'] if compare_raw else cfg['centroid_root']) if raw else None
    prior_result_name=name+('_raw.json' if compare_raw else '_centroid.json')
    allowed={name+'.teacher.private.pt',name+'.S_new.features.private.pt'}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in allowed or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Centroid fit/query boundary'
        if prior_centroid and p.is_relative_to(prior_centroid):assert sealed and p.name==prior_result_name,'Prior centroid aggregate only after seal'
        if p.is_relative_to(refroot):assert sealed and p.name==name+'_factorial.json','Historical aggregate only after seal'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    state=load(prior/(name+'.teacher.private.pt'));bank=load(root/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    teacher_model=tuple(torch.zeros_like(v) for v in state['teacher']) if raw else state['teacher']
    matched_label='identity_ETF' if raw else 'teacher_ETF'
    save(root/(job['id']+'.progress.json'),dict(phase='construct_centroid_head',dataset=name));heads=[];stats=[]
    with torch.no_grad():
        for c in range(seen):
            if c<old:x,p=bank[c]
            else:x=new['x'][new['y']==c];p=torch.full((len(x),),1/len(x),dtype=DT)
            z=torch.cat([features(x[i:i+256],teacher_model) for i in range(0,len(x),256)])
            q,n=centroid(z,p);location_norm=n
            if median:q,location_norm=coordinate_median(z,p)
            heads.append(q);stats.append(dict(class_index=c,count=len(x),mean_norm=n,location_norm=location_norm))
    head=torch.stack(heads,1);assert head.shape==(state['frame'].shape[0],seen)
    obj=dict(model=teacher_model,frame=head)
    if euclidean:
        obj['frame'],obj['bias']=euclidean_head(head,stats)
        assert obj['bias'].shape==(seen,) and bool(torch.isfinite(obj['bias']).all())
    assert bool(torch.isfinite(obj['frame']).all())
    torch.save(obj,root/(name+'.centroid.private.pt'));sealed=True;sealed_at=time.time()
    ref=json.loads((refroot/(name+'_factorial.json')).read_text());rows=[r for r in ref['class_rows'] if r['model'] in ('MM','HH')];summary=[];queries=[]
    references=['MM','HH',matched_label]
    if raw:
        prior_result=json.loads((prior_centroid/prior_result_name).read_text())
        if compare_raw:
            rows.extend(dict(r,model='raw_centroid' if r['model']=='centroid' else r['model']) for r in prior_result['class_rows'] if r['model'] not in ('MM','HH'))
            references.extend(['teacher_centroid','teacher_ETF','raw_centroid'])
        else:
            rows.extend(dict(r,model='teacher_centroid' if r['model']=='centroid' else 'teacher_ETF') for r in prior_result['class_rows'] if r['model'] in ('centroid','teacher_ETF'))
            references.extend(['teacher_centroid','teacher_ETF'])
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));queries.append(dict(query=query,count=len(data['y'])))
        models=[('centroid',obj)] if compare_raw else [(matched_label,dict(model=teacher_model,frame=state['frame'])),('centroid',obj)]
        for label,model in models:
            _,rr,_=metrics(model,data,old,seen);rows.extend(dict(dataset=name,query=query,model=label,**r) for r in rr)
        for label in references+['centroid']:
            for group in ('all','old','new'):
                rr=[r for r in rows if r['query']==query and r['model']==label and (group=='all' or r['group']==group)]
                assert len(rr)==(seen if group=='all' else old if group=='old' else seen-old)
                summary.append(dict(dataset=name,query=query,model=label,group=group,BA=sum(r['recall'] for r in rr)/len(rr),CE=sum(r['CE'] for r in rr)/len(rr)))
    gates={}
    for query in ('Qfit','Qgate'):
        get=lambda label:{r['group']:r['BA'] for r in summary if r['query']==query and r['model']==label}
        gates[query]={label:select(get(label),get('centroid')) for label in references}
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,
        class_rows=rows,summaries=summary,gates=gates,primary_both_queries_pass=all(v['MM']['selected']=='candidate' for v in gates.values()),
        representation='raw_frozen_features' if raw else 'old_teacher_features',head_stats=stats,head_shape=list(head.shape),head_elements=head.numel(),head_bytes=head.numel()*head.element_size(),logit_scale=BETA,
        head_metric='squared_euclidean' if euclidean else 'cosine',center_estimator='coordinate_median' if median else 'weighted_mean',derived_bias_elements=seen if euclidean else 0,derived_bias_bytes=seen*head.element_size() if euclidean else 0,
        support_feature_forwards=sum(v['count'] for v in stats),query_sample_forwards=(1 if compare_raw else 2)*sum(v['count'] for v in queries),query_counts=queries,
        gradient_evaluations=0,adapter_updates=0,head_constructions=1,teacher_updates=0,encoder_updates=0,memory_updates=0,
        query_used_for_fitting=False,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,
        scope='Structural reference: fitted centroid head; representation recorded explicitly. MM/HH have different post-increment representation. No architecture-matched adapter improvement claim.'))


if __name__=='__main__':main()
