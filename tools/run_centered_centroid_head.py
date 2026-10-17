"""Class-balanced support centering before the fixed mean cosine head."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA,DT,features
from query_node_weights import select
from run_frozen_centroid_head import centroid
from run_real_query_node import load,save


def centered(z,center):
    delta=z-center;norm=delta.norm(dim=1,keepdim=True)
    assert bool(torch.isfinite(delta).all()) and bool((norm>1e-12).all()),'Zero centered residual must stop'
    return delta/norm


def unit():
    z=torch.eye(2,dtype=DT);center=z.mean(0);expected=torch.tensor([[1.,-1.],[-1.,1.]],dtype=DT)/2**.5
    assert torch.allclose(centered(z,center),expected)
    assert torch.allclose(centered(z,torch.zeros(2,dtype=DT)),z)
    assert torch.allclose(torch.stack([z[0],z[1]]).mean(0),torch.stack([z[0:1].mean(0),z[1:].repeat(3,1).mean(0)]).mean(0))
    try:centered(z,z[0])
    except AssertionError:pass
    else:raise AssertionError('Zero residual accepted')


def geometry(head):
    cos=head.T@head;v=cos[torch.triu(torch.ones_like(cos,dtype=torch.bool),diagonal=1)]
    return dict(pair_cosine_min=float(v.min()),pair_cosine_mean=float(v.mean()),pair_cosine_max=float(v.max()))


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    prior=Path(cfg['prior_root']);medroot=Path(cfg['median_root']);sealed=False
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in {name+'.teacher.private.pt',name+'.S_new.features.private.pt'} or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Center fit/query boundary'
        if p.is_relative_to(medroot):assert sealed and p.name==name+'_median.json','Reference before seal'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    state=load(prior/(name+'.teacher.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'));bank=load(root/(name+'.herding.private.pt'))
    model=tuple(torch.zeros_like(v) for v in state['teacher']);support=[];means=[]
    save(root/(job['id']+'.progress.json'),dict(phase='class_balanced_center',dataset=name))
    with torch.no_grad():
        for c in range(seen):
            if c<old:x,p=bank[c]
            else:x=new['x'][new['y']==c];p=torch.full((len(x),),1/len(x),dtype=DT)
            assert bool((p>=0).all()) and abs(float(p.sum())-1)<1e-12
            z=torch.cat([features(x[i:i+256],model) for i in range(0,len(x),256)]);support.append((z,p));means.append((z*p[:,None]).sum(0))
        means=torch.stack(means);center=means.mean(0);assert center.shape==(1536,) and bool(torch.isfinite(center).all())
        center_norm=float(center.norm());heads=[];rawheads=[];stats=[]
        for c,(z,p) in enumerate(support):
            rawq,rawnorm=centroid(z,p);transformed=centered(z,center);q,n=centroid(transformed,p);heads.append(q);rawheads.append(rawq)
            residual=(z-center).norm(dim=1)
            stats.append(dict(class_index=c,count=len(z),raw_mean_norm=rawnorm,centered_mean_norm=n,raw_head_common_direction_cosine=float(rawq@center/center_norm) if center_norm>1e-12 else None,residual_norm_min=float(residual.min()),residual_norm_mean=float(residual.mean()),residual_norm_max=float(residual.max())))
        head=torch.stack(heads,1);rawhead=torch.stack(rawheads,1)
    assert head.shape==(1536,seen) and bool(torch.isfinite(head).all())
    torch.save(dict(model=model,frame=head,center=center),root/(name+'.centered.private.pt'));sealed=True;sealed_at=time.time()
    ref=json.loads((medroot/(name+'_median.json')).read_text());rows=[dict(r,model='coordinate_median' if r['model']=='centroid' else r['model']) for r in ref['class_rows']]
    references=['raw_centroid','MM','HH','identity_ETF','teacher_centroid','teacher_ETF','coordinate_median'];summaries=[];counts=[];gates={}
    for query in ('Qfit','Qgate'):
        save(root/(job['id']+'.progress.json'),dict(phase='sealed_query',query=query));data=load(prior/(name+'.'+query+'.features.private.pt'));y=data['y'];counts.append(dict(query=query,count=len(y)))
        assert y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        with torch.no_grad():
            z=torch.cat([features(data['x'][i:i+256],model) for i in range(0,len(y),256)]);ss=BETA*centered(z,center)@head;assert bool(torch.isfinite(ss).all())
            pred=ss.argmax(1);ce=-ss.log_softmax(1)[torch.arange(len(y)),y]
        for c in range(seen):
            mask=y==c;pr=pred[mask];n=int(mask.sum());assert n>0
            rows.append(dict(dataset=name,query=query,model='centered_centroid',class_index=c,group='old' if c<old else 'new',count=n,correct=int((pr==c).sum()),recall=float((pr==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((pr!=c)&(pr<old)).sum()),errors_to_new=int(((pr!=c)&(pr>=old)).sum())))
        for label in references+['centered_centroid']:
            for group in ('all','old','new'):
                rr=[r for r in rows if r['query']==query and r['model']==label and (group=='all' or r['group']==group)]
                assert len(rr)==(seen if group=='all' else old if group=='old' else seen-old)
                summaries.append(dict(dataset=name,query=query,model=label,group=group,BA=sum(r['recall'] for r in rr)/len(rr),CE=sum(r['CE'] for r in rr)/len(rr)))
        get=lambda label:{r['group']:r['BA'] for r in summaries if r['query']==query and r['model']==label}
        gates[query]={label:select(get(label),get('centered_centroid')) for label in references}
    support_count=sum(r['count'] for r in stats);query_count=sum(r['count'] for r in counts)
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,class_rows=rows,summaries=summaries,gates=gates,primary_reference='raw_centroid',primary_both_queries_pass=all(g['raw_centroid']['selected']=='candidate' for g in gates.values()),head_stats=stats,geometry=dict(center_norm=center_norm,raw=geometry(rawhead),centered=geometry(head)),head_shape=list(head.shape),head_elements=head.numel(),center_elements=center.numel(),total_fitted_elements=head.numel()+center.numel(),total_fitted_bytes=(head.numel()+center.numel())*head.element_size(),head_constructions=1,center_fits=1,derived_bias_elements=0,support_feature_forwards=support_count,query_sample_forwards=query_count,centered_support_transforms=support_count,centered_query_transforms=query_count,model_logit_sample_evaluations=query_count,query_counts=counts,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,query_used_for_fitting=False,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Fixed class-balanced support centering; development comparison only. Center and heads are data-fitted, not zero fitting. Geometry expansion is not classification improvement.'))


if __name__=='__main__':main()
