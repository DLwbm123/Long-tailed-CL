"""Fixed teacher features with a support-derived cosine centroid head."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT,BETA,features
from query_node_weights import select
from run_real_query_node import load,metrics,save


def centroid(z,p):
    assert bool(torch.isfinite(z).all()) and bool((p>=0).all()) and abs(float(p.sum())-1)<1e-12
    mean=(z*p[:,None]).sum(0);norm=mean.norm();assert float(norm)>1e-12
    return mean/norm,float(norm)


def unit():
    z=torch.eye(2,dtype=DT);q,n=centroid(z,torch.tensor([.75,.25],dtype=DT))
    assert torch.allclose(q,torch.tensor([3.,1.],dtype=DT)/(10**.5)) and abs(n-(10**.5)/4)<1e-12
    try:centroid(torch.tensor([[1.,0.],[-1.,0.]],dtype=DT),torch.ones(2,dtype=DT)/2)
    except AssertionError:pass
    else:raise AssertionError('Zero centroid must stop')


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];prior=Path(cfg['prior_root']);refroot=Path(cfg['bank_root']);sealed=False
    allowed={name+'.teacher.private.pt',name+'.S_new.features.private.pt'}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in allowed or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Centroid fit/query boundary'
        if p.is_relative_to(refroot):assert sealed and p.name==name+'_factorial.json','Historical aggregate only after seal'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    state=load(prior/(name+'.teacher.private.pt'));bank=load(root/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    save(root/(job['id']+'.progress.json'),dict(phase='construct_centroid_head',dataset=name));heads=[];stats=[]
    with torch.no_grad():
        for c in range(seen):
            if c<old:x,p=bank[c]
            else:x=new['x'][new['y']==c];p=torch.full((len(x),),1/len(x),dtype=DT)
            z=torch.cat([features(x[i:i+256],state['teacher']) for i in range(0,len(x),256)])
            q,n=centroid(z,p);heads.append(q);stats.append(dict(class_index=c,count=len(x),mean_norm=n))
    head=torch.stack(heads,1);assert head.shape==(state['frame'].shape[0],seen)
    obj=dict(model=state['teacher'],frame=head);torch.save(obj,root/(name+'.centroid.private.pt'));sealed=True;sealed_at=time.time()
    ref=json.loads((refroot/(name+'_factorial.json')).read_text());rows=[r for r in ref['class_rows'] if r['model'] in ('MM','HH')];summary=[];queries=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));queries.append(dict(query=query,count=len(data['y'])))
        for label,model in [('teacher_ETF',dict(model=state['teacher'],frame=state['frame'])),('centroid',obj)]:
            _,rr,_=metrics(model,data,old,seen);rows.extend(dict(dataset=name,query=query,model=label,**r) for r in rr)
        for label in ('MM','HH','teacher_ETF','centroid'):
            for group in ('all','old','new'):
                rr=[r for r in rows if r['query']==query and r['model']==label and (group=='all' or r['group']==group)]
                assert len(rr)==(seen if group=='all' else old if group=='old' else seen-old)
                summary.append(dict(dataset=name,query=query,model=label,group=group,BA=sum(r['recall'] for r in rr)/len(rr),CE=sum(r['CE'] for r in rr)/len(rr)))
    gates={}
    for query in ('Qfit','Qgate'):
        get=lambda label:{r['group']:r['BA'] for r in summary if r['query']==query and r['model']==label}
        gates[query]={label:select(get(label),get('centroid')) for label in ('MM','HH','teacher_ETF')}
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,
        class_rows=rows,summaries=summary,gates=gates,primary_both_queries_pass=all(v['MM']['selected']=='candidate' for v in gates.values()),
        head_stats=stats,head_shape=list(head.shape),head_elements=head.numel(),head_bytes=head.numel()*head.element_size(),logit_scale=BETA,
        support_feature_forwards=sum(v['count'] for v in stats),query_sample_forwards=2*sum(v['count'] for v in queries),query_counts=queries,
        gradient_evaluations=0,adapter_updates=0,head_constructions=1,teacher_updates=0,encoder_updates=0,memory_updates=0,
        query_used_for_fitting=False,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,
        scope='Structural reference: fitted centroid head on fixed teacher features. Not architecture-matched adapter improvement; MM/HH have different post-increment representation.'))


if __name__=='__main__':main()
