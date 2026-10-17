"""Fixed old/new block factorial for frozen mean and median heads."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA,DT,features
from run_real_query_node import load,save


def blocks(a,b,old):
    return dict(mean_mean=a,median_mean=torch.cat((b[:,:old],a[:,old:]),1),
                mean_median=torch.cat((a[:,:old],b[:,old:]),1),median_median=b)


def unit():
    a=torch.tensor([[3.,2.],[3.,2.],[2.,3.]],dtype=DT)
    b=torch.tensor([[1.,2.],[3.,4.],[4.,3.]],dtype=DT)
    ss=blocks(a,b,1)
    assert ss['median_mean'].argmax(1).tolist()==[1,0,0]
    assert ss['mean_median'].argmax(1).tolist()==[0,1,1]
    z=torch.eye(3,dtype=DT)
    for k,f in blocks(a,b,1).items():
        # Column blocks apply to scores and head matrices identically.
        assert torch.equal(z@f,blocks(z@a,z@b,1)[k])


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    prior=Path(cfg['prior_root']);rawroot=Path(cfg['raw_centroid_root']);medroot=Path(cfg['median_root']);sealed=False
    allowed={prior:{name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'},rawroot:{name+'.centroid.private.pt'},medroot:{name+'.centroid.private.pt',name+'_median.json'}}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        for base,names in allowed.items():
            if p.is_relative_to(base):
                assert p.name in names,'Block diagnostic input boundary'
                if base==prior:assert sealed,'Query before fixed block construction'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','S_new.features.private.pt','E.features.private.pt','teacher.private.pt','Qfit.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    raw=load(rawroot/(name+'.centroid.private.pt'));med=load(medroot/(name+'.centroid.private.pt'))
    for obj in (raw,med):
        assert obj['frame'].shape==(1536,seen) and 'bias' not in obj
        assert all(bool((v==0).all()) for v in obj['model'])
        assert bool(torch.isfinite(obj['frame']).all())
        assert torch.allclose(obj['frame'].norm(dim=0),torch.ones(seen,dtype=DT),atol=1e-12,rtol=0)
    heads=blocks(raw['frame'],med['frame'],old);sealed=True;sealed_at=time.time()
    ref=json.loads((medroot/(name+'_median.json')).read_text())
    classes=[];summary=[];paired=[];switches=[];effects=[];counts=[]
    for query in ('Qfit','Qgate'):
        save(root/(job['id']+'.progress.json'),dict(phase='fixed_block_diagnostic',query=query))
        data=load(prior/(name+'.'+query+'.features.private.pt'));y=data['y'];counts.append(dict(query=query,count=len(y)))
        assert y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        with torch.no_grad():
            z=torch.cat([features(data['x'][i:i+256],raw['model']) for i in range(0,len(y),256)])
            ss=blocks(BETA*z@heads['mean_mean'],BETA*z@heads['median_median'],old)
            assert all(bool(torch.isfinite(v).all()) for v in ss.values())
            pred={k:v.argmax(1) for k,v in ss.items()};ce={k:-v.log_softmax(1)[torch.arange(len(y)),y] for k,v in ss.items()}
        for model in ss:
            for c in range(seen):
                mask=y==c;n=int(mask.sum());assert n>0;pr=pred[model][mask]
                row=dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',count=n,correct=int((pr==c).sum()),recall=float((pr==c).to(DT).mean()),CE=float(ce[model][mask].mean()),errors_to_old=int(((pr!=c)&(pr<old)).sum()),errors_to_new=int(((pr!=c)&(pr>=old)).sum()))
                classes.append(row)
                if model in ('mean_mean','median_median'):
                    label='raw_centroid' if model=='mean_mean' else 'centroid'
                    rr=next(v for v in ref['class_rows'] if v['model']==label and v['query']==query and v['class_index']==c)
                    assert all(rr[k]==row[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(rr['CE']-row['CE'])<1e-10,'Published aggregate mismatch'
            for group in ('all','old','new'):
                rr=[v for v in classes if v['query']==query and v['model']==model and (group=='all' or v['group']==group)]
                summary.append(dict(dataset=name,query=query,model=model,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
            if model=='mean_mean':continue
            a=pred['mean_mean'];b=pred[model];ca=a==y;cb=b==y;changed=a!=b
            for c in range(seen):
                m=y==c;row=dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',count=int(m.sum()),rescued=int((m&~ca&cb).sum()),harmed=int((m&ca&~cb).sum()),both_correct=int((m&ca&cb).sum()),both_wrong=int((m&~ca&~cb).sum()),changed=int((m&changed).sum()),wrong_to_other_wrong=int((m&~ca&~cb&changed).sum()))
                assert row['rescued']+row['harmed']+row['both_correct']+row['both_wrong']==row['count']
                assert row['changed']==row['rescued']+row['harmed']+row['wrong_to_other_wrong']
                assert row['rescued']-row['harmed']==int((m&cb).sum())-int((m&ca).sum());paired.append(row)
            for c,aa,bb in sorted(set(zip(y[changed].tolist(),a[changed].tolist(),b[changed].tolist()))):
                switches.append(dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',mean_prediction=aa,block_prediction=bb,from_group='old' if aa<old else 'new',to_group='old' if bb<old else 'new',outcome='harmed' if aa==c else 'rescued' if bb==c else 'wrong_to_other_wrong',count=int(((y==c)&(a==aa)&(b==bb)).sum())))
            assert sum(v['count'] for v in switches if v['query']==query and v['model']==model)==int(changed.sum())
        for group in ('all','old','new'):
            for metric in ('BA','CE'):
                v={r['model']:r[metric] for r in summary if r['query']==query and r['group']==group};a=v['mean_mean'];b=v['median_mean'];c=v['mean_median'];d=v['median_median']
                effects.append(dict(dataset=name,query=query,group=group,metric=metric,old_block_at_mean_new=b-a,new_block_at_mean_old=c-a,old_block_at_median_new=d-c,new_block_at_median_old=d-b,interaction=d-b-c+a,total=d-a))
                assert abs((b-a)+(c-a)+(d-b-c+a)-(d-a))<1e-12
    n=sum(v['count'] for v in counts)
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,class_rows=classes,summaries=summary,paired_rows=paired,switch_rows=switches,effects=effects,query_counts=counts,published_aggregates_matched=True,shared_feature_sample_forwards=n,model_logit_sample_evaluations=2*n,decision_sample_evaluations=4*n,hybrid_head_compositions=2,hybrid_head_elements=2*1536*seen,hybrid_head_bytes=2*1536*seen*8,head_fits=0,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,support_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Fixed 2x2 old/new blocks, descriptive development ablation only. No model selection, fitting, independent validation or causal population claim.'))


if __name__=='__main__':main()
