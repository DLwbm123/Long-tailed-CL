"""Fixed algebraic score stages for raw and centered mean heads."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA,DT,features
from run_real_query_node import load,save


def stages(z,raw,head,center):
    norm=(z-center).norm(dim=1);assert bool((norm>1e-12).all())
    a=BETA*z@raw;b=BETA*z@head;c=b-BETA*center@head;d=c/norm[:,None]
    return dict(A_raw=a,B_head=b,C_bias=c,D_scaled=d),1/norm


def unit():
    z=torch.eye(2,dtype=DT);head=torch.tensor([[.8,-.6],[.6,.8]],dtype=DT);center=torch.tensor([.1,.2],dtype=DT)
    ss,scale=stages(z,z,head,center)
    direct=BETA*((z-center)/(z-center).norm(dim=1,keepdim=True))@head
    assert torch.allclose(ss['D_scaled'],direct,atol=1e-12,rtol=0)
    assert torch.equal(ss['C_bias'].argmax(1),ss['D_scaled'].argmax(1))
    zero,_=stages(z,z,z,torch.zeros(2,dtype=DT));assert all(torch.equal(v,zero['A_raw']) for v in zero.values())
    c=torch.tensor([[2.,1.],[1.,2.]],dtype=DT);loss=-c.log_softmax(1)[:,0];scaled=-(2*c).log_softmax(1)[:,0]
    assert scaled[0]<loss[0] and scaled[1]>loss[1] and torch.equal(c.argmax(1),(2*c).argmax(1))


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    prior=Path(cfg['prior_root']);rawroot=Path(cfg['raw_centroid_root']);croot=Path(cfg['centered_root']);sealed=False
    allowed={prior:{name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'},rawroot:{name+'.centroid.private.pt'},croot:{name+'.centered.private.pt',name+'_centered.json'}}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        for base,names in allowed.items():
            if p.is_relative_to(base):
                assert p.name in names,'Score diagnostic input boundary'
                if base==prior:assert sealed,'Query before fixed stages'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','S_new.features.private.pt','E.features.private.pt','teacher.private.pt','Qfit.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    raw=load(rawroot/(name+'.centroid.private.pt'));current=load(croot/(name+'.centered.private.pt'))
    for obj in (raw,current):
        assert obj['frame'].shape==(1536,seen) and 'bias' not in obj
        assert all(bool((v==0).all()) for v in obj['model'])
        assert bool(torch.isfinite(obj['frame']).all()) and torch.allclose(obj['frame'].norm(dim=0),torch.ones(seen,dtype=DT),atol=1e-12,rtol=0)
    center=current['center'];assert center.shape==(1536,) and bool(torch.isfinite(center).all());sealed=True;sealed_at=time.time()
    ref=json.loads((croot/(name+'_centered.json')).read_text())
    classes=[];summary=[];pairs=[];switches=[];scale_rows=[];counts=[]
    edges=[('A_raw','B_head'),('B_head','C_bias'),('C_bias','D_scaled'),('A_raw','D_scaled')]
    for query in ('Qfit','Qgate'):
        save(root/(job['id']+'.progress.json'),dict(phase='fixed_score_stages',query=query));data=load(prior/(name+'.'+query+'.features.private.pt'));y=data['y'];counts.append(dict(query=query,count=len(y)))
        assert y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        with torch.no_grad():
            z=torch.cat([features(data['x'][i:i+256],raw['model']) for i in range(0,len(y),256)]);ss,scale=stages(z,raw['frame'],current['frame'],center)
            assert all(bool(torch.isfinite(v).all()) for v in ss.values())
            pred={k:v.argmax(1) for k,v in ss.items()};ce={k:-v.log_softmax(1)[torch.arange(len(y)),y] for k,v in ss.items()};confidence={k:v.softmax(1).max(1).values for k,v in ss.items()}
            assert torch.equal(pred['C_bias'],pred['D_scaled']),'Positive scaling changed prediction'
        for model in ss:
            for c in range(seen):
                mask=y==c;pr=pred[model][mask];n=int(mask.sum());assert n>0
                row=dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',count=n,correct=int((pr==c).sum()),recall=float((pr==c).to(DT).mean()),CE=float(ce[model][mask].mean()),mean_max_probability=float(confidence[model][mask].mean()),errors_to_old=int(((pr!=c)&(pr<old)).sum()),errors_to_new=int(((pr!=c)&(pr>=old)).sum()));classes.append(row)
                if model in ('A_raw','D_scaled'):
                    label='raw_centroid' if model=='A_raw' else 'centered_centroid';rr=next(v for v in ref['class_rows'] if v['model']==label and v['query']==query and v['class_index']==c)
                    assert all(rr[k]==row[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(rr['CE']-row['CE'])<1e-10,'Published aggregate mismatch'
            for group in ('all','old','new'):
                rr=[r for r in classes if r['query']==query and r['model']==model and (group=='all' or r['group']==group)]
                summary.append(dict(dataset=name,query=query,model=model,group=group,BA=sum(r['recall'] for r in rr)/len(rr),CE=sum(r['CE'] for r in rr)/len(rr),macro_mean_max_probability=sum(r['mean_max_probability'] for r in rr)/len(rr)))
        for akey,bkey in edges:
            a=pred[akey];b=pred[bkey];ca=a==y;cb=b==y;changed=a!=b
            for c in range(seen):
                m=y==c;row=dict(dataset=name,query=query,from_stage=akey,to_stage=bkey,class_index=c,group='old' if c<old else 'new',count=int(m.sum()),rescued=int((m&~ca&cb).sum()),harmed=int((m&ca&~cb).sum()),both_correct=int((m&ca&cb).sum()),both_wrong=int((m&~ca&~cb).sum()),changed=int((m&changed).sum()),wrong_to_other_wrong=int((m&~ca&~cb&changed).sum()))
                assert row['rescued']+row['harmed']+row['both_correct']+row['both_wrong']==row['count']
                assert row['changed']==row['rescued']+row['harmed']+row['wrong_to_other_wrong']
                assert row['rescued']-row['harmed']==int((m&cb).sum())-int((m&ca).sum());pairs.append(row)
            for c,aa,bb in sorted(set(zip(y[changed].tolist(),a[changed].tolist(),b[changed].tolist()))):
                switches.append(dict(dataset=name,query=query,from_stage=akey,to_stage=bkey,class_index=c,group='old' if c<old else 'new',from_prediction=aa,to_prediction=bb,from_group='old' if aa<old else 'new',to_group='old' if bb<old else 'new',outcome='harmed' if aa==c else 'rescued' if bb==c else 'wrong_to_other_wrong',count=int(((y==c)&(a==aa)&(b==bb)).sum())))
            assert sum(r['count'] for r in switches if r['query']==query and r['from_stage']==akey and r['to_stage']==bkey)==int(changed.sum())
        for c in range(seen):
            m=y==c;s=scale[m]
            scale_rows.append(dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',count=int(m.sum()),scale_min=float(s.min()),scale_median=float(torch.quantile(s,.5)),scale_mean=float(s.mean()),scale_max=float(s.max()),CE_delta_scaling=float((ce['D_scaled'][m]-ce['C_bias'][m]).mean())))
    n=sum(v['count'] for v in counts)
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,class_rows=classes,summaries=summary,paired_rows=pairs,switch_rows=switches,scale_rows=scale_rows,query_counts=counts,published_aggregates_matched=True,scaling_predictions_identical=True,shared_feature_sample_forwards=n,model_logit_sample_evaluations=2*n,decision_sample_evaluations=4*n,head_fits=0,head_constructions=0,center_fits=0,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,support_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Fixed ordered score decomposition, descriptive development evidence only. No tuning, stage selection, independent validation or independent causal contributions.'))


if __name__=='__main__':main()
