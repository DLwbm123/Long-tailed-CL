"""Read-only paired diagnostics for two already frozen centroid heads."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA,DT,features
from run_real_query_node import load,save


def decompose(s,n,a,b):
    i=torch.arange(len(a));base=s[i,b]-s[i,a]
    scale=(n[b]-1)*s[i,b]-(n[a]-1)*s[i,a]
    bias=-BETA/2*(n[b].square()-n[a].square())
    return base,scale,bias


def unit():
    s=torch.tensor([[3.,2.],[1.,2.],[2.,3.]],dtype=DT);n=torch.tensor([.5,.9],dtype=DT)
    e=s*n-BETA/2*n.square();a=s.argmax(1);b=e.argmax(1)
    terms=decompose(s,n,a,b);i=torch.arange(len(a))
    assert torch.allclose(sum(terms),e[i,b]-e[i,a])
    y=torch.tensor([0,0,1]);ca=a==y;cb=b==y
    assert int((~ca&cb).sum())==1 and int((ca&~cb).sum())==1
    assert int((ca&cb).sum())+int((~ca&~cb).sum())+int((ca&~cb).sum())+int((~ca&cb).sum())==len(y)


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    prior=Path(cfg['prior_root']);rawroot=Path(cfg['raw_centroid_root']);eroot=Path(cfg['euclidean_root'])
    allowed={prior:{name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'},rawroot:{name+'.centroid.private.pt'},eroot:{name+'.centroid.private.pt',name+'_euclidean.json'}}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        for base,names in allowed.items():
            if p.is_relative_to(base):assert p.name in names,'Paired audit input boundary'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','S_new.features.private.pt','E.features.private.pt','teacher.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    raw=load(rawroot/(name+'.centroid.private.pt'));eu=load(eroot/(name+'.centroid.private.pt'))
    ref=json.loads((eroot/(name+'_euclidean.json')).read_text())
    assert raw['frame'].shape==eu['frame'].shape==(1536,seen)
    assert all(bool((v==0).all()) for o in (raw,eu) for v in o['model'])
    n=eu['frame'].norm(dim=0)
    assert bool(torch.isfinite(n).all()) and bool((n>0).all()) and bool((n<=1+1e-12).all())
    assert torch.allclose(raw['frame'].norm(dim=0),torch.ones(seen,dtype=DT),atol=1e-12,rtol=0)
    assert torch.allclose(eu['frame'],raw['frame']*n,atol=1e-12,rtol=0)
    assert torch.allclose(eu['bias'],-BETA/2*n.square(),atol=1e-12,rtol=0)
    classes=[];switches=[];groups=[];queries=[];max_residual=0.
    for query in ('Qfit','Qgate'):
        save(root/(job['id']+'.progress.json'),dict(phase='paired_diagnostic',query=query))
        data=load(prior/(name+'.'+query+'.features.private.pt'));y=data['y'];queries.append(dict(query=query,count=len(y)))
        with torch.no_grad():
            chunks=[features(data['x'][i:i+256],raw['model']) for i in range(0,len(y),256)]
            s=torch.cat([BETA*z@raw['frame'] for z in chunks]);e=torch.cat([BETA*z@eu['frame']+eu['bias'] for z in chunks])
            assert bool(torch.isfinite(s).all()) and bool(torch.isfinite(e).all())
            a=s.argmax(1);b=e.argmax(1);ca=a==y;cb=b==y;changed=a!=b
            base,scale,bias=decompose(s,n,a,b);idx=torch.arange(len(y));margin=e[idx,b]-e[idx,a]
            residual=float((base+scale+bias-margin).abs().max());assert residual<1e-10;max_residual=max(max_residual,residual)
            ce_s=-s.log_softmax(1)[idx,y];ce_e=-e.log_softmax(1)[idx,y]
        for c in range(seen):
            mask=y==c;count=int(mask.sum());assert count>0
            rr=dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',count=count,cosine_correct=int((mask&ca).sum()),euclidean_correct=int((mask&cb).sum()),rescued=int((mask&~ca&cb).sum()),harmed=int((mask&ca&~cb).sum()),both_correct=int((mask&ca&cb).sum()),both_wrong=int((mask&~ca&~cb).sum()),changed=int((mask&changed).sum()),wrong_to_other_wrong=int((mask&~ca&~cb&changed).sum()))
            assert rr['rescued']-rr['harmed']==rr['euclidean_correct']-rr['cosine_correct']
            assert rr['changed']==rr['rescued']+rr['harmed']+rr['wrong_to_other_wrong']
            rr['delta_recall_pp']=100*(rr['rescued']-rr['harmed'])/count;classes.append(rr)
            for model,correct,ce in [('raw_centroid',rr['cosine_correct'],ce_s),('centroid',rr['euclidean_correct'],ce_e)]:
                row=next(r for r in ref['class_rows'] if r['model']==model and r['query']==query and r['class_index']==c)
                assert row['count']==count and row['correct']==correct and abs(row['CE']-float(ce[mask].mean()))<1e-10,'Published aggregate mismatch'
        for c,aa,bb in sorted(set(zip(y[changed].tolist(),a[changed].tolist(),b[changed].tolist()))):
            mask=(y==c)&(a==aa)&(b==bb);count=int(mask.sum())
            row=dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',cosine_prediction=aa,euclidean_prediction=bb,from_group='old' if aa<old else 'new',to_group='old' if bb<old else 'new',outcome='harmed' if aa==c else 'rescued' if bb==c else 'wrong_to_other_wrong',count=count,from_mean_norm=float(n[aa]),to_mean_norm=float(n[bb]),scale_favors_switch=int((scale[mask]>0).sum()),bias_favors_switch=int((bias[mask]>0).sum()),both_terms_favor_switch=int(((scale[mask]>0)&(bias[mask]>0)).sum()))
            for key,values in [('cosine_margin',base),('scaling_term',scale),('norm_bias_term',bias),('euclidean_margin',margin)]:
                row[key+'_mean']=float(values[mask].mean());row[key+'_min']=float(values[mask].min());row[key+'_max']=float(values[mask].max())
            switches.append(row)
        for group in ('all','old','new'):
            rr=[r for r in classes if r['query']==query and (group=='all' or r['group']==group)]
            groups.append(dict(dataset=name,query=query,group=group,classes=len(rr),**{k:sum(r[k] for r in rr) for k in ('count','rescued','harmed','changed','wrong_to_other_wrong')},delta_BA_pp=sum(r['delta_recall_pp'] for r in rr)/len(rr)))
        assert sum(r['count'] for r in switches if r['query']==query)==int(changed.sum())
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,class_rows=classes,switch_rows=switches,group_rows=groups,query_counts=queries,published_aggregates_matched=True,max_decomposition_residual=max_residual,shared_feature_sample_forwards=sum(v['count'] for v in queries),model_logit_sample_evaluations=2*sum(v['count'] for v in queries),head_constructions=0,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,support_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Descriptive paired development diagnostic only; no new candidate or model selection; no sample identities or per-sample outputs.'))


if __name__=='__main__':main()
