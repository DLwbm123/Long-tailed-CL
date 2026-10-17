"""Exact single-support deletion stress test on reused development queries."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA,DT,features
from run_frozen_centroid_head import centroid
from run_centroid_support_sensitivity import unit as support_unit
from run_real_query_node import load,save


def replacement_predictions(scores,c,values):
    other=scores.clone();other[:,c]=-torch.inf;best,index=other.max(1)
    wins=(values>best[None,:])|((values==best[None,:])&(c<index[None,:]))
    return torch.where(wins,torch.full_like(wins,c,dtype=torch.long),index[None,:])


def unit():
    support_unit();s=torch.tensor([[2.,2.,1.],[1.,2.,2.]],dtype=DT);values=torch.tensor([[2.,2.],[1.,0.]],dtype=DT)
    for c in range(3):
        fast=replacement_predictions(s,c,values)
        for i in range(2):
            direct=s.clone();direct[:,c]=values[i];assert torch.equal(fast[i],direct.argmax(1))


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    prior=Path(cfg['prior_root']);rawroot=Path(cfg['raw_centroid_root'])
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name==name+'.S_new.features.private.pt' or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Support then query boundary'
        if p.is_relative_to(rawroot):assert p.name in {name+'.centroid.private.pt',name+'.herding.private.pt'} or (sealed and p.name==name+'_raw.json'),'Frozen reference boundary'
    sys.addaudithook(audit)
    for suffix in ('Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt','S_old.features.private.pt','teacher.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    raw=load(rawroot/(name+'.centroid.private.pt'));bank=load(rawroot/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    assert raw['frame'].shape==(1536,seen) and all(bool((v==0).all()) for v in raw['model'])
    directions=[];support=[]
    with torch.no_grad():
        for c in range(seen):
            if c<old:x,p=bank[c]
            else:x=new['x'][new['y']==c];p=torch.full((len(x),),1/len(x),dtype=DT)
            assert len(x)==lock['support_counts'][name][c] and torch.allclose(p,torch.full_like(p,1/len(x)),atol=1e-12,rtol=0)
            z=torch.cat([features(x[i:i+256],raw['model']) for i in range(0,len(x),256)]);q,n=centroid(z,p)
            assert torch.allclose(q,raw['frame'][:,c],atol=1e-12,rtol=0)
            if len(x)==1:heads=torch.empty((0,len(q)),dtype=DT);empty=1;zero=0
            else:
                means=(q*n-p[:,None]*z)/(1-p[:,None]);norm=means.norm(dim=1);valid=norm>1e-12
                heads=means[valid]/norm[valid,None];empty=0;zero=int((~valid).sum())
            assert bool(torch.isfinite(heads).all())
            directions.append(heads);support.append(dict(class_index=c,count=len(x),defined_deletions=len(heads),undefined_empty_support=empty,undefined_zero_mean=zero))
    sealed=True;sealed_at=time.time();save(root/(job['id']+'.progress.json'),dict(phase='directions_sealed',support=support,sealed_at=sealed_at))
    ref=json.loads((rawroot/(name+'_raw.json')).read_text());pairs=[];perturbations=[];global_rows=[];queries=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));y=data['y'];queries.append(dict(query=query,count=len(y)))
        with torch.no_grad():
            z=torch.cat([features(data['x'][i:i+256],raw['model']) for i in range(0,len(y),256)])
            scores=torch.cat([BETA*z[i:i+256]@raw['frame'] for i in range(0,len(y),256)]);base=scores.argmax(1);base_correct=base==y;ce=-scores.log_softmax(1)[torch.arange(len(y)),y]
            for c in range(seen):
                row=next(r for r in ref['class_rows'] if r['model']=='centroid' and r['query']==query and r['class_index']==c);mask=y==c
                assert int(mask.sum())==row['count'] and int((mask&base_correct).sum())==row['correct'] and abs(float(ce[mask].mean())-row['CE'])<1e-10
            any_change=torch.zeros(len(y),dtype=torch.bool);any_harm=any_change.clone();any_rescue=any_change.clone()
            for k,heads in enumerate(directions):
                if not len(heads):continue
                values=BETA*heads@z.T;assert bool(torch.isfinite(values).all())
                pred=replacement_predictions(scores,k,values);correct=pred==y[None,:];changed=pred!=base[None,:]
                rescued=correct&~base_correct[None,:];harmed=~correct&base_correct[None,:]
                any_change|=changed.any(0);any_harm|=harmed.any(0);any_rescue|=rescued.any(0);deltas=[]
                for c in range(seen):
                    mask=y==c;count=int(mask.sum());rr=dict(dataset=name,query=query,perturbed_class=k,perturbed_group='old' if k<old else 'new',true_class=c,true_group='old' if c<old else 'new',query_count=count,defined_deletions=len(heads),changed_events=int(changed[:,mask].sum()),rescued_events=int(rescued[:,mask].sum()),harmed_events=int(harmed[:,mask].sum()),wrong_to_other_wrong_events=int((changed[:,mask]&~correct[:,mask]&~base_correct[None,mask]).sum()),queries_with_any_change=int(changed[:,mask].any(0).sum()),queries_with_any_rescue=int(rescued[:,mask].any(0).sum()),queries_with_any_harm=int(harmed[:,mask].any(0).sum()))
                    assert rr['changed_events']==rr['rescued_events']+rr['harmed_events']+rr['wrong_to_other_wrong_events'];pairs.append(rr)
                    deltas.append(correct[:,mask].to(DT).mean(1)-base_correct[mask].to(DT).mean())
                delta=torch.stack(deltas,1)
                for group,cs in [('all',range(seen)),('old',range(old)),('new',range(old,seen))]:
                    d=delta[:,list(cs)].mean(1)*100
                    perturbations.append(dict(dataset=name,query=query,perturbed_class=k,group=group,defined_deletions=len(heads),delta_BA_pp_min=float(d.min()),delta_BA_pp_mean=float(d.mean()),delta_BA_pp_max=float(d.max())))
            for c in range(seen):
                mask=y==c;global_rows.append(dict(dataset=name,query=query,true_class=c,group='old' if c<old else 'new',query_count=int(mask.sum()),baseline_correct=int((mask&base_correct).sum()),queries_with_any_change=int((mask&any_change).sum()),queries_with_any_harm=int((mask&any_harm).sum()),queries_with_any_rescue=int((mask&any_rescue).sum()),queries_unchanged_over_defined_deletions=int((mask&~any_change).sum())))
        save(root/(job['id']+'.progress.json'),dict(phase='query_diagnostic_complete',query=query))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,support_rows=support,pair_rows=pairs,perturbation_rows=perturbations,global_rows=global_rows,query_counts=queries,reference_aggregates_matched=True,reference_class_means_recomputed=seen,delete_one_attempts=sum(r['count'] for r in support),defined_delete_one_class_vectors=sum(len(h) for h in directions),support_feature_sample_forwards=sum(r['count'] for r in support),query_feature_sample_forwards=sum(r['count'] for r in queries),baseline_full_head_sample_evaluations=sum(r['count'] for r in queries),replaced_class_score_evaluations=sum(len(h) for h in directions)*sum(r['count'] for r in queries),head_artifacts_constructed=0,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Exhaustive valid single-support perturbation diagnostic; no best deletion selected or deployed, no independent accuracy estimate. Undefined deletion families excluded explicitly, not certified stable.'))


if __name__=='__main__':main()
