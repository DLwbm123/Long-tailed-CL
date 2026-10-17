"""Support-only delete-one direction sensitivity; no query or model selection."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA,DT,features
from run_frozen_centroid_head import centroid
from run_real_query_node import load,save


def sensitivity(z,p):
    q,norm=centroid(z,p);count=len(z)
    assert torch.allclose(p,torch.full_like(p,1/count),atol=1e-12,rtol=0)
    info=dict(count=count,reference_mean_norm=norm,delete_one_attempts=count,undefined_empty_support=int(count==1),undefined_zero_mean=0,defined_directions=0)
    if count==1:return q,info
    means=(q*norm-p[:,None]*z)/(1-p[:,None]);norms=means.norm(dim=1);valid=norms>1e-12
    info.update(undefined_zero_mean=int((~valid).sum()),defined_directions=int(valid.sum()))
    assert bool(torch.isfinite(means).all())
    if bool(valid.any()):
        heads=means[valid]/norms[valid,None];angles=torch.acos((heads@q).clamp(-1,1))*180/torch.pi
        displacement=(heads-q).norm(dim=1)
        info.update(angle_degrees_mean=float(angles.mean()),angle_degrees_median=float(torch.quantile(angles,.5)),angle_degrees_max=float(angles.max()),direction_displacement_mean=float(displacement.mean()),direction_displacement_max=float(displacement.max()),max_unit_query_score_change_bound=float(BETA*displacement.max()),leave_one_mean_norm_min=float(norms[valid].min()),leave_one_mean_norm_max=float(norms[valid].max()))
    return q,info


def unit():
    z=torch.eye(2,dtype=DT);_,r=sensitivity(z,torch.ones(2,dtype=DT)/2)
    assert r['defined_directions']==2 and abs(r['angle_degrees_max']-45)<1e-10
    assert abs(r['max_unit_query_score_change_bound']-BETA*(2-2**.5)**.5)<1e-10
    _,r=sensitivity(z[:1],torch.ones(1,dtype=DT));assert r['undefined_empty_support']==1 and r['defined_directions']==0
    _,r=sensitivity(torch.tensor([[1.,0.],[-1.,0.],[1.,0.]],dtype=DT),torch.ones(3,dtype=DT)/3)
    assert r['undefined_zero_mean']==2 and r['defined_directions']==1
    _,r=sensitivity(z[:1].repeat(3,1),torch.ones(3,dtype=DT)/3);assert r['direction_displacement_max']<1e-12


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    prior=Path(cfg['prior_root']);rawroot=Path(cfg['raw_centroid_root'])
    allowed={prior:{name+'.S_new.features.private.pt'},rawroot:{name+'.centroid.private.pt',name+'.herding.private.pt'}}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        for base,names in allowed.items():
            if p.is_relative_to(base):assert p.name in names,'Support-only input boundary'
    sys.addaudithook(audit)
    for suffix in ('Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt','S_old.features.private.pt','teacher.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    raw=load(rawroot/(name+'.centroid.private.pt'));bank=load(rawroot/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    assert raw['frame'].shape==(1536,seen) and all(bool((v==0).all()) for v in raw['model'])
    rows=[]
    with torch.no_grad():
        for c in range(seen):
            if c<old:x,p=bank[c]
            else:x=new['x'][new['y']==c];p=torch.full((len(x),),1/len(x),dtype=DT)
            assert len(x)==lock['support_counts'][name][c]
            z=torch.cat([features(x[i:i+256],raw['model']) for i in range(0,len(x),256)])
            q,r=sensitivity(z,p);assert torch.allclose(q,raw['frame'][:,c],atol=1e-12,rtol=0),'Published cosine head reconstruction mismatch'
            assert r['defined_directions']+r['undefined_empty_support']+r['undefined_zero_mean']==len(x)
            rows.append(dict(dataset=name,class_index=c,group='old' if c<old else 'new',**r))
            save(root/(job['id']+'.progress.json'),dict(phase='support_delete_one',completed_classes=c+1,total_classes=seen))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,class_rows=rows,reference_head_reconstruction_matched=True,reference_class_means_recomputed=seen,delete_one_attempts=sum(r['delete_one_attempts'] for r in rows),defined_delete_one_class_vectors=sum(r['defined_directions'] for r in rows),undefined_empty_support=sum(r['undefined_empty_support'] for r in rows),undefined_zero_mean=sum(r['undefined_zero_mean'] for r in rows),support_feature_sample_forwards=sum(r['count'] for r in rows),query_sample_forwards=0,head_artifacts_constructed=0,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,query_access=False,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Deterministic support sensitivity, not accuracy or statistical uncertainty; recomputed class vectors are data dependent; no model selection or stored new head.'))


if __name__=='__main__':main()
