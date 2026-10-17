"""Existing class-balanced ridge on the same frozen raw support features."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from check_reserved_competition_memory import BETA,DT,features
from query_node_weights import select
from run_real_query_node import load,metrics,save
import prototype_analytic as analytic


REGULARIZATION=.001


def unit():
    x=np.eye(2,dtype=np.float64);bank=analytic.append(analytic.empty(2),x,np.array([0,1]),[0,1]);w,res=analytic.ridge(bank,REGULARIZATION)
    assert np.allclose(w,np.eye(2)/(1+2*REGULARIZATION),atol=1e-12,rtol=0) and res<1e-12
    xx=np.vstack((x,x[1:]));bb=analytic.append(analytic.empty(2),xx,np.array([0,1,1]),[0,1]);ww,_=analytic.ridge(bb,REGULARIZATION)
    assert np.allclose(ww,w,atol=1e-12,rtol=0),'Class-balanced duplication changed solution'


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['regularization']==REGULARIZATION
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];prior=Path(cfg['prior_root']);refroot=Path(cfg['centered_root']);sealed=False
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in {name+'.teacher.private.pt',name+'.S_new.features.private.pt'} or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Ridge fit/query boundary'
        if p.is_relative_to(refroot):assert sealed and p.name==name+'_centered.json','Historical aggregates before seal'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    state=load(prior/(name+'.teacher.private.pt'));bank=load(root/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'));model=tuple(torch.zeros_like(v) for v in state['teacher'])
    zz=[];yy=[];stats=[];save(root/(job['id']+'.progress.json'),dict(phase='fixed_class_balanced_ridge',dataset=name))
    with torch.no_grad():
        for c in range(seen):
            if c<old:x,p=bank[c]
            else:x=new['x'][new['y']==c];p=torch.full((len(x),),1/len(x),dtype=DT)
            assert len(x)>0 and torch.allclose(p,torch.full_like(p,1/len(x)),atol=1e-12,rtol=0),'Existing helper requires uniform class weights'
            z=torch.cat([features(x[i:i+256],model) for i in range(0,len(x),256)]);assert z.shape==(len(x),1536) and bool(torch.isfinite(z).all())
            zz.append(z.numpy());yy.append(np.full(len(x),c,dtype=np.int64));stats.append(dict(class_index=c,count=len(x)))
    z=np.concatenate(zz);y=np.concatenate(yy);moments=analytic.append(analytic.empty(1536),z,y,list(range(seen)))
    w,residual=analytic.ridge(moments,REGULARIZATION);head=torch.from_numpy(w);assert head.shape==(1536,seen) and bool(torch.isfinite(head).all())
    obj=dict(model=model,frame=head);torch.save(obj,root/(name+'.ridge.private.pt'));sealed=True;sealed_at=time.time()
    for c in range(seen):stats[c]['head_norm']=float(head[:,c].norm())
    ref=json.loads((refroot/(name+'_centered.json')).read_text());rows=list(ref['class_rows']);references=['raw_centroid','MM','HH','identity_ETF','teacher_centroid','teacher_ETF','coordinate_median','centered_centroid'];summaries=[];gates={};counts=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));assert data['y'].dtype==torch.int64 and bool(((data['y']>=0)&(data['y']<seen)).all())
        counts.append(dict(query=query,count=len(data['y'])));_,rr,_=metrics(obj,data,old,seen);rows.extend(dict(dataset=name,query=query,model='raw_ridge',**r) for r in rr)
        for label in references+['raw_ridge']:
            for group in ('all','old','new'):
                cc=[r for r in rows if r['query']==query and r['model']==label and (group=='all' or r['group']==group)]
                assert len(cc)==(seen if group=='all' else old if group=='old' else seen-old)
                summaries.append(dict(dataset=name,query=query,model=label,group=group,BA=sum(r['recall'] for r in cc)/len(cc),CE=sum(r['CE'] for r in cc)/len(cc)))
        get=lambda label:{r['group']:r['BA'] for r in summaries if r['query']==query and r['model']==label}
        gates[query]={label:select(get(label),get('raw_ridge')) for label in references}
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,class_rows=rows,summaries=summaries,gates=gates,primary_reference='raw_centroid',primary_both_queries_pass=all(g['raw_centroid']['selected']=='candidate' for g in gates.values()),head_stats=stats,head_shape=list(head.shape),head_elements=head.numel(),head_bytes=head.numel()*head.element_size(),regularization=REGULARIZATION,solve_relative_residual=residual,production_linear_solves=1,engineering_small_linear_solves=2,logit_scale=BETA,head_constructions=1,head_column_normalization=False,derived_bias_elements=0,support_feature_forwards=len(z),query_sample_forwards=sum(v['count'] for v in counts),query_counts=counts,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,query_used_for_fitting=False,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Existing class-balanced ridge reference on fixed H/new-S raw features, not a novel method. Data-fitted linear weights; no temperature or regularization selection. CE depends on unnormalized learned weight scale.'))


if __name__=='__main__':main()
