"""Full old-support CE reference with the original memory retention kept fixed."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT,clone_model,scores,training_batch,initial_model,frame
from check_fullslot_retention_memory import centered,train
from query_node_weights import select
from run_real_query_node import load,metrics,save


def train_reference(teacher,current,old_ce,memory,q,steps):
    old=len(memory);seen=old+len(current);x,y,p=training_batch(current,old_ce)
    rx,ry,rp=training_batch({},memory);params=clone_model(teacher,True)
    assert all(abs(float(p[y==c].sum())-1/seen)<1e-12 for c in range(seen))
    assert abs(float(rp.sum())-1)<1e-12
    with torch.no_grad():
        ts=scores(rx,teacher,q)[:,:old];target=centered(ts);correct=ts.argmax(1)==ry
    trace=[]
    for step in range(steps):
        ss=scores(x,params,q)[:,:seen]
        ce=-(p*ss.log_softmax(1)[torch.arange(len(y)),y]).sum()
        now=centered(scores(rx,params,q)[:,:old])
        retention=((now-target).square().mean(1)*correct*rp).sum()
        grad=torch.autograd.grad(ce+retention,params)
        with torch.no_grad():
            for v,g in zip(params,grad):v.sub_(g/16);assert bool(torch.isfinite(v).all())
        if step in (0,steps-1):trace.append(dict(preupdate_step=step,CE=float(ce.detach()),retention=float(retention.detach())))
    return clone_model(params),dict(steps=steps,trace=trace,retention_weighted_teacher_correct=float((correct*rp).sum()),class_mass=1/seen)


def unit():
    q=frame(8,3,74002);teacher=initial_model(8,74002);g=torch.Generator().manual_seed(74002)
    memory={c:(torch.nn.functional.normalize(q[:,c][None]+.1*torch.randn(n,8,dtype=DT,generator=g),dim=1),torch.arange(1,n+1,dtype=DT)/(n*(n+1)/2)) for c,n in [(0,2),(1,4)]}
    current={2:torch.nn.functional.normalize(torch.randn(3,8,dtype=DT,generator=g),dim=1)}
    a,info=train_reference(teacher,current,memory,memory,q,3);b,_=train(teacher,current,memory,q,[0,1,2],steps=3)
    err=max(float((v-w).abs().max()) for v,w in zip(a,b));assert err<1e-12 and info['retention_weighted_teacher_correct']>0
    return dict(status='COMPLETED',unit='split_CE_retention_matches_original_on_memory',max_parameter_error=err,engineering_adapter_updates=6)


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE']);job=json.loads(os.environ['Q356_JOB'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT)
    if job['role']=='unit':
        result=unit();result['source_commit']=os.environ['Q356_SHA'];save(root/(job['id']+'.json'),result);return
    name=job['dataset'];case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    prior=Path(cfg['prior_root']);diagnostic=Path(cfg['diagnostic_root']);sealed=False
    allowed={name+'.'+s for s in ('teacher.private.pt','S_old.features.private.pt','S_new.features.private.pt')}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in allowed or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Full support reference role boundary'
        if p.is_relative_to(diagnostic):assert sealed and p.name==name+'_half.json','References only after sealing'
    sys.addaudithook(audit)
    for suffix in ('E.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Pre-seal access guard failed')
    start=time.monotonic();save(root/(job['id']+'.progress.json'),dict(phase='full_old_support_CE_fixed_memory_retention',dataset=name))
    state=load(prior/(name+'.teacher.private.pt'));data=load(prior/(name+'.S_old.features.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    old_ce={c:(data['x'][data['y']==c],torch.ones(int((data['y']==c).sum()),dtype=DT)/int((data['y']==c).sum())) for c in range(old)}
    current={c:new['x'][new['y']==c] for c in range(old,seen)}
    model,cost=train_reference(state['teacher'],current,old_ce,state['memory'],state['frame'],lock['adapter_steps'])
    obj=dict(model=model,frame=state['frame']);torch.save(obj,root/(name+'.full_ce.private.pt'));sealed=True;sealed_at=time.time()
    ref=json.loads((diagnostic/(name+'_half.json')).read_text());rows=[r for r in ref['class_rows'] if r['model']=='baseline'];summary=[]
    for query in ('Qfit','Qgate'):
        _,rr,_=metrics(obj,load(prior/(name+'.'+query+'.features.private.pt')),old,seen)
        rows.extend(dict(dataset=name,query=query,model='full_ce',**r) for r in rr)
        for mode in ('baseline','full_ce'):
            for group in ('all','old','new'):
                selected=[r for r in rows if r['query']==query and r['model']==mode and (group=='all' or r['group']==group)]
                summary.append(dict(dataset=name,query=query,model=mode,group=group,BA=sum(r['recall'] for r in selected)/len(selected),CE=sum(r['CE'] for r in selected)/len(selected)))
    gates={}
    for query in ('Qfit','Qgate'):
        get=lambda mode:{r['group']:r['BA'] for r in summary if r['query']==query and r['model']==mode}
        gates[query]=select(get('baseline'),get('full_ce'))
    result=dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,adapter_updates=lock['adapter_steps'],
        training=cost,old_support_count=len(data['y']),new_support_count=len(new['y']),old_support_feature_bytes=data['x'].numel()*data['x'].element_size(),
        original_memory_node_count=sum(len(z) for z,p in state['memory'].values()),original_retention_nodes_weights_and_targets=True,
        old_S_access=True,query_used_for_fitting=False,E_access=False,new_labels=0,GPU_used=False,
        gates=gates,both_queries_pass=all(v['selected']=='candidate' for v in gates.values()),class_rows=rows,summaries=summary,
        scope='Additional old-support resource reference, not memory-budget-compliant candidate or mathematical upper bound',operation_seconds=time.monotonic()-start)
    save(root/(job['id']+'.json'),result)


if __name__=='__main__':main()
