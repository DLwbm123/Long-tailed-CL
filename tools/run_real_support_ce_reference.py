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


def group_classification_loss(ss,y,p,old,use_max,balance_groups=False):
    old_norm=ss[:,:old].logsumexp(1);new_norm=ss[:,old:].logsumexp(1)
    within=torch.where(y<old,old_norm,new_norm)-ss[torch.arange(len(y)),y]
    margin=ss[:,old:].max(1).values-ss[:,:old].max(1).values if use_max else new_norm-old_norm
    between=torch.nn.functional.softplus(torch.where(y<old,margin,-margin))
    if balance_groups:
        old_mass=p[y<old].sum();new_mass=p[y>=old].sum()
        assert 0<float(old_mass)<1 and 0<float(new_mass)<1
        gp=p/(2*torch.where(y<old,old_mass,new_mass))
        assert abs(float(gp[y<old].sum())-.5)<1e-12 and abs(float(gp[y>=old].sum())-.5)<1e-12
        return (p*within).sum()+(gp*between).sum()
    return (p*(within+between)).sum()


def check_group_loss():
    ss=torch.tensor([[1.,2.,3.],[2.,1.,0.],[0.,1.,2.]],dtype=DT,requires_grad=True)
    y=torch.arange(3);p=torch.ones(3,dtype=DT)/3
    direct=-(p*ss.log_softmax(1)[torch.arange(3),y]).sum()
    decomposed=group_classification_loss(ss,y,p,2,False)
    assert abs(float((direct-decomposed).detach()))<1e-12
    assert float((torch.autograd.grad(direct,ss)[0]-torch.autograd.grad(decomposed,ss)[0]).abs().max())<1e-12
    zero=torch.zeros(3,3,dtype=DT,requires_grad=True);loss=group_classification_loss(zero,y,p,2,True)
    assert abs(float(loss.detach())-(5/3)*float(torch.log(torch.tensor(2.,dtype=DT))))<1e-12
    grad=torch.autograd.grad(loss,zero)[0]
    assert bool(torch.isfinite(grad).all()) and float(grad.sum(1).abs().max())<1e-12
    assert abs(float(group_classification_loss(zero+3,y,p,2,True).detach())-float(loss.detach()))<1e-12

    probe=torch.tensor([[0.,0.,2.]]*3,dtype=DT)
    balanced=group_classification_loss(probe,y,p,2,True,True)
    expected=(2/3)*torch.log(torch.tensor(2.,dtype=DT))+.5*(torch.nn.functional.softplus(torch.tensor(2.,dtype=DT))+torch.nn.functional.softplus(torch.tensor(-2.,dtype=DT)))
    assert abs(float(balanced-expected))<1e-12
    assert abs(float(group_classification_loss(probe,y,p,2,True)-balanced)-1/3)<1e-12

    margin=torch.tensor(2.,dtype=DT)-torch.log(torch.tensor(2.,dtype=DT))
    expected_lse=(2/3)*torch.log(torch.tensor(2.,dtype=DT))+.5*(torch.nn.functional.softplus(margin)+torch.nn.functional.softplus(-margin))
    assert abs(float(group_classification_loss(probe,y,p,2,False,True)-expected_lse))<1e-12


def project_pair(a,b):
    dot=sum((x*y).sum() for x,y in zip(a,b))
    if float(dot)>=0:return a,b,False
    aa=sum(x.square().sum() for x in a);bb=sum(y.square().sum() for y in b)
    assert float(aa)>0 and float(bb)>0
    return tuple(x-dot/bb*y for x,y in zip(a,b)),tuple(y-dot/aa*x for x,y in zip(a,b)),True


def check_projection():
    a=(torch.tensor([1.,0.],dtype=DT),);b=(torch.tensor([-1.,1.],dtype=DT),)
    pa,pb,active=project_pair(a,b)
    assert active and torch.allclose(pa[0],torch.tensor([.5,.5],dtype=DT)) and torch.allclose(pb[0],torch.tensor([0.,1.],dtype=DT))
    for direction in (pa[0],pb[0]):assert float(a[0]@direction)>=0 and float(b[0]@direction)>=0
    for other in (a,(torch.zeros(2,dtype=DT),)):
        x,y,active=project_pair(a,other);assert not active and x is a and y is other


def train_reference(teacher,current,old_ce,memory,q,steps,retention_scale=1.,max_group=False,balance_groups=False,project_conflicts=False):
    assert 0<float(retention_scale)<float('inf')
    assert not project_conflicts or not (max_group or balance_groups)
    old=len(memory);seen=old+len(current);x,y,p=training_batch(current,old_ce)
    rx,ry,rp=training_batch({},memory);params=clone_model(teacher,True)
    assert all(abs(float(p[y==c].sum())-1/seen)<1e-12 for c in range(seen))
    assert abs(float(rp.sum())-1)<1e-12
    with torch.no_grad():
        ts=scores(rx,teacher,q)[:,:old];target=centered(ts);correct=ts.argmax(1)==ry
    trace=[];projection_trace=[]
    for step in range(steps):
        ss=scores(x,params,q)[:,:seen]
        standard_ce=-(p*ss.log_softmax(1)[torch.arange(len(y)),y]).sum()
        ce=group_classification_loss(ss,y,p,old,max_group,balance_groups) if max_group or balance_groups else standard_ce
        now=centered(scores(rx,params,q)[:,:old])
        retention=((now-target).square().mean(1)*correct*rp).sum()*retention_scale
        if project_conflicts:
            mask=y<old;mass=old/seen;sample_ce=-ss.log_softmax(1)[torch.arange(len(y)),y]
            old_ce=(p[mask]*sample_ce[mask]).sum()/mass;new_ce=(p[~mask]*sample_ce[~mask]).sum()/(1-mass)
            ga=torch.autograd.grad(old_ce,params,retain_graph=True);gb=torch.autograd.grad(new_ce,params)
            gr=torch.autograd.grad(retention,params);pa,pb,active=project_pair(ga,gb)
            original=tuple(mass*a+(1-mass)*b+r for a,b,r in zip(ga,gb,gr))
            grad=tuple(mass*a+(1-mass)*b+r for a,b,r in zip(pa,pb,gr))
            derivative=lambda g,d:-float(sum((a*b).sum() for a,b in zip(g,d)))
            projection_trace.append(dict(step=step,projected=active,old_new_dot=float(sum((a*b).sum() for a,b in zip(ga,gb))),
                original_derivatives=[derivative(g,original) for g in (ga,gb,gr)],updated_derivatives=[derivative(g,grad) for g in (ga,gb,gr)]))
        else:grad=torch.autograd.grad(ce+retention,params)
        with torch.no_grad():
            for v,g in zip(params,grad):v.sub_(g/16);assert bool(torch.isfinite(v).all())
        if step in (0,steps-1):trace.append(dict(preupdate_step=step,CE=float(standard_ce.detach()),classification_loss=float(ce.detach()),retention=float(retention.detach())))
    return clone_model(params),dict(steps=steps,trace=trace,projection_trace=projection_trace,update_rule='symmetric_conflict_projection_then_original_weights_plus_retention' if project_conflicts else 'gradient_descent',adapter_gradient_evaluations=steps*(3 if project_conflicts else 1),classification_objective=('within_CE_plus_balanced_max_group_logistic' if max_group else 'within_CE_plus_balanced_logsumexp_group_logistic') if balance_groups else 'within_CE_plus_max_group_logistic' if max_group else 'full_CE',retention_weighted_teacher_correct=float((correct*rp).sum()),retention_scale=retention_scale,class_mass=1/seen)


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
