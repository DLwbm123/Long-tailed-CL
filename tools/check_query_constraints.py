"""Fixed finite post-update selection with temporal and audited S/Q/E separation."""
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from check_reserved_competition_memory import DT, clone_model, scores, lowrank_fixture, training_batch, initial_model, frame
from check_fullslot_retention_memory import centered
from check_memory_stability import decision
from check_support_reference import statistics, transitions as reference_transitions


def transitions(a,b,c,old_n):
    return {k.replace("full_support_","selected_"):v for k,v in reference_transitions(a,b,c,old_n).items()}


def save_once(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def model_name(sid,rho):return sid+('_half' if rho==.5 else '_double')+'.sealed.private.pt'


def permitted(path,role,root,selected=()):
    p=Path(path).resolve()
    prior=root.parent/'memory_stability_design1_20261016'
    if role in ('query','evaluation') and (p==prior or prior in p.parents):return False
    if root not in p.parents:return True
    name=p.name
    if role in ('training','query') and (name.startswith('E_') or name.startswith('sealed_evaluation')):return False
    if role=='training' and (name.startswith('query_selection') or name=='SELECTION_LOCK.json'):return False
    if role=='evaluation' and (name.startswith('query_selection') or name.startswith('Q_')):return False
    if role=='evaluation' and name.endswith('.sealed.private.pt') and name not in selected:return False
    return True


def install_guard(role,root,selected=()):
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]);assert permitted(p,role,root,selected), 'Role-denied data access'
    sys.addaudithook(audit)


def objective(model,q,x,y,p,old_n,seen,target,correct,rho):
    ss=scores(x,model,q)[:,:seen];mask=y<old_n
    factor=torch.where(mask,rho,1.)/(rho*old_n/seen+(seen-old_n)/seen)
    ce=-(ss.log_softmax(1)[torch.arange(len(y)),y]*p*factor).sum()
    retain=(((centered(ss[mask,:old_n])-target).square().mean(1)*correct)*p[mask]*seen/old_n).sum()
    return ce,retain


def select(rows):
    feasible=[v for v in rows if all(v['delta_'+g]>=-.01-1e-12 for g in ('old','new'))]
    best=max(v['delta_all'] for v in feasible)
    if best<=1e-12:return dict(rho=1.,reason='NO_FEASIBLE_POSITIVE_GAIN')
    ties=[v for v in feasible if abs(v['delta_all']-best)<=1e-12]
    chosen=min(ties,key=lambda v:(v['rho']!=1.,abs(math.log(v['rho'])),v['rho']))
    return dict(rho=chosen['rho'],reason='MAX_FEASIBLE_Q_BA')


def bank(cfg,seed,n=1000):
    # Same fixed class distribution; never instantiate unseen-class features or labels.
    k=cfg['classes'];rank=2*k;seen=sum(cfg['task_sizes'][:2]);gen=torch.Generator().manual_seed(seed)
    xx=[];yy=[];zz=[]
    for c in range(seen):
        noise=F.normalize(torch.randn(n,rank,generator=gen,dtype=DT),dim=1)
        component=torch.arange(n)%5==0;v=.85*noise;v[:,c]+=.45;v[:,k+c]+=.6*torch.where(component,.8,-.2).to(DT)
        out=torch.zeros(n,cfg['dimension'],dtype=DT);out[:,:rank]=F.normalize(v,dim=1)
        xx.append(out);yy.append(torch.full((n,),c,dtype=torch.long));zz.append(component.long())
    permutation=torch.randperm(n*seen,generator=gen)
    return torch.cat(xx)[permutation],torch.cat(yy)[permutation],torch.cat(zz)[permutation]


def predict_stats(x,y,model,q,old_n,seen):
    with torch.no_grad():ss=torch.cat([scores(x[i:i+512],model,q)[:,:seen] for i in range(0,len(x),512)])
    return statistics(ss,y,old_n)


def class_metrics(sid,role,mode,s,y,old_n):
    rows=[]
    for c in range(int(y.max())+1):
        cm=y==c;p=s['pred'][cm]
        rows.append(dict(scene=sid,role=role,model=mode,class_index=c,group='old' if c<old_n else 'new',count=int(cm.sum()),correct=int((p==c).sum()),CE=float(s['CE'][cm].mean()),margin=float(s['margin'][cm].mean()),opposite_margin=float(s['opposite_margin'][cm].mean()),errors_to_old=int(((p!=c)&(p<old_n)).sum()),errors_to_new=int(((p!=c)&(p>=old_n)).sum())))
    return rows


def groups(rows):
    result={}
    for g in ('all','old','new'):
        rr=[r for r in rows if g=='all' or r['group']==g]
        result[g]=dict(BA=sum(r['correct']/r['count'] for r in rr)/len(rr),CE=sum(r['CE'] for r in rr)/len(rr),margin=sum(r['margin'] for r in rr)/len(rr))
    return result


def intervals(strata,old_n,seed,draws):
    rng=np.random.default_rng(seed);per_class=[]
    for c in sorted({r['class_index'] for r in strata}):
        rr=[r for r in strata if r['class_index']==c];total=sum(r['count'] for r in rr);sample=np.zeros(draws)
        for r in rr:
            counts=[r[k] for k in ('both_correct','rescued','harmed','both_wrong')];assert sum(counts)==r['count']
            v=rng.multinomial(r['count'],np.array(counts)/r['count'],size=draws)
            sample+=(v[:,1]-v[:,2])/total
        per_class.append(sample)
    matrix=np.stack(per_class,1);out=[]
    for g,cs in [('all',list(range(len(per_class)))),('old',list(range(old_n))),('new',list(range(old_n,len(per_class))))]:
        point=sum(sum(r['rescued']-r['harmed'] for r in strata if r['class_index']==c)/sum(r['count'] for r in strata if r['class_index']==c) for c in cs)/len(cs)
        lo,hi=np.quantile(matrix[:,cs].mean(1),[.025,.975],method='linear')
        out.append(dict(group=g,BA_difference=point,lower=float(lo),upper=float(hi),confidence=.95,bootstrap_seed=seed,bootstrap_draws=draws))
    return out


def unit():
    gen=torch.Generator().manual_seed(355);teacher=initial_model(8,355);model=clone_model(teacher,True);q=frame(8,3,355)
    x=F.normalize(torch.randn(12,8,generator=gen,dtype=DT),dim=1);old={0:(x[:3],torch.ones(3,dtype=DT)/3),1:(x[3:7],torch.ones(4,dtype=DT)/4)}
    xx,yy,p=training_batch({2:x[7:]},old);mask=yy<2
    with torch.no_grad():ts=scores(xx[mask],teacher,q)[:,:2];target=centered(ts);correct=ts.argmax(1)==yy[mask]
    with torch.no_grad():model[1].add_(torch.randn(model[1].shape,generator=gen,dtype=DT)*.01)
    ce,re=objective(model,q,xx,yy,p,2,3,target,correct,1.)
    ss=scores(xx,model,q)[:,:3];literal=-(ss.log_softmax(1)[torch.arange(len(yy)),yy]*p).sum()+(((centered(ss[mask,:2])-target).square().mean(1)*correct)*p[mask]*3/2).sum()
    a=torch.autograd.grad(ce+re,model,retain_graph=True);b=torch.autograd.grad(literal,model)
    assert abs(float((ce+re-literal).detach()))<1e-12 and max(float((u-v).abs().max()) for u,v in zip(a,b))<1e-12
    for rho in (.5,2.):
        factor=torch.where(mask,rho,1.)/(rho*2/3+1/3);assert abs(float((p*factor).sum())-1)<1e-12
        assert abs(float(objective(model,q,xx,yy,p,2,3,target,correct,rho)[1]-re))<1e-12
    base=dict(rho=1.,delta_all=0.,delta_old=0.,delta_new=0.);bad=dict(rho=2.,delta_all=.1,delta_old=.15,delta_new=-.02);good=dict(rho=.5,delta_all=.02,delta_old=-.01,delta_new=.08)
    assert select([base,bad])['rho']==1 and select([base,bad,good])['rho']==.5
    assert select([base,dict(good,delta_all=0)])['rho']==1
    assert select([base,good,dict(good,rho=2.)])['rho']==.5
    cfg=dict(classes=3,dimension=8,task_sizes=[1,1],seed=355,n=30,count_weights=[1,1,1]);xb,yb,zb=bank(cfg,355,10);ref,labels,_=lowrank_fixture(cfg)
    # Compare as multisets to verify the query sampler preserves the existing distribution exactly.
    assert all(torch.equal(xb[yb==c].sort(dim=0).values,ref[labels==c].sort(dim=0).values) for c in (0,1))
    assert all(int(((yb==c)&(zb==1)).sum())==2 for c in (0,1))
    dummy=[dict(class_index=c,count=n,both_correct=n,rescued=0,harmed=0,both_wrong=0) for c in range(3) for n in (2,8)]
    assert all(v['lower']==v['upper']==v['BA_difference']==0 for v in intervals(dummy,2,355,100))
    root=Path('/tmp/unit355');assert not permitted(root.parent/'memory_stability_design1_20261016'/'a','query',root)
    assert not permitted(root/'E_SEEDS.private.json','query',root) and not permitted(root/'query_selection.json','evaluation',root)
    assert not permitted(root/'other.sealed.private.pt','evaluation',root,('chosen.sealed.private.pt',))
    return dict(status='PASS',adapter_updates=0,memory_updates=0,rho_one_objective_gradient_match=True,CE_normalization_retention_checked=True,selector_checks=True,sampler_matches_existing=True,stratified_bootstrap_zero_checked=True,access_rules_checked=True)


def train_case(cfg,lock,root,journal):
    scene=cfg['scene'];sid=scene['id'];rho=cfg['rho'];prior=Path(os.environ['Q355_PRIOR'])
    state=torch.load(prior/(sid+'.state.private.pt'),weights_only=True);sealed=torch.load(prior/(sid+'.sealed.private.pt'),weights_only=True)
    assert state['case']==sealed['case']==scene
    start=time.monotonic();old_n=scene['task_sizes'][0];seen=sum(scene['task_sizes'][:2]);x,y,counts=lowrank_fixture(scene)
    current={c:x[y==c] for c in range(old_n,seen)};xx,yy,p=training_batch(current,state['memories']['moment']);mask=yy<old_n
    teacher=state['teacher'];q=state['frame'];model=clone_model(teacher,True)
    with torch.no_grad():ts=scores(xx[mask],teacher,q)[:,:old_n];target=centered(ts);correct=ts.argmax(1)==yy[mask]
    journal(dict(phase='rho_training',rho=rho,completed_updates=0));trace=[]
    for step in range(48):
        ce,retain=objective(model,q,xx,yy,p,old_n,seen,target,correct,rho);gg=torch.autograd.grad(ce+retain,model)
        if step in (0,47):trace.append(dict(preupdate_step=step,classification=float(ce),retention=float(retain),gradient_norm=float(torch.cat([g.flatten() for g in gg]).norm())))
        with torch.no_grad():
            for a,b in zip(model,gg):a.sub_(b/16);assert bool(torch.isfinite(a).all())
    training=dict(seconds=time.monotonic()-start,updates=48,rho=rho,trace=trace,synthetic_counts=counts)
    torch.save(dict(scene=scene,rho=rho,model=clone_model(model),training=training,adapter_updates=48,memory_updates=0,sealed_at=time.time()),root/model_name(sid,rho))
    base=root/(sid+'.baseline.private.pt')
    if not base.exists():torch.save(dict(scene=scene,model=sealed['models']['moment'],frame=q),base)
    journal(dict(phase='sealed',completed_updates=48))
    return dict(status='COMPLETED',scene=sid,rho=rho,training=training,adapter_updates=48,memory_updates=0,sealed_at=time.time())


def query(lock,root,journal):
    program=json.loads((root/'CPU_PROGRAM.json').read_text());assert len(program['cases'])==9 and not program['failures']
    assert not (root/'E_SEEDS.private.json').exists() and not (root/'SELECTION_LOCK.json').exists()
    for c in lock['cases'][1:9]:
        v=torch.load(root/model_name(c['scene']['id'],c['rho']),weights_only=True);assert v['adapter_updates']==48 and v['scene']==c['scene'] and v['rho']==c['rho']
    started=time.time();all_rows=[];comparisons=[];choices=[];receipts=[]
    for scene in lock['scenes']:
        sid=scene['id'];old_n=scene['task_sizes'][0];seen=sum(scene['task_sizes'][:2]);journal(dict(phase='Q_generated_after_all_candidates_sealed',scene=sid))
        seed=lock['query_seeds'][sid];x,y,z=bank(scene,seed);base=torch.load(root/(sid+'.baseline.private.pt'),weights_only=True);summary={}
        for rho in lock['rhos']:
            model=base['model'] if rho==1 else torch.load(root/model_name(sid,rho),weights_only=True)['model']
            ss=predict_stats(x,y,model,base['frame'],old_n,seen);rows=class_metrics(sid,'Q',str(rho),ss,y,old_n);all_rows+=rows;summary[rho]=groups(rows)
        rr=[]
        for rho in lock['rhos']:
            row=dict(scene=sid,rho=rho,groups=summary[rho],**{'delta_'+g:summary[rho][g]['BA']-summary[1.][g]['BA'] for g in ('all','old','new')})
            row['feasible']=all(row['delta_'+g]>=-.01-1e-12 for g in ('old','new'));rr.append(row)
        chosen=select(rr);choices.append(dict(scene=sid,**chosen,model_file=sid+'.baseline.private.pt' if chosen['rho']==1 else model_name(sid,chosen['rho'])))
        comparisons+=rr;receipts.append(dict(scene=sid,role='Q',seed=seed,samples=len(y),per_class=1000,generated_at=time.time(),unseen_classes_generated=False))
    locked=time.time();save_once(root/'SELECTION_LOCK.json',dict(source_commit=program['source_commit'],locked_at=locked,query_started_at=started,choices=choices))
    return dict(status='COMPLETED',class_rows=all_rows,comparisons=comparisons,choices=choices,receipts=receipts,locked_at=locked,adapter_updates=0,memory_updates=0)


def evaluate(lock,root,journal):
    program=json.loads((root/'CPU_PROGRAM.json').read_text());assert len(program['cases'])==10 and not program['failures']
    selected=json.loads((root/'SELECTION_LOCK.json').read_text());seeds=json.loads((root/'E_SEEDS.private.json').read_text());assert selected['locked_at']<=seeds['created_at']<=time.time()
    rows=[];pairs=[];strata=[];summaries=[];cis=[];decisions=[];receipts=[]
    for scene in lock['scenes']:
        sid=scene['id'];choice=next(v for v in selected['choices'] if v['scene']==sid);old_n=scene['task_sizes'][0];seen=sum(scene['task_sizes'][:2])
        generated=time.time();journal(dict(phase='E_generated_after_all_choices_locked',scene=sid));x,y,z=bank(scene,seeds['seeds'][sid]);base=torch.load(root/(sid+'.baseline.private.pt'),weights_only=True)
        chosen=base if choice['rho']==1 else torch.load(root/choice['model_file'],weights_only=True)
        stats={m:predict_stats(x,y,obj['model'],base['frame'],old_n,seen) for m,obj in [('baseline',base),('selected',chosen)]};scene_rows={}
        for m,s in stats.items():scene_rows[m]=class_metrics(sid,'E',m,s,y,old_n);rows+=scene_rows[m]
        gs={m:groups(cr) for m,cr in scene_rows.items()}
        local=[]
        for c in range(seen):
            cm=y==c;a=stats['baseline']['pred'][cm];b=stats['selected']['pred'][cm]
            pairs.append(dict(scene=sid,class_index=c,group='old' if c<old_n else 'new',count=int(cm.sum()),**transitions(a,b,c,old_n)))
            for component in (0,1):
                mm=(y==c)&(z==component);a=stats['baseline']['pred'][mm];b=stats['selected']['pred'][mm]
                local.append(dict(scene=sid,class_index=c,component=component,count=int(mm.sum()),**transitions(a,b,c,old_n)))
        strata+=local;cc=intervals(local,old_n,lock['bootstrap_seeds'][sid],10000);cis.extend(dict(scene=sid,**v) for v in cc);decisions.append(dict(scene=sid,rho=choice['rho'],**decision(cc)))
        for g in ('all','old','new'):summaries.append(dict(scene=sid,group=g,rho=choice['rho'],baseline=gs['baseline'][g],selected=gs['selected'][g],BA_difference=gs['selected'][g]['BA']-gs['baseline'][g]['BA']))
        receipts.append(dict(scene=sid,role='E',seed=seeds['seeds'][sid],samples=len(y),per_class=1000,generated_at=generated,selection_locked_at=selected['locked_at'],E_seed_created_at=seeds['created_at'],evaluated_models=['baseline','selected'],unseen_classes_generated=False))
    return dict(status='COMPLETED',class_rows=rows,paired=pairs,strata=strata,summaries=summaries,intervals=cis,decisions=decisions,receipts=receipts,selection_lock=selected,adapter_updates=0,memory_updates=0)


def run(cfg,lock,journal):
    root=Path(os.environ['Q355_ROOT']);role=cfg['kind'];torch.set_num_threads(2);torch.set_default_dtype(DT);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    chosen=()
    if role=='evaluation':chosen=tuple(v['model_file'] for v in json.loads((root/'SELECTION_LOCK.json').read_text())['choices'])
    install_guard(role,root,chosen)
    start=time.monotonic()
    result=unit() if role=='unit' else train_case(cfg,lock,root,journal) if role=='training' else query(lock,root,journal) if role=='query' else evaluate(lock,root,journal)
    result.update(id=cfg['id'],role=role,operation_seconds=time.monotonic()-start,GPU_initialized=torch.cuda.is_initialized(),medical_images=0,val_accessed=False,test_accessed=False,role_access_guard=True)
    assert not result['GPU_initialized'];return result
