"""One candidate continuation and common-state, zero-update gradient diagnostics."""
import json
import os
import time
from pathlib import Path
import torch
from check_fullslot_retention_memory import train, centered
from check_reserved_competition_memory import DT, initial_model, clone_model, frame, scores, training_batch, lowrank_fixture
from check_firststage_query_audit import comparison
from check_memory_stability import predict, uncertainty, decision


def objective_parts(model, teacher, current, memory, q, slots):
    x,y,p=training_batch(current,memory);old_n=len(memory);mask=y<old_n
    ss=scores(x,model,q)[:,slots]
    ce=-ss.log_softmax(1)[torch.arange(len(x)),y]*p
    with torch.no_grad():
        original=scores(x[mask],teacher,q)[:,slots[:old_n]]
        correct=original.argmax(1)==y[mask];target=centered(original)
    retain=((centered(ss[mask,:old_n])-target).square().mean(1)*correct*p[mask]*len(slots)/old_n).sum()
    old_ce=ce[mask].sum();new_ce=ce[~mask].sum()
    return dict(old_CE=old_ce,new_CE=new_ce,classification=old_ce+new_ce,retention=retain,total=old_ce+new_ce+retain)


def gradients(model,teacher,current,memory,q,slots):
    params=clone_model(model,True);parts=objective_parts(params,teacher,current,memory,q,slots)
    values={k:float(v.detach()) for k,v in parts.items()}
    gs={k:torch.cat([g.flatten() for g in torch.autograd.grad(v,params,retain_graph=True)]).detach() for k,v in parts.items()}
    assert float((gs['old_CE']+gs['new_CE']+gs['retention']-gs['total']).abs().max())<1e-10
    return values,gs


def unit_checks():
    g=torch.Generator().manual_seed(353);q=frame(12,4,353);teacher=initial_model(12,353)
    x=torch.nn.functional.normalize(torch.randn(18,12,generator=g,dtype=DT),dim=1)
    old={c:(x[c*6:(c+1)*6],torch.ones(6,dtype=DT)/6) for c in range(2)};current={2:x[12:]};slots=[0,1,2]
    model=(teacher[0],torch.randn(12,8,generator=g,dtype=DT)*.02)
    params=clone_model(model,True);parts=objective_parts(params,teacher,current,old,q,slots)
    # Independent literal training objective verifies the diagnostic's weighting and teacher mask.
    xx,yy,pp=training_batch(current,old);ss=scores(xx,params,q)[:,:3];mask=yy<2
    ts=scores(xx[mask],teacher,q)[:,:2].detach();correct=ts.argmax(1)==yy[mask]
    literal=-(ss.log_softmax(1)[torch.arange(len(xx)),yy]*pp).sum()+((centered(ss[mask,:2])-centered(ts)).square().mean(1)*correct*pp[mask]*3/2).sum()
    assert abs(float(literal-parts['total']))<1e-12
    values,gs=gradients(model,teacher,current,old,q,slots);assert values['retention']>0
    direction=tuple(torch.randn(v.shape,generator=g,dtype=DT) for v in model)
    norm=torch.cat([v.flatten() for v in direction]).norm();direction=tuple(v/norm for v in direction)
    eps=1e-6
    f=lambda sign:objective_parts(tuple(v+sign*eps*d for v,d in zip(model,direction)),teacher,current,old,q,slots)['total']
    fd=float((f(1)-f(-1))/(2*eps));analytic=float(torch.dot(gs['total'],torch.cat([v.flatten() for v in direction])))
    err=abs(fd-analytic);assert err<1e-7
    vv,gg=gradients(teacher,teacher,current,old,q,slots);assert vv['retention']==0 and float(gg['retention'].norm())==0
    assert comparison(gs['total'],gs['total'])['absolute_error']==0
    return dict(status='PASS',objective_literal_error=abs(float(literal-parts['total'])),total_gradient_FD_error=err,teacher_retention_gradient_zero=True,adapter_updates=0,memory_updates=0)


def load(cfg):
    prior=Path(os.environ['Q353_PRIOR']);sid=cfg['id']
    state=torch.load(prior/(sid+'.state.private.pt'),map_location='cpu',weights_only=True)
    refined=torch.load(prior/(sid+'.candidate.private.pt'),map_location='cpu',weights_only=True)
    sealed=torch.load(prior/(sid+'.sealed.private.pt'),map_location='cpu',weights_only=True)
    assert state['case']==refined['case']==sealed['case']==cfg
    old_n=cfg['task_sizes'][0];memories={'moment':state['memories']['moment'],'candidate':refined['candidate'],'selected':refined['selected']}
    for mem in memories.values():
        assert set(mem)==set(range(old_n))
        for c,(z,p) in mem.items():
            assert z.shape==(16,cfg['dimension']) and p.shape==(16,) and bool((p>0).all()) and abs(float(p.sum())-1)<1e-12
            assert torch.equal(z,memories['moment'][c][0])
    return state,sealed,memories


def train_case(cfg,lock,journal):
    start=time.monotonic();root=Path(os.environ['Q353_ROOT']);state,sealed,memories=load(cfg)
    old_n=cfg['task_sizes'][0];seen=sum(cfg['task_sizes'][:2]);x,y,counts=lowrank_fixture(cfg)
    current={c:x[y==c] for c in range(old_n,seen)};teacher=state['teacher'];q=state['frame'];slots=list(range(seen))
    journal(dict(phase='candidate_training',completed_updates=0))
    model,cost=train(teacher,current,memories['candidate'],q,slots,steps=48,full=False)
    torch.save(dict(case=cfg,model=model,adapter_updates=48,memory_updates=0),root/(cfg['id']+'.sealed.private.pt'))
    journal(dict(phase='candidate_sealed',completed_updates=48))
    result=diagnose_case(cfg,journal)
    result.update(training=cost,adapter_updates=48,pipeline_seconds=time.monotonic()-start)
    return result


def diagnose_case(cfg,journal):
    state,sealed,memories=load(cfg);old_n=cfg['task_sizes'][0];seen=sum(cfg['task_sizes'][:2])
    x,y,counts=lowrank_fixture(cfg);current={c:x[y==c] for c in range(old_n,seen)}
    teacher=state['teacher'];q=state['frame'];slots=list(range(seen))
    diagnostic_start=time.monotonic();full={c:(x[y==c],torch.ones(int((y==c).sum()),dtype=DT)/int((y==c).sum())) for c in range(old_n)}
    rows=[];checks=[]
    for name,common in [('teacher',teacher),('moment_endpoint',sealed['models']['moment'])]:
        journal(dict(phase='zero_update_diagnostics',common_state=name))
        reference,rg=gradients(common,teacher,current,full,q,slots)
        for mode,mem in [('full_support',full),*memories.items()]:
            values,gs=(reference,rg) if mode=='full_support' else gradients(common,teacher,current,mem,q,slots)
            new_error=float((gs['new_CE']-rg['new_CE']).abs().max());assert new_error<1e-10
            if name=='teacher':assert abs(values['retention'])<=1e-12 and float(gs['retention'].norm())<=1e-12
            checks.append(dict(common_state=name,mode=mode,new_CE_gradient_max_difference=new_error,gradient_additivity_checked=True))
            for part in values:
                rows.append(dict(common_state=name,mode=mode,component=part,value=values[part],reference_value=reference[part],**comparison(gs[part],rg[part])))
    return dict(status='COMPLETED',case=cfg,diagnostics=rows,diagnostic_checks=checks,diagnostic_seconds=time.monotonic()-diagnostic_start,synthetic_counts=counts,accepted_classes=sealed['accepted_classes'],adapter_updates=0,memory_updates=0,diagnostic_updates=0,sealed=True)


def evaluate(lock,journal):
    start=time.monotonic();root=Path(os.environ.get('Q353_CANDIDATE_ROOT',os.environ['Q353_ROOT']))
    cases=lock['cases'][1:-1]
    for cfg in cases:
        sealed=torch.load(root/(cfg['id']+'.sealed.private.pt'),map_location='cpu',weights_only=True)
        assert sealed['case']==cfg and sealed['adapter_updates']==48 and sealed['memory_updates']==0
    prior_rows=json.loads((Path(os.environ['Q353_SOURCE'])/'docs/memory_stability_design1_20261016/CLASS_RECALL.json').read_text())
    prior_counts={(r['scene'],r['class_index']):(r['moment_correct'],r['selected_correct']) for r in prior_rows}
    class_rows=[];paired=[];summaries=[];intervals=[];decisions=[];evaluation=[]
    for base in lock['base_cases']:
        scale=base['id'];old_n=base['task_sizes'][0];seen=sum(base['task_sizes'][:2])
        cfg_eval=dict(base,seed=lock['evaluation_seeds'][scale],n=base['classes']*1000,count_weights=[1]*base['classes'])
        x,y,counts=lowrank_fixture(cfg_eval);keep=y<seen;x=x[keep];y=y[keep];assert counts==[1000]*base['classes']
        for cfg in [c for c in cases if c['scale']==scale]:
            sid=cfg['id'];journal(dict(phase='evaluation',scene=sid));state,sealed,_=load(cfg)
            candidate=torch.load(root/(sid+'.sealed.private.pt'),map_location='cpu',weights_only=True);assert candidate['case']==cfg and candidate['adapter_updates']==48
            models=dict(sealed['models'],candidate=candidate['model']);pred={m:predict(x,v,state['frame'],seen) for m,v in models.items()}
            for c in range(seen):
                cm=y==c;correct={m:int((p[cm]==c).sum()) for m,p in pred.items()}
                assert (correct['moment'],correct['selected'])==prior_counts[sid,c]
                class_rows.append(dict(scene=sid,class_index=c,group='old' if c<old_n else 'new',subgroup='accepted_old' if c in sealed['accepted_classes'] else 'fallback_old' if c<old_n else 'new',count=int(cm.sum()),**{m+'_correct':v for m,v in correct.items()}))
            for i,base_mode in enumerate(('moment','selected')):
                pair='candidate_vs_'+base_mode;pp=[]
                for row in [v for v in class_rows if v['scene']==sid]:
                    c=row['class_index'];cm=y==c;a=pred[base_mode][cm];b=pred['candidate'][cm];ac=a==c;bc=b==c
                    pp.append(dict(row,pair=pair,baseline=base_mode,baseline_correct=row[base_mode+'_correct'],recall_difference=float((bc.to(DT)-ac.to(DT)).mean()),both_correct=int((ac&bc).sum()),rescued=int((~ac&bc).sum()),harmed=int((ac&~bc).sum()),both_wrong=int((~ac&~bc).sum()),wrong_to_different_wrong=int((~ac&~bc&(a!=b)).sum()),baseline_errors_to_old=int((~ac&(a<old_n)).sum()),baseline_errors_to_new=int((~ac&(a>=old_n)).sum()),candidate_errors_to_old=int((~bc&(b<old_n)).sum()),candidate_errors_to_new=int((~bc&(b>=old_n)).sum())))
                paired.extend(pp)
                ci=uncertainty(pp,old_n,lock['bootstrap_seeds'][sid]+i,lock['bootstrap_draws']);intervals.extend(dict(scene=sid,pair=pair,**v) for v in ci)
                decisions.append(dict(scene=sid,pair=pair,confirmatory=False,**decision(ci)))
                for group in ('all','old','new','accepted_old','fallback_old'):
                    rr=[v for v in pp if group=='all' or v['group']==group or v['subgroup']==group]
                    mean=lambda mode:sum(v[mode+'_correct']/v['count'] for v in rr)/len(rr) if rr else None
                    a,b=mean(base_mode),mean('candidate')
                    summaries.append(dict(scene=sid,pair=pair,group=group,classes=len(rr),count=sum(v['count'] for v in rr),baseline_BA=a,candidate_BA=b,BA_difference=b-a if rr else None))
        evaluation.append(dict(scale=scale,seed=cfg_eval['seed'],per_class=1000,seen_classes=seen,samples=len(y),same_prior_bank=True,prior_control_counts_reproduced=True,all_four_candidates_sealed=True))
    return dict(status='COMPLETED',class_rows=class_rows,paired=paired,summaries=summaries,intervals=intervals,decisions=decisions,evaluation=evaluation,evaluation_seconds=time.monotonic()-start,adapter_updates=0,memory_updates=0)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=unit_checks() if cfg['kind']=='unit' else evaluate(lock,journal) if cfg['kind']=='evaluation' else train_case(cfg,lock,journal)
    r.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,val_accessed=False,test_accessed=False)
    assert not r['GPU_initialized'];return r


def resume_zero_update(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=evaluate(lock,journal) if cfg['kind']=='evaluation' else diagnose_case(cfg,journal)
    r.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,val_accessed=False,test_accessed=False,engineering_continuation=True)
    assert not r['GPU_initialized'] and r['adapter_updates']==r['memory_updates']==0
    return r
