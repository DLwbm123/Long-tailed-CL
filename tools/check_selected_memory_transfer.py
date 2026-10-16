"""One paired next-stage test; all trajectories sealed before fresh synthetic evaluation."""
import json
import os
import time
from pathlib import Path
import torch
from check_fullslot_retention_memory import train
from check_reserved_competition_memory import DT, clone_model, scores, lowrank_fixture


def gate(rows):
    old=[r for r in rows if r['group']=='old'];new=[r for r in rows if r['group']=='new']
    assert old and new and all(r['count']>0 for r in rows)
    old_gain=sum((r['selected_correct']-r['moment_correct'])/r['count'] for r in old)/len(old)
    new_gain=sum((r['selected_correct']-r['moment_correct'])/r['count'] for r in new)/len(new)
    no_harm=all(r['selected_correct']>=r['moment_correct'] for r in rows)
    # Exact counts enforce zero class harm; the tolerance only governs strict mean improvement.
    return dict(pass_strict=old_gain>1e-12 and no_harm,old_BA_gain=old_gain,new_BA_gain=new_gain,
        every_class_no_harm=no_harm,harmed_classes=[r['class_index'] for r in rows if r['selected_correct']<r['moment_correct']],
        improved_old_classes=[r['class_index'] for r in old if r['selected_correct']>r['moment_correct']])


def load(cfg):
    state=torch.load(Path(os.environ['Q351_PRIOR'])/(cfg['id']+'.state.private.pt'),map_location='cpu',weights_only=True)
    refined=torch.load(Path(os.environ['Q351_SELECTED'])/(cfg['id']+'.candidate.private.pt'),map_location='cpu',weights_only=True)
    assert state['case']==refined['case']==cfg
    moment=state['memories']['moment'];selected=refined['selected'];accepted=[]
    assert set(moment)==set(selected)==set(range(cfg['task_sizes'][0]))
    for row in refined['acceptance']:
        c=row['class_index'];source=refined['candidate'][c] if row['accepted'] else moment[c]
        assert all(torch.equal(a,b) for a,b in zip(selected[c],source))
        assert torch.equal(moment[c][0],selected[c][0])
        assert abs(float(selected[c][1].sum())-1)<1e-12 and bool((selected[c][1]>0).all())
        if row['accepted']:accepted.append(c)
    return state,moment,selected,accepted


def unit_checks(lock):
    a=[dict(class_index=0,group='old',count=10,moment_correct=5,selected_correct=6),dict(class_index=1,group='new',count=10,moment_correct=5,selected_correct=5)]
    assert gate(a)['pass_strict']
    assert not gate([dict(a[0],selected_correct=5),a[1]])['pass_strict']
    assert not gate([a[0],dict(a[1],selected_correct=4)])['pass_strict']
    assert not gate(a+[dict(class_index=2,group='old',count=1,moment_correct=1,selected_correct=0)])['pass_strict']
    state,moment,selected,accepted=load(lock['small_identity_case'])
    assert not accepted and all(torch.equal(a,b) for c in moment for a,b in zip(moment[c],selected[c]))
    return dict(status='PASS',gate_positive_zero_new_harm_old_harm_checked=True,small_exact_fallback_identity=True,small_training_updates=0,adapter_updates=0,memory_updates=0)


def train_scale(cfg,lock,journal):
    start=time.monotonic();state,moment,selected,accepted=load(cfg)
    assert accepted==lock['accepted_classes'][cfg['id']]
    teacher=state['teacher'];q=state['frame'];frozen=clone_model(teacher)
    original={m:{c:(z.clone(),p.clone()) for c,(z,p) in mem.items()} for m,mem in [('moment',moment),('selected',selected)]}
    x,y,counts=lowrank_fixture(cfg);old_n=cfg['task_sizes'][0];seen=sum(cfg['task_sizes'][:2])
    current={c:x[y==c] for c in range(old_n,seen)};slots=list(range(seen));models={};training=[]
    load_seconds=time.monotonic()-start
    for mode,mem in [('moment',moment),('selected',selected)]:
        journal(dict(phase='next_stage_training',mode=mode,completed_updates=0))
        models[mode],cost=train(teacher,current,mem,q,slots,steps=48,full=False)
        training.append(dict(mode=mode,**cost))
        journal(dict(phase='next_stage_training',mode=mode,completed_updates=48))
    assert all(torch.equal(a,b) for a,b in zip(teacher,frozen))
    assert all(torch.equal(a,b) for mode,mem in [('moment',moment),('selected',selected)] for c,zp in mem.items() for a,b in zip(zp,original[mode][c]))
    torch.save(dict(case=cfg,models=models,accepted_classes=accepted,adapter_updates=96),Path(os.environ['Q351_ROOT'])/(cfg['id']+'.sealed.private.pt'))
    return dict(status='COMPLETED',id=cfg['id'],case=cfg,training=training,accepted_classes=accepted,old_classes=old_n,new_classes=seen-old_n,
        adapter_updates=96,memory_updates=0,load_seconds=load_seconds,pipeline_seconds=time.monotonic()-start,
        evaluation_generated=False,sealed=True,teacher_and_memories_unchanged=True,old_support_used_in_training=False,new_training_counts=[counts[c] for c in range(old_n,seen)])


def evaluate(lock,journal):
    start=time.monotonic();root=Path(os.environ['Q351_ROOT']);program=json.loads((root/'CPU_PROGRAM.json').read_text())
    configs=lock['cases'][1:3];sealed={}
    for cfg in configs:
        cid=cfg['id'];done=next(v for v in program['cases'] if v['id']==cid)
        assert done['status']=='COMPLETED'
        result=json.loads((root/(cid+'.json')).read_text());assert result['sealed'] and result['adapter_updates']==96
        sealed[cid]=torch.load(root/(cid+'.sealed.private.pt'),map_location='cpu',weights_only=True)
        assert sealed[cid]['case']==cfg and sealed[cid]['adapter_updates']==96
    # Neither scale's evaluation samples exist before all four training trajectories are sealed.
    rows=[];classes=[];paired=[];gates=[];baseline=[];eval_info=[]
    for cfg in configs:
        cid=cfg['id'];journal(dict(phase='sealed_independent_evaluation',case=cid))
        state,_,_,accepted=load(cfg);old_n=cfg['task_sizes'][0];seen=sum(cfg['task_sizes'][:2]);slots=list(range(seen))
        x,y,counts=lowrank_fixture(dict(cfg,seed=cfg['seed']+20000));mask=y<seen;x=x[mask];y=y[mask]
        models={'start':state['teacher'],**sealed[cid]['models']};pred={}
        with torch.no_grad():
            for mode,model in models.items():pred[mode]=scores(x,model,state['frame'])[:,slots].argmax(1)
            oldmask=y<old_n;oldonly=scores(x[oldmask],state['teacher'],state['frame'])[:,:old_n].argmax(1)
        for c in range(seen):
            cm=y==c;n=int(cm.sum());group='old' if c<old_n else 'new'
            subgroup=('accepted_old' if c in accepted else 'fallback_old') if c<old_n else 'new'
            ac=pred['moment'][cm]==c;bc=pred['selected'][cm]==c
            row=dict(case=cid,class_index=c,group=group,subgroup=subgroup,count=n,
                start_correct=int((pred['start'][cm]==c).sum()),moment_correct=int(ac.sum()),selected_correct=int(bc.sum()))
            classes.append(row)
            paired.append(dict(case=cid,class_index=c,group=group,subgroup=subgroup,count=n,
                both_correct=int((ac&bc).sum()),rescued=int((~ac&bc).sum()),harmed=int((ac&~bc).sum()),both_wrong=int((~ac&~bc).sum()),
                wrong_to_different_wrong=int((~ac&~bc&(pred['moment'][cm]!=pred['selected'][cm])).sum()),
                moment_errors_to_old=int((~ac&(pred['moment'][cm]<old_n)).sum()),moment_errors_to_new=int((~ac&(pred['moment'][cm]>=old_n)).sum()),
                selected_errors_to_old=int((~bc&(pred['selected'][cm]<old_n)).sum()),selected_errors_to_new=int((~bc&(pred['selected'][cm]>=old_n)).sum())))
        case_rows=[r for r in classes if r['case']==cid]
        for mode in models:
            for group in ('all','old','new','accepted_old','fallback_old'):
                cr=[r for r in case_rows if group=='all' or r['group']==group or r['subgroup']==group]
                assert cr
                ba=sum(r[mode+'_correct']/r['count'] for r in cr)/len(cr)
                initial=sum(r['start_correct']/r['count'] for r in cr)/len(cr)
                rows.append(dict(case=cid,mode=mode,group=group,classes=len(cr),count=sum(r['count'] for r in cr),balanced_accuracy=ba,
                    accuracy=sum(r[mode+'_correct'] for r in cr)/sum(r['count'] for r in cr),forgetting_from_shared_expanded_start=initial-ba if group in ('old','accepted_old','fallback_old') else None))
        baseline.append(dict(case=cid,old_only_BA=sum(float((oldonly[y[oldmask]==c]==c).to(DT).mean()) for c in range(old_n))/old_n,
            expanded_start_old_BA=next(r['balanced_accuracy'] for r in rows if r['case']==cid and r['mode']=='start' and r['group']=='old')))
        gates.append(dict(case=cid,**gate(case_rows)))
        eval_info.append(dict(case=cid,seed=cfg['seed']+20000,counts=counts[:seen],old_classes=old_n,seen_classes=seen,all_four_trajectories_sealed_before_generation=True))
    return dict(status='COMPLETED',rows=rows,class_rows=classes,paired=paired,gates=gates,baseline=baseline,evaluation=eval_info,
        joint_pass=all(g['pass_strict'] for g in gates),adapter_updates=0,memory_updates=0,evaluation_seconds=time.monotonic()-start)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=unit_checks(lock) if cfg['kind']=='unit' else evaluate(lock,journal) if cfg['kind']=='evaluation' else train_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,val_accessed=False,test_accessed=False)
    assert not result['GPU_initialized'];return result
