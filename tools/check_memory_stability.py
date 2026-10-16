"""Frozen two-seed paired replication with balanced, sealed evaluation."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from check_fullslot_retention_memory import train, compress
from check_reserved_competition_memory import DT, initial_model, clone_model, frame, scores, lowrank_fixture
import check_weight_refinement as refinement


def uncertainty(rows,old_n,seed,draws=10000):
    rng=np.random.default_rng(seed);sample=[]
    for r in rows:
        n=r['count'];counts=[r[k] for k in ('both_correct','rescued','harmed','both_wrong')]
        assert sum(counts)==n and n>0 and min(counts)>=0
        b=rng.multinomial(n,np.array(counts,dtype=float)/n,size=draws)
        sample.append((b[:,1]-b[:,2])/n)
    sample=np.stack(sample,axis=1);out=[]
    for group,idx in [('all',list(range(len(rows)))),('old',list(range(old_n))),('new',list(range(old_n,len(rows))))]:
        v=sample[:,idx].mean(1);point=np.mean([(rows[i]['rescued']-rows[i]['harmed'])/rows[i]['count'] for i in idx])
        low,high=np.quantile(v,[.025,.975],method='linear')
        out.append(dict(group=group,BA_difference=float(point),lower=float(low),upper=float(high),confidence=.95,bootstrap_draws=draws,bootstrap_seed=seed))
    return out


def decision(intervals):
    g={v['group']:v for v in intervals};positive=g['all']['BA_difference']>1e-12
    point_cost=all(g[k]['BA_difference']>=-.01-1e-12 for k in ('old','new'))
    interval_cost=all(g[k]['lower']>=-.01-1e-12 for k in ('old','new'))
    interval_gain=g['all']['lower']>1e-12
    status='SUPPORTED' if positive and point_cost and interval_cost and interval_gain else 'COST_EXCEEDS_LIMIT' if not point_cost else 'NO_POSITIVE_GAIN' if not positive else 'INCONCLUSIVE'
    return dict(status=status,positive_overall_BA=positive,point_cost_acceptable=point_cost,interval_cost_acceptable=interval_cost,interval_gain_positive=interval_gain)


def unit_checks():
    result=refinement.unit_checks()
    identical=[dict(count=1000,both_correct=500,rescued=0,harmed=0,both_wrong=500) for _ in range(3)]
    a=uncertainty(identical,2,352,100);assert all(v['BA_difference']==v['lower']==v['upper']==0 for v in a)
    assert decision(a)['status']=='NO_POSITIVE_GAIN'
    positive=[dict(count=1000,both_correct=0,rescued=1000,harmed=0,both_wrong=0) for _ in range(3)]
    b=uncertainty(positive,2,352,100);assert all(v['BA_difference']==v['lower']==v['upper']==1 for v in b)
    assert decision(b)['status']=='SUPPORTED'
    bad=[dict(v) for v in b];bad[2].update(BA_difference=-.02,lower=-.03,upper=-.01)
    assert decision(bad)['status']=='COST_EXCEEDS_LIMIT'
    uncertain=[dict(v) for v in b];uncertain[2].update(BA_difference=0,lower=-.02,upper=.02)
    assert decision(uncertain)['status']=='INCONCLUSIVE'
    result.update(bootstrap_contingency_checked=True,group_cost_status_checked=True,adapter_updates=0,memory_updates=0)
    return result


def train_case(cfg,lock,journal):
    root=Path(os.environ['Q352_ROOT']);start=time.monotonic();old_n=cfg['task_sizes'][0];seen=sum(cfg['task_sizes'][:2])
    x,y,counts=lowrank_fixture(cfg);current={c:x[y==c] for c in range(old_n)}
    q=frame(cfg['dimension'],cfg['classes'],cfg['seed']);initial=initial_model(cfg['dimension'],cfg['seed'])
    journal(dict(phase='shared_teacher',completed_updates=0))
    teacher,teacher_cost=train(initial,current,{},q,list(range(old_n)),steps=48,full=False)
    journal(dict(phase='moment_construction',completed_updates=0))
    moment,moment_cost=compress(current,cfg['seed'],'moment')
    torch.save(dict(case=cfg,teacher=teacher,frame=q,memories=dict(moment=moment)),root/(cfg['id']+'.state.private.pt'))
    # Reuse the frozen weight construction and endpoint gate without changing their science.
    os.environ['Q350_PRIOR']=str(root);os.environ['Q350_ROOT']=str(root)
    refined=refinement.run_scale(cfg,lock,journal)
    saved=torch.load(root/(cfg['id']+'.candidate.private.pt'),map_location='cpu',weights_only=True)
    selected=saved['selected'];accepted=[v['class_index'] for v in saved['acceptance'] if v['accepted']]
    assert saved['case']==cfg and refined['actual_memory_updates']==48
    models={};training=[];next_classes={c:x[y==c] for c in range(old_n,seen)};original=clone_model(teacher)
    frozen={m:{c:(z.clone(),p.clone()) for c,(z,p) in mem.items()} for m,mem in [('moment',moment),('selected',selected)]}
    for mode,mem in [('moment',moment),('selected',selected)]:
        journal(dict(phase='next_stage_training',mode=mode,completed_updates=0))
        models[mode],cost=train(teacher,next_classes,mem,q,list(range(seen)),steps=48,full=False)
        training.append(dict(mode=mode,**cost));journal(dict(phase='next_stage_training',mode=mode,completed_updates=48))
    assert all(torch.equal(a,b) for a,b in zip(teacher,original))
    assert all(torch.equal(a,b) for m,mem in [('moment',moment),('selected',selected)] for c,zp in mem.items() for a,b in zip(zp,frozen[m][c]))
    torch.save(dict(case=cfg,models=models,frame=q,accepted_classes=accepted,adapter_updates=144,memory_updates=96),root/(cfg['id']+'.sealed.private.pt'))
    return dict(status='COMPLETED',case=cfg,accepted_classes=accepted,acceptance=refined['acceptance'],refinement=refined,teacher_training=teacher_cost,moment_construction=moment_cost,training=training,
        adapter_updates=144,memory_updates=96,sealed=True,evaluation_generated=False,teacher_and_memories_unchanged=True,old_support_used_in_next_stage_training=False,pipeline_seconds=time.monotonic()-start)


def predict(x,model,q,seen):
    with torch.no_grad():return torch.cat([scores(x[i:i+512],model,q)[:,:seen].argmax(1) for i in range(0,len(x),512)])


def evaluate(lock,journal):
    start=time.monotonic();root=Path(os.environ['Q352_ROOT']);program=json.loads((root/'CPU_PROGRAM.json').read_text());states={}
    for cfg in lock['cases'][1:-1]:
        assert next(v for v in program['cases'] if v['id']==cfg['id'])['status']=='COMPLETED'
        s=torch.load(root/(cfg['id']+'.sealed.private.pt'),map_location='cpu',weights_only=True)
        assert s['case']==cfg and s['adapter_updates']==144 and s['memory_updates']==96;states[cfg['id']]=s
    summaries=[];class_rows=[];paired=[];intervals=[];decisions=[];evaluation=[]
    for base in lock['base_cases']:
        scale=base['id'];old_n=base['task_sizes'][0];seen=sum(base['task_sizes'][:2]);journal(dict(phase='balanced_evaluation',scale=scale))
        evaluation_cfg=dict(base,seed=lock['evaluation_seeds'][scale],n=base['classes']*1000,count_weights=[1]*base['classes'])
        x,y,counts=lowrank_fixture(evaluation_cfg);keep=y<seen;x=x[keep];y=y[keep];assert counts==[1000]*base['classes']
        old=torch.load(Path(os.environ['Q352_TRANSFER'])/(scale+'.sealed.private.pt'),map_location='cpu',weights_only=True)
        original=torch.load(Path(os.environ['Q352_PRIOR'])/(scale+'.state.private.pt'),map_location='cpu',weights_only=True)
        assert old['case']==original['case']==base
        scenes=[(scale+'_development',dict(old,frame=original['frame']),False)]
        scenes +=[(cfg['id'],states[cfg['id']],True) for cfg in lock['cases'][1:-1] if cfg['scale']==scale]
        for sid,state,confirmatory in scenes:
            predictions={m:predict(x,state['models'][m],state['frame'],seen) for m in ('moment','selected')};cr=[];pp=[]
            for c in range(seen):
                cm=y==c;a=predictions['moment'][cm];b=predictions['selected'][cm];ac=a==c;bc=b==c
                tag='accepted_old' if c in state['accepted_classes'] else 'fallback_old' if c<old_n else 'new'
                row=dict(scene=sid,scale=scale,training_seed=state['case']['seed'],confirmatory=confirmatory,class_index=c,group='old' if c<old_n else 'new',subgroup=tag,count=int(cm.sum()),moment_correct=int(ac.sum()),selected_correct=int(bc.sum()),recall_difference=float((bc.to(DT)-ac.to(DT)).mean()))
                cr.append(row)
                pp.append(dict(row,both_correct=int((ac&bc).sum()),rescued=int((~ac&bc).sum()),harmed=int((ac&~bc).sum()),both_wrong=int((~ac&~bc).sum()),wrong_to_different_wrong=int((~ac&~bc&(a!=b)).sum()),moment_errors_to_old=int((~ac&(a<old_n)).sum()),moment_errors_to_new=int((~ac&(a>=old_n)).sum()),selected_errors_to_old=int((~bc&(b<old_n)).sum()),selected_errors_to_new=int((~bc&(b>=old_n)).sum())))
            class_rows.extend(cr);paired.extend(pp)
            ci=uncertainty(pp,old_n,lock['bootstrap_seeds'][sid],lock['bootstrap_draws']);intervals.extend(dict(scene=sid,scale=scale,confirmatory=confirmatory,**v) for v in ci)
            decisions.append(dict(scene=sid,scale=scale,confirmatory=confirmatory,**decision(ci)))
            for group in ('all','old','new','accepted_old','fallback_old'):
                subset=[v for v in cr if group=='all' or v['group']==group or v['subgroup']==group]
                means={m:sum(v[m+'_correct']/v['count'] for v in subset)/len(subset) if subset else None for m in ('moment','selected')}
                summaries.append(dict(scene=sid,scale=scale,training_seed=state['case']['seed'],confirmatory=confirmatory,group=group,classes=len(subset),count=sum(v['count'] for v in subset),moment_BA=means['moment'],selected_BA=means['selected'],BA_difference=means['selected']-means['moment'] if subset else None))
        evaluation.append(dict(scale=scale,seed=evaluation_cfg['seed'],samples_per_seen_class=1000,seen_classes=seen,total_seen_samples=len(y),all_eight_new_trajectories_sealed_before_generation=True,shared_across_all_three_seeds_and_both_arms=True))
    return dict(status='COMPLETED',summaries=summaries,class_rows=class_rows,paired=paired,intervals=intervals,decisions=decisions,evaluation=evaluation,adapter_updates=0,memory_updates=0,evaluation_seconds=time.monotonic()-start)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=unit_checks() if cfg['kind']=='unit' else evaluate(lock,journal) if cfg['kind']=='evaluation' else train_case(cfg,lock,journal)
    r.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,val_accessed=False,test_accessed=False)
    assert not r['GPU_initialized'];return r
