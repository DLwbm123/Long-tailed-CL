"""Aggregate scoring, fixed-set transition accounting and paired group bootstrap."""
import numpy as np
from next1_support import summarize

ORDER=[4,0,3,7,5,6,2,1]
TAIL={4,7,5,6}


def evaluate(stages,labels):
    reports=[];learning={}
    for task,scores in enumerate(stages,1):
        seen=ORDER[:2*task];mask=np.isin(labels,seen);y=labels[mask];p=np.asarray(seen)[scores.argmax(1)]
        recalls={str(c):float((p[y==c]==c).mean()) for c in seen}
        reports.append(dict(task=task,seen=seen,validation_n=len(y),accuracy=float((p==y).mean()),
            balanced_accuracy=float(np.mean(list(recalls.values()))),per_class_recall=recalls,
            per_class_n={str(c):int((y==c).sum()) for c in seen},tail_recall=float(np.mean([recalls[str(c)] for c in seen if c in TAIL])),
            score_diagnostics=summarize(scores,y,seen)))
    for c in ORDER:
        available=[r for r in reports if str(c) in r['per_class_recall']];v=[r['per_class_recall'][str(c)] for r in available]
        learning[str(c)]=dict(first_task=available[0]['task'],first_recall=v[0],historical_max_recall=max(v),final_recall=v[-1],
            first_minus_final=v[0]-v[-1],standard_forgetting=max(v[:-1])-v[-1] if len(v)>1 else None)
    final=reports[-1]['per_class_recall'];forgetting=np.mean([learning[str(c)]['standard_forgetting'] for c in ORDER[:6]])
    return dict(stages=reports,class_learning=learning,final_balanced_accuracy=reports[-1]['balanced_accuracy'],
        average_incremental_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reports])),
        final_tail_recall=reports[-1]['tail_recall'],forgetting=float(forgetting),
        final_old_recall=float(np.mean([final[str(c)] for c in ORDER[:6]])),
        final_new_recall=float(np.mean([final[str(c)] for c in ORDER[6:]])),test_accessed=False,independent_confirmation=False)


def transitions(stages,labels):
    rows=[]
    for task in (2,3,4):
        old=ORDER[:2*(task-1)];seen=ORDER[:2*task];new=seen[-2:]
        prev=stages[task-2];now=stages[task-1];yp=labels[np.isin(labels,old)];yn=labels[np.isin(labels,seen)]
        previous=np.asarray(old)[prev.argmax(1)];all_prediction=np.asarray(seen)[now.argmax(1)]
        restricted_old=np.asarray(old)[now[:,:len(old)].argmax(1)]
        restricted_new=np.asarray(new)[now[:,len(old):].argmax(1)]
        values=[]
        for c in old:
            a=int((previous[yp==c]==c).sum());b=int((restricted_old[yn==c]==c).sum());d=int((all_prediction[yn==c]==c).sum());n=int((yn==c).sum())
            row=dict(task=task,class_id=c,group='old',n=n,previous_correct=a,restricted_correct=b,all_correct=d,
                D_internal=(a-b)/n,D_competition=(b-d)/n,D_total=(a-d)/n)
            assert row['D_competition']>=0;values.append(row);rows.append(row)
        rows.append(dict(task=task,class_id='macro',group='old',n=sum(v['n'] for v in values),
            previous_correct=sum(v['previous_correct'] for v in values),restricted_correct=sum(v['restricted_correct'] for v in values),all_correct=sum(v['all_correct'] for v in values),
            **{key:float(np.mean([v[key] for v in values])) for key in ('D_internal','D_competition','D_total')}))
        for c in new:
            n=int((yn==c).sum());a=int((restricted_new[yn==c]==c).sum());b=int((all_prediction[yn==c]==c).sum())
            rows.append(dict(task=task,class_id=c,group='new',n=n,previous_correct=None,restricted_correct=a,all_correct=b,
                D_internal=None,D_competition=(a-b)/n,D_total=None))
        selected=[v for v in rows if v['task']==task and v['group']=='new']
        rows.append(dict(task=task,class_id='macro',group='new',n=sum(v['n'] for v in selected),previous_correct=None,
            restricted_correct=sum(v['restricted_correct'] for v in selected),all_correct=sum(v['all_correct'] for v in selected),D_internal=None,
            D_competition=float(np.mean([v['D_competition'] for v in selected])),D_total=None))
    return rows


def differences(a,b):
    keys=['final_balanced_accuracy','final_tail_recall','forgetting','final_new_recall','final_old_recall']
    d={k:a[k]-b[k] for k in keys}
    d['passed']=d[keys[0]]>=.01 and d[keys[1]]>=-.005 and d[keys[2]]<=.01 and d[keys[3]]>=-.01
    return d


def bootstrap(scores,labels,groups,budget):
    if len(groups)!=len(labels) or any(not g for g in groups):return dict(status='NA',reason='Missing identity groups')
    unique=sorted(set(groups));codes=np.asarray([unique.index(g) for g in groups]);rng=np.random.default_rng(74102)
    values={name:[] for name in scores};invalid=0
    truth=np.asarray([ORDER.index(int(c)) for c in labels]);masks=[truth<2*t for t in (1,2,3,4)]
    predictions={a:[s.argmax(1) for s in stages] for a,stages in scores.items()}
    for iteration in range(2000):
        if iteration%25==0:budget()
        weights=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[codes]
        denominator=np.bincount(truth,weights,minlength=8)
        if (denominator==0).any():invalid+=1;continue
        for arm,pred in predictions.items():
            recalls=np.full((4,8),np.nan)
            for i,mask in enumerate(masks):
                correct=pred[i]==truth[mask];numerator=np.bincount(truth[mask],weights[mask]*correct,minlength=8)
                recalls[i,:2*(i+1)]=numerator[:2*(i+1)]/denominator[:2*(i+1)]
            final=recalls[-1];values[arm].append([final.mean(),final[[0,3,4,5]].mean(),
                (np.nanmax(recalls[:3,:6],axis=0)-final[:6]).mean(),final[6:].mean(),final[:6].mean()])
    keys=['final_balanced_accuracy','final_tail_recall','forgetting','final_new_recall','final_old_recall']
    def intervals(a):
        return {key:dict(valid_replicates=len(a),low=float(np.quantile(a[:,i],.025)),high=float(np.quantile(a[:,i],.975)))
            if len(a)>=1900 else dict(valid_replicates=len(a),low=None,high=None,reason='Fewer than1900 valid draws') for i,key in enumerate(keys)}
    arrays={a:np.asarray(v).reshape(-1,5) for a,v in values.items()}
    return dict(status='COMPLETE',draws=2000,invalid_draws=invalid,seed=74102,identity_groups=len(unique),
        confidence=.95,absolute={a:intervals(v) for a,v in arrays.items()},
        versus_R={a:intervals(v-arrays['R']) for a,v in arrays.items() if a!='R'},
        versus_C0={a:intervals(v-arrays['C0']) for a,v in arrays.items() if a not in ('R','C0')},
        independent_confirmation=False,conditional_development_uncertainty_only=True)
