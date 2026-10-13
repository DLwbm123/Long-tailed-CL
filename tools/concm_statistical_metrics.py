"""Anonymous unconstrained paired scores; all old/new positive and negative changes."""
import numpy as np

FIELDS = ('final_balanced_accuracy', 'average_incremental_balanced_accuracy',
          'final_tail_recall', 'forgetting', 'final_old_recall', 'final_new_recall')


def summary(x, small=False):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite margin')
    v = dict(n=len(x), positive=int((x>0).sum()), zero=int((x==0).sum()), negative=int((x<0).sum()))
    v.update(mean=float(x.mean()) if len(x) and not small else None,
             minimum=float(x.min()) if len(x) and not small else None,
             maximum=float(x.max()) if len(x) and not small else None,
             continuous_suppressed=bool(small))
    return v


def true_margin(scores, y):
    other=scores.copy();other[np.arange(len(y)),y]=-np.inf
    return scores[np.arange(len(y)),y]-other.max(1)


def stage(off, on, raw, labels, data, task):
    seen=data['order'][:sum(data['task_sizes'][:task])]
    old=sum(data['task_sizes'][:task-1]);k=len(seen);n=len(labels)
    counts={str(c):int((labels==c).sum()) for c in seen}
    if (counts!={str(c):data['val_counts'][str(c)] for c in seen}
            or off.shape!=(n,k) or on.shape!=off.shape or raw.shape!=off.shape
            or not all(np.isfinite(z).all() for z in (off,on,raw))):
        raise ValueError('Stage labels, support or scores differ')
    y=np.array([seen.index(int(c)) for c in labels]);po=off.argmax(1);pn=on.argmax(1)
    mo,mn,mr=[true_margin(z,y) for z in (off,on,raw)]
    new=y>=old
    paired={};margins={};flows={};metrics={}
    groups=[('all',np.ones(n,dtype=bool)),('old',~new),('new',new)]+[(str(c),labels==c) for c in seen]
    for name,mask in groups:
        co=po[mask]==y[mask];cn=pn[mask]==y[mask];changed=po[mask]!=pn[mask]
        a=dict(n=int(mask.sum()),changed=int(changed.sum()),rescued=int((~co&cn).sum()),
            harmed=int((co&~cn).sum()),wrong_to_different_wrong=int((~co&~cn&changed).sum()),
            correct_off=int(co.sum()),correct_on=int(cn.sum()),net_correct=int(cn.sum()-co.sum()))
        assert a['changed']==a['rescued']+a['harmed']+a['wrong_to_different_wrong']
        assert a['net_correct']==a['rescued']-a['harmed']
        paired[name]=a;small=name not in ('all','old','new') and a['n']<5
        margins[name]={key:summary(value[mask],small) for key,value in
            [('off',mo),('on',mn),('applied_delta',mn-mo),('raw_delta',mr-mo)]}
    ranked=sorted(data['order'],key=lambda c:(-data['train_counts'][str(c)],c))
    tail=set(ranked[len(ranked)//2:])
    for mode,pred in [('OFF',po),('ON',pn)]:
        correct=pred==y;pred_old=pred<old
        flows[mode]=dict(old_correct=int((~new&correct).sum()),old_to_new=int((~new&~pred_old).sum()),
            old_to_old_wrong=int((~new&pred_old&~correct).sum()),new_correct=int((new&correct).sum()),
            new_to_old=int((new&pred_old).sum()),new_to_new_wrong=int((new&~pred_old&~correct).sum()))
        assert sum(flows[mode].values())==n
        recalls={str(c):float(correct[labels==c].mean()) for c in seen}
        f1=[]
        for ix in range(k):
            tp=int(((pred==ix)&(y==ix)).sum());fp=int(((pred==ix)&(y!=ix)).sum());fn=int(((pred!=ix)&(y==ix)).sum())
            f1.append(2*tp/(2*tp+fp+fn))
        metrics[mode]=dict(task=task,seen=seen,validation_n=n,per_class_n=counts,per_class_recall=recalls,
            accuracy=float(correct.mean()),balanced_accuracy=float(np.mean(list(recalls.values()))),
            tail_recall=float(np.mean([recalls[str(c)] for c in seen if c in tail])) if tail.intersection(seen) else None,
            macro_F1=float(np.mean(f1)))
    rms=lambda a:float(np.sqrt(np.mean(a.astype(np.float64)**2)))
    effect=dict(raw_rms=rms(raw-off),applied_rms=rms(on-off),
        applied_relative_rms=rms(on-off)/max(rms(off),1e-30),
        raw_new_margin_decreased=int(((mr-mo)[new]<0).sum()),
        new_margin_changed=int((mo[new]!=mn[new]).sum()),
        new_prediction_changed=int(sum(np.count_nonzero((po==c)!=(pn==c)) for c in range(old,k))),
        new_sample_wrong_label_changed=int((po[new]!=pn[new]).sum()),
        predicted_group_changed=int(((po<old)!=(pn<old)).sum()),
        old_internal_order_changed=int((raw[:,:old].argmax(1)!=off[:,:old].argmax(1)).sum()) if old else 0)
    return dict(task=task,old_count=old,metrics=metrics,paired=paired,margins=margins,flows=flows,effect=effect)


def aggregate(stages,data):
    if len(stages)!=len(data['task_sizes']):raise ValueError('Incomplete stage trajectory')
    for i,s in enumerate(stages):
        seen=data['order'][:sum(data['task_sizes'][:i+1])]
        if s['seen']!=seen or s['per_class_n']!={str(c):data['val_counts'][str(c)] for c in seen}:
            raise ValueError('Stage order/support mismatch')
    final=stages[-1];last_new=data['task_sizes'][-1]
    forgetting=[]
    for c in data['order']:
        earlier=[s['per_class_recall'][str(c)] for s in stages[:-1] if c in s['seen']]
        if earlier:forgetting.append(max(earlier)-final['per_class_recall'][str(c)])
    return dict(stages=stages,final_balanced_accuracy=final['balanced_accuracy'],
        average_incremental_balanced_accuracy=float(np.mean([s['balanced_accuracy'] for s in stages])),
        final_tail_recall=final['tail_recall'],forgetting=float(np.mean(forgetting)),
        final_old_recall=float(np.mean([final['per_class_recall'][str(c)] for c in data['order'][:-last_new]])),
        final_new_recall=float(np.mean([final['per_class_recall'][str(c)] for c in data['order'][-last_new:]])),
        final_accuracy=final['accuracy'],average_incremental_accuracy=float(np.mean([s['accuracy'] for s in stages])),
        final_macro_F1=final['macro_F1'],test_accessed=False,evaluation_split='development_validation')
