"""One sealed batch: checkpoint-only inference followed by aggregate paired reporting."""
import csv
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score,roc_auc_score
from torch.utils.data import DataLoader
from run_medical_v2 import Images,MedicalLearner,args_for,seed_all,sha,write_json,headnorm_alpha

def mean_or_none(values):return float(np.mean(values)) if len(values) else None

def metrics(raw,targets,rows,order,known,groups):
    seen=raw.shape[1];pred=raw.argmax(1);correct=pred==targets
    per_class=[]
    for c in range(seen):
        mask=targets==c;lesions=sorted({r['lesion_id'] for i,r in enumerate(rows) if mask[i]})
        lesion_values=[correct[[i for i,r in enumerate(rows) if mask[i] and r['lesion_id']==lesion]].mean() for lesion in lesions]
        per_class.append(dict(head_index=c,original_label=order[c],n_images=int(mask.sum()),n_lesions=len(lesions),
                              recall=float(correct[mask].mean()*100),lesion_equal_recall=float(np.mean(lesion_values)*100)))
    recalls=[r['recall'] for r in per_class]
    result=dict(balanced_accuracy=float(np.mean(recalls)),lesion_equal_balanced_accuracy=float(np.mean([r['lesion_equal_recall'] for r in per_class])),
                accuracy=float(correct.mean()*100),macro_f1=float(f1_score(targets,pred,labels=range(seen),average='macro',zero_division=0)*100),
                n_test_images=len(rows),n_test_lesions=len({r['lesion_id'] for r in rows}),n_test_components=len({r['identity_component'] for r in rows}))
    prob=torch.tensor(raw).softmax(1).numpy()
    try:result['macro_ovr_auroc']=float(roc_auc_score(targets,prob,labels=list(range(seen)),average='macro',multi_class='ovr'))
    except ValueError as error:result['macro_ovr_auroc']=None;result['auroc_reason']=str(error)
    for name,labels in groups.items():result[name]=mean_or_none([r['recall'] for r in per_class if r['original_label'] in labels])
    for scope,indices in [('old',range(known)),('current',range(known,seen))]:
        mask=np.isin(targets,list(indices));n=int(mask.sum())
        result[scope+'_macro_recall']=mean_or_none([recalls[c] for c in indices])
        result[scope+'_accuracy']=float(correct[mask].mean()*100) if n else None
        result[scope+'_n_images']=n
        if n:
            subset=raw[mask][:,list(indices)];restricted=np.array(list(indices))[subset.argmax(1)]
            result['oracle_restricted_'+scope+'_accuracy']=float((restricted==targets[mask]).mean()*100)
        else:result['oracle_restricted_'+scope+'_accuracy']=None
        for group in ('head_rank2','tail_rank2'):
            cs=[c for c in indices if order[c] in groups[group]]
            result[scope+'_'+group]=mean_or_none([recalls[c] for c in cs])
            result[scope+'_'+group+'_n_classes']=len(cs)
            result[scope+'_'+group+'_n_images']=int(np.isin(targets,cs).sum())
    old=targets<known;current=~old
    cc=int((correct&current).sum());co=int((current&(pred<known)).sum());cw=int((current&~correct&(pred>=known)).sum())
    assert cc+co+cw==int(current.sum())
    for name,count,denom in [('correct_current',cc,int(current.sum())),('current_to_old',co,int(current.sum())),('current_to_wrong_current',cw,int(current.sum())),('old_to_current',int((old&(pred>=known)).sum()),int(old.sum()))]:
        result[name+'_count']=count;result[name+'_rate']=count/denom*100 if denom else None
    for metric in ('macro_recall','accuracy'):
        a=result['old_'+metric];b=result['current_'+metric]
        result['old_current_hm_'+metric]=2*a*b/(a+b) if a is not None and a+b>0 else (0. if a==0 and b==0 else None)
    return result,per_class

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def aggregate(session_rows,class_rows,out,groups):
    final=[];forgetting=[]
    for seed in (1993,1994,1995):
        for variant in ('M0','M1','M2','M3'):
            rows=sorted([r for r in session_rows if r['train_seed']==seed and r['eval_variant']==variant],key=lambda r:r['session'])
            assert len(rows)==3
            last=rows[-1]
            row=dict(train_seed=seed,eval_variant=variant,Final_BA=last['balanced_accuracy'],Average_BA=float(np.mean([r['balanced_accuracy'] for r in rows])),
                     Incremental_Average_BA=float(np.mean([r['balanced_accuracy'] for r in rows[1:]])),
                     tail_rank2=last['tail_rank2'],old_tail_rank2=last['old_tail_rank2'],Final_Lesion_BA=last['lesion_equal_balanced_accuracy'])
            for label in range(8):
                cr=sorted([r for r in class_rows if r['train_seed']==seed and r['eval_variant']==variant and r['original_label']==label],key=lambda r:r['session'])
                eligible=len(cr)>1
                forgetting.append(dict(train_seed=seed,eval_variant=variant,original_label=label,first_session=cr[0]['session'],eligible=eligible,
                                       first_recall=cr[0]['recall'],final_recall=cr[-1]['recall'],first_minus_final=cr[0]['recall']-cr[-1]['recall'] if eligible else None,
                                       max_seen_minus_final=max(r['recall'] for r in cr)-cr[-1]['recall'] if eligible else None))
            eligible=[r for r in forgetting if r['train_seed']==seed and r['eval_variant']==variant and r['eligible']]
            row['forgetting_eligible_classes']=len(eligible);row['forgetting_mean']=float(np.mean([r['max_seen_minus_final'] for r in eligible]))
            tails=[r for r in eligible if r['original_label'] in groups['tail_rank2']]
            row['tail_forgetting_eligible_classes']=len(tails);row['tail_forgetting_mean']=mean_or_none([r['max_seen_minus_final'] for r in tails])
            final.append(row)
    contrasts={'M3-M0':(1,0,0,-1),'M1-M0':(0,0,1,-1),'M2-M0':(0,1,0,-1),
               'M3-M2':(1,-1,0,0),'M3-M1':(1,0,-1,0),'interaction':(1,-1,-1,1)}
    paired=[]
    for metric in ('Final_BA','Average_BA','tail_rank2','old_tail_rank2'):
        for name,weights in contrasts.items():
            values=[]
            for seed in (1993,1994,1995):
                by={r['eval_variant']:r[metric] for r in final if r['train_seed']==seed}
                value=sum(w*by[m] for w,m in zip(weights,('M3','M2','M1','M0')) if w) if all(by[m] is not None for w,m in zip(weights,('M3','M2','M1','M0')) if w) else None
                values.append(value);paired.append(dict(metric=metric,contrast=name,train_seed=seed,value=value,mean=None,std_ddof1=None))
            valid=all(v is not None for v in values)
            paired.append(dict(metric=metric,contrast=name,train_seed='paired_summary',value=None,mean=float(np.mean(values)) if valid else None,std_ddof1=float(np.std(values,ddof=1)) if valid else None))
    overall=[r['value'] for r in paired if r['metric']=='Final_BA' and r['contrast']=='M3-M0' and r['train_seed']!='paired_summary']
    average=next(r['mean'] for r in paired if r['metric']=='Average_BA' and r['contrast']=='M3-M0' and r['train_seed']=='paired_summary')
    tail=next(r['mean'] for r in paired if r['metric']=='tail_rank2' and r['contrast']=='M3-M0' and r['train_seed']=='paired_summary')
    zeros=[]
    for r in class_rows:
        if r['eval_variant']=='M3' and r['head_index']>=r['old_classes'] and r['recall']==0:
            base=next(b for b in class_rows if b['eval_variant']=='M0' and b['train_seed']==r['train_seed'] and b['session']==r['session'] and b['original_label']==r['original_label'])
            if base['recall']>0:zeros.append((r['train_seed'],r['session'],r['original_label']))
    status='COMPLETE_CONSISTENT_SIGNAL' if all(v>0 for v in overall) and average>0 and tail>0 and not zeros else ('COMPLETE_MIXED_SIGNAL' if np.mean(overall)>0 else 'COMPLETE_NO_TRANSFER')
    write_csv(out/'session_metrics.csv',session_rows);write_csv(out/'per_class_metrics.csv',class_rows)
    write_csv(out/'paired_results.csv',paired);write_csv(out/'repeat_results.csv',final);write_csv(out/'forgetting.csv',forgetting)
    variant_summary=[]
    for variant in ('M0','M1','M2','M3'):
        for field in ('Final_BA','Average_BA','Incremental_Average_BA','tail_rank2','old_tail_rank2','Final_Lesion_BA','forgetting_mean','tail_forgetting_mean'):
            values=[r[field] for r in final if r['eval_variant']==variant];valid=all(v is not None for v in values)
            variant_summary.append(dict(eval_variant=variant,metric=field,mean=float(np.mean(values)) if valid else None,std_ddof1=float(np.std(values,ddof=1)) if valid else None,n_repeats=3))
    write_csv(out/'variant_summary.csv',variant_summary)
    conclusion=dict(status=status,primary_paired_final_ba=overall,new_zero_current_classes=zeros,test_batch_count=1,trajectories=6,session_checkpoints=18)
    write_json(out/'conclusion.json',conclusion)
    lines=['# V2 final results','',f'Status: {status}. Six fixed trajectories and 18 final-session checkpoints; one sealed test batch.',
           '', '| Seed | Variant | Final BA | Average BA | Final tail recall | Lesion-equal BA |','|---|---|---:|---:|---:|---:|']
    for r in final:lines.append(f"| {r['train_seed']} | {r['eval_variant']} | {r['Final_BA']:.3f} | {r['Average_BA']:.3f} | {r['tail_rank2']:.3f} | {r['Final_Lesion_BA']:.3f} |")
    lines += ['', 'All six paired contrasts and ddof=1 summaries are in paired_results.csv. Per-stage, per-class and eligible-class forgetting tables are supplied separately.',
              '', 'V1 was blocked before training and has no performance results. V2 uses the documented-lesion-disjoint subset, with missing-ID selection bias and UNKNOWN patient isolation/development exposure. Numerical finding IDs have no verified disease-name mapping.',
              '', 'APART (raw-count capacity-adapted), fixed new ImageNet-pretrained initialization, and legacy_effective_v1 optimization. This is not an unchanged APART, historical CIFAR checkpoint, full ConCM, patient-level or external clinical reproduction. Three paired training repeats do not establish statistical significance.',
              '', 'There was no hyperparameter search, checkpoint selection, early stopping for weak results, image replay, or test-driven retraining. No further experiments are authorized by this report.']
    (out/'V2_FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return conclusion

def evaluate(config):
    output=Path(config['output']);lock=json.loads((output/'ALL_CHECKPOINTS_LOCK.json').read_text())
    assert lock['protocol_sha256']==config['protocol_sha256'] and lock['code_commit']==config['code_commit']
    for path,expected in config['code_sha256'].items():assert sha(Path(__file__).resolve().parents[1]/path)==expected,'BLOCKED_EVALUATOR_CODE_DRIFT'
    assert len(lock['checkpoints'])==18 and len({(r['seed'],r['branch'],r['session']) for r in lock['checkpoints']})==18
    for entry in lock['checkpoints']:assert sha(entry['path'])==entry['sha256']
    marker=output/'TEST_BATCH_STARTED.json'
    with marker.open('x') as f:json.dump(dict(started=time.time(),checkpoint_lock_sha=sha(output/'ALL_CHECKPOINTS_LOCK.json')),f)
    result=output/'results';result.mkdir(exist_ok=False);private=output/'predictions_private';private.mkdir(exist_ok=False)
    summary=json.loads((Path(config['protocol'])/'V2_DATASET_SUMMARY.json').read_text());sessions=[];classes=[]
    start=time.monotonic();prediction_count=0
    for entry in lock['checkpoints']:
        seed=entry['seed'];branch=entry['branch'];task=entry['session'];known=[0,4,6][task];seen=[4,6,8][task]
        seed_all(seed);args,order=args_for(config,seed,branch,False)
        learner=MedicalLearner(args,config,order,output/'evaluation_state');learner.restore(entry['path']);learner._network.eval()
        ds=Images(Path(config['protocol'])/'test.csv',config['images'],order,range(seen),False)
        loader=DataLoader(ds,batch_size=48,shuffle=False,num_workers=8,generator=torch.Generator().manual_seed(seed))
        outputs=[];targets=[]
        with torch.no_grad():
            for _,x,y in loader:
                o=learner._network(x.cuda(),task_id=task,train=False)
                outputs.append((o['logits']+o['logits_few'])[:,:seen].cpu());targets.append(y)
        raw=torch.cat(outputs).numpy();target=torch.cat(targets).numpy();assert np.isfinite(raw).all()
        prediction_count+=len(target)
        np.savez(private/f'{seed}_{branch}_s{task}.npz',raw_logits=raw,targets=target)
        alpha=headnorm_alpha(learner._network.backbone,seen,known)
        for hn in (False,True):
            scores=raw.copy()
            if hn:scores[:,:known]*=alpha
            variant=('M2' if branch=='C' else 'M0') if not hn else ('M3' if branch=='C' else 'M1')
            common=dict(protocol_id=summary['protocol_id'],run_id='q8m7',code_sha=config.get('code_commit',config['code_sha256']['tools/run_medical_v2.py']),checkpoint_sha=entry['sha256'],split='test',repeat_id=seed-1992,order_seed=seed,train_seed=seed,train_branch=branch,eval_variant=variant,session=task,seen_classes=seen,old_classes=known,current_classes=seen-known)
            m,pc=metrics(scores,target,ds.rows,order,known,summary['frequency_groups'])
            sessions.append(dict(**common,alpha=alpha if hn else 1.,**m));classes += [dict(**common,**c) for c in pc]
        del learner,outputs,targets,o,x;gc.collect();torch.cuda.empty_cache()
    conclusion=aggregate(sessions,classes,result,summary['frequency_groups'])
    write_json(result/'evaluation_usage.json',dict(seconds=time.monotonic()-start,model_test_predictions=prediction_count,batch_evaluations=1))
    return conclusion
