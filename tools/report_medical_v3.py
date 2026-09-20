"""Aggregate validation-only paired evidence; separate facts from causal claims."""
import csv
import json
from pathlib import Path
import numpy as np
from run_medical_v2 import write_json,sha
from evaluate_medical_v2 import write_csv
from diagnose_medical_v3 import SEEDS

def flatten(d,prefix=''):
    r={}
    for k,v in d.items():
        key=prefix+k
        if isinstance(v,dict):r.update(flatten(v,key+'.'))
        elif isinstance(v,list):r[key]=json.dumps(v)
        else:r[key]=v
    return r

def report(config):
    root=Path(config['output']);out=root/'results';out.mkdir(exist_ok=True)
    metrics=[];perclass=[];epochs=[];checkpoints=[]
    for seed in SEEDS:
        for branch in ('B','C','D','E'):
            for task in (0,1,2):
                inherited=branch in ('D','E') and task==0
                parent={'D':'B','E':'C'}.get(branch,branch)
                if branch in ('B','C') or inherited:
                    value=json.loads((root/'p0'/f'{seed}_{parent}_s{task}.json').read_text())['diagnostics']['val']
                else:value=json.loads((root/f'{seed}_{branch}'/f'session{task}_val.json').read_text())
                prefix={'seed':seed,'branch':branch,'session':task,'inherited_s0':inherited,'new_training_epochs':10 if branch in ('D','E') and task else 0}
                metrics += [dict(**prefix,**m) for m in value['metrics']]
                perclass += [dict(**prefix,**p) for p in value['per_class']]
        for branch in ('D','E'):
            d=root/f'{seed}_{branch}';rows=[json.loads(line) for line in (d/'epochs.jsonl').read_text().splitlines()]
            assert len(rows)==20 and {(r['session'],r['epoch']) for r in rows}=={(t,e) for t in (1,2) for e in range(1,11)},'BLOCKED_EPOCH_MATRIX'
            for r in rows:
                c=dict(r['components']);c.pop('real_stream_sha256',None)
                epochs.append(dict(seed=seed,branch=branch,session=r['session'],epoch=r['epoch'],**flatten(c)))
            checkpoints+=json.loads((d/'TRAJECTORY_COMPLETE.json').read_text())['sessions']
        d=[json.loads(x) for x in (root/f'{seed}_D/epochs.jsonl').read_text().splitlines()]
        e=[json.loads(x) for x in (root/f'{seed}_E/epochs.jsonl').read_text().splitlines()]
        assert all(a['components']['real_stream_sha256']==b['components']['real_stream_sha256'] for a,b in zip(d,e)), 'BLOCKED_PAIRED_REAL_STREAM'
    assert len(checkpoints)==12 and len(epochs)==120
    primary=[r for r in metrics if r['head']=='sum'];paired=[];means=[];decisions={}
    fields=['balanced_accuracy','old_macro_recall','current_macro_recall','old_current_hm_macro_recall','tail_rank2','old_tail_rank2','current_tail_rank2','restricted_current_ba','oracle_restricted_current_accuracy','current_to_old_rate','current_to_wrong_current_rate','old_to_current_rate']
    for a,b in [('E','C'),('D','B'),('E','D')]:
        for task in (1,2):
            for field in fields:
                values=[]
                for seed in SEEDS:
                    x=next(r for r in primary if (r['seed'],r['branch'],r['session'])==(seed,a,task))
                    y=next(r for r in primary if (r['seed'],r['branch'],r['session'])==(seed,b,task))
                    v=x[field]-y[field] if x[field] is not None and y[field] is not None else None
                    paired.append({'contrast':a+'-'+b,'session':task,'metric':field,'seed':seed,'difference_pp':v});values.append(v)
                means.append({'contrast':a+'-'+b,'session':task,'metric':field,'mean_difference_pp':float(np.mean(values)) if None not in values else None,
                              'std_ddof1':float(np.std(values,ddof=1)) if None not in values else None,'n':3})
    for a,b in [('E','C'),('D','B'),('E','D')]:
        rows=[r for r in means if r['contrast']==a+'-'+b and r['session']==2];v={r['metric']:r['mean_difference_pp'] for r in rows}
        improved=0;newzero=[]
        for seed in SEEDS:
            x=next(r for r in primary if (r['seed'],r['branch'],r['session'])==(seed,a,2));y=next(r for r in primary if (r['seed'],r['branch'],r['session'])==(seed,b,2))
            improved+=int(x['balanced_accuracy']>y['balanced_accuracy'] and x['current_macro_recall']>y['current_macro_recall'])
            for task in (1,2):
                for p in perclass:
                    if (p['seed'],p['branch'],p['session'],p['head'])==(seed,a,task,'sum') and p['head_index']>=[0,4,6][task] and p['recall']==0:
                        q=next(q for q in perclass if (q['seed'],q['branch'],q['session'],q['head'],q['original_label'])==(seed,b,task,'sum',p['original_label']))
                        if q['recall']>0:newzero.append([seed,task,p['original_label']])
        threshold=v['balanced_accuracy']>0 and v['current_macro_recall']>=10 and v['old_macro_recall']>=-5 and improved>=2 and not newzero
        decisions[a+'-'+b]={'final_mean_deltas':v,'seeds_improved_ba_and_current':improved,'new_zero_current':newzero,'resource_review_threshold_pass':threshold}
    write_csv(out/'val_session_metrics.csv',[flatten(r) for r in metrics]);write_csv(out/'val_per_class_metrics.csv',perclass)
    write_csv(out/'training_components.csv',epochs);write_csv(out/'paired_val_differences.csv',paired);write_csv(out/'paired_val_summary.csv',means)
    write_json(root/'ALL_NEW_CHECKPOINTS_LOCK.json',{'checkpoints':checkpoints,'code_commit':config['code_commit'],'protocol_sha256':config['protocol_sha256'],'test_predictions':0})
    p1=json.loads((root/'p1/COMPLETE.json').read_text());partial=any(r['status']!='COMPLETE' for r in p1['encoders'])
    frozen=list(csv.DictReader((root/'p1/frozen_val_metrics.csv').open()));h1=[]
    for f in frozen:
        if int(f['session'])!=2:continue
        seed=int(f['order_seed'])
        for branch in ('B','C'):
            base=next(r for r in primary if (r['seed'],r['branch'],r['session'])==(seed,branch,2))
            h1.append({'encoder':f['encoder'],'classifier':f['classifier'],'order_seed':seed,'V2_branch':branch,
                       'frozen_final_ba':float(f['balanced_accuracy']),'V2_final_val_ba':base['balanced_accuracy'],
                       'difference_pp':float(f['balanced_accuracy'])-base['balanced_accuracy'],
                       'frozen_result_is_one_deterministic_fit':True})
    write_csv(out/'frozen_vs_v2_val.csv',h1)
    facts={'status':'PARTIAL_BACKBONE_COMPARISON' if partial else 'COMPLETE_P0_P1_P2','p0_checkpoints':18,'p2_trajectories':6,'p2_new_checkpoints':12,
           'p2_epochs':120,'test_predictions':0,'paired_real_streams':'PASS','V2_conclusion_preserved':'COMPLETE_MIXED_SIGNAL','contrasts':decisions,'next_phase':'STOP'}
    write_json(out/'V3_DECISION.json',facts)
    lines=['# P2 all-seen real CE results','','Actual completed matrix: six S0 forks, twelve final incremental checkpoints, 120 new epochs. No S0 retraining, early stopping, checkpoint selection, head-norm or test inference. All comparisons use the identical sorted validation layout.','','| Seed | Branch | Session | BA | Old recall | Current recall | Current→old | Restricted current BA |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in primary:
        if r['session']>0:lines.append(f"| {r['seed']} | {r['branch']} | {r['session']} | {r['balanced_accuracy']:.3f} | {r['old_macro_recall']:.3f} | {r['current_macro_recall']:.3f} | {r['current_to_old_rate']:.3f} | {r['restricted_current_ba']:.3f} |")
    lines+=['','Paired raw differences and mean ± sample SD (ddof=1) are supplied for all three contrasts, stages and metrics. Per-class and lesion-equal recall and old/current tail coverage are in the aggregate tables. Detailed component losses, fixed-train-probe gradients, margins and exposures are in training_components.csv. Numerical labels are not mapped to unverified disease names.']
    (out/'P2_ALL_SEEN_CE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    decision=['# V3 decision','','Status: '+facts['status']+'.','', '## Observed facts','', 'P0 and all six P2 pairs are complete; new model test predictions = 0. V2 remains COMPLETE_MIXED_SIGNAL.','']
    for name,d in decisions.items():
        v=d['final_mean_deltas'];decision.append(f"- {name}: final mean BA delta {v['balanced_accuracy']:+.3f} pp; current recall {v['current_macro_recall']:+.3f} pp; old recall {v['old_macro_recall']:+.3f} pp; paired BA/current improvements {d['seeds_improved_ba_and_current']}/3; predefined resource-review gate {d['resource_review_threshold_pass']}.")
    for encoder,method,branch in sorted({(r['encoder'],r['classifier'],r['V2_branch']) for r in h1}):
        rows=[r for r in h1 if (r['encoder'],r['classifier'],r['V2_branch'])==(encoder,method,branch)]
        decision.append(f"- H1 {encoder}-{method} vs V2-{branch}: frozen BA {rows[0]['frozen_final_ba']:.3f}; paired-layout differences {[round(r['difference_pp'],3) for r in rows]} pp. This repeats one deterministic frozen result against three trained comparators.")
    main=decisions['E-C']['final_mean_deltas']
    interpretation=('Mean current recall recovered but old recall fell by more than 5 pp: observed tradeoff, not a resolved collapse.' if main['current_macro_recall']>0 and main['old_macro_recall']<-5 else
                    'The predefined E-C resource-review criterion is met; this supports considering a separate future request, not automatic expansion.' if decisions['E-C']['resource_review_threshold_pass'] else
                    'The predefined E-C resource-review criterion is not met; the observed evidence does not justify automatic progression.')
    decision+=['','## Mechanistic interpretation','',interpretation,'','The shared-logit-shift gradient test establishes a local directional asymmetry of the objectives. The paired validation intervention measures its end-to-end effect. Current recall recovery with old recall loss is a tradeoff, not a resolution of stability/plasticity. A positive overall score alone does not establish success. The three-seed resource thresholds are not statistical significance tests. Frozen-encoder comparisons address separability under the specified simple classifiers, not a universal representation upper bound.','','## Unverified questions and limits','','No new test inference, external cohort validation or patient-level isolation claim. The small, imbalanced validation set and missing-lesion-ID selection remain limitations. Disease-name mapping and pretraining development exposure remain unverified. Batch-context dependence is recorded without changing routing. DINOv2 unavailability limits the cross-encoder comparison where marked. No claim that encoder differences isolate self-supervision.','','Stop here. P3, Joint Training, more encoders, Full Dynamic and hyperparameter searches have not been launched.']
    (out/'V3_DECISION.md').write_text('\n'.join(decision)+'\n');write_json(root/'FINAL_STATUS.json',facts)
    return facts
