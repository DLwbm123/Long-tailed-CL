"""Locked-score-only NB-RL-A1 analysis; no training or image access."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ct2d_math import metrics, classify
from report_locked_holdout_r1 import csvwrite, bootstrap_weights, boot_unit
from report_ct6f import recall_groups

METHODS=('S','H','K','G','R','E','F_S','F_R')
SEEDS=(1993,1994,1995)
GROUPS=dict(head=[0,1],mid=[2,3,4,5],tail=[6,7])


def report(entries,root):
    root=Path(root);pub=root/'public';private=root/'private';start=time.monotonic()
    data={};point={};stage=[];classes=[];errors=[];head_diagnostics=[]
    for entry in entries:
        p=private/'sealed'/entry['prediction_file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['prediction_sha256']
        with np.load(p) as f:v={k:f[k].copy() for k in f.files}
        key=entry['method'],entry['seed'],entry['task'];assert key not in data
        assert np.array_equal(v['order'][v['y']],v['original'])
        assert len(set(v['ids']))==len(v['y']) and len(v['order'])==2*key[2]
        data[key]=v;meta=dict(method=key[0],seed=key[1],task=key[2],head='W_final')
        m,pc,er=metrics(v['raw'],v['y'],v['order'],2*(key[2]-1),GROUPS,meta,v['component'])
        m={k:x for k,x in m.items() if not k.startswith('restricted_')}
        stage.append(m);point[key]=m;classes+=pc;errors+=er
        ms,_,_=metrics(v['start_raw'],v['y'],v['order'],2*(key[2]-1),GROUPS,meta,v['component'])
        a=classify(v['raw'],v['order']);b=classify(v['start_raw'],v['order'])
        head_diagnostics.append(dict(method=key[0],seed=key[1],task=key[2],W_start_BA=ms['balanced_accuracy'],
               W_final_BA=m['balanced_accuracy'],refit_BA_change=m['balanced_accuracy']-ms['balanced_accuracy'],
               changed_predictions=int((a!=b).sum()),refit_corrected=int(((b!=v['y'])&(a==v['y'])).sum()),
               refit_new_errors=int(((b==v['y'])&(a!=v['y'])).sum()),head_selection=False))
    assert len(stage)==96 and len(classes)==480 and set(data)=={(m,s,t) for m in METHODS for s in SEEDS for t in range(1,5)}
    canonical=data['R',1993,4];ids={sid:i for i,sid in enumerate(canonical['ids'])}
    weights=bootstrap_weights(canonical,2000,64201,range(8));boot={}
    for key,v in data.items():
        ix=np.array([ids[sid] for sid in v['ids']]);assert np.array_equal(v['original'],canonical['original'][ix])
        assert np.array_equal(v['component'],canonical['component'][ix])
        w=weights[:,ix];rec=boot_unit(v,w,True);boot[key]=recall_groups(rec,2*(key[2]-1),v['order'],GROUPS['tail'])
        pred=classify(v['raw'],v['order']);f1=[]
        for c in range(len(v['order'])):
            tp=w[:,(v['y']==c)&(pred==c)].sum(1);den=w[:,v['y']==c].sum(1)+w[:,pred==c].sum(1)
            f1.append(np.divide(200*tp,den,out=np.zeros(len(w)),where=den!=0))
        boot[key]['MacroF1']=np.mean(f1,axis=0)
    final=[];summary=[];forgetting=[]
    for method in METHODS:
        for seed in SEEDS:
            ms=[point[method,seed,t] for t in range(1,5)]
            row=dict(ms[-1],AvgBA_inc=float(np.mean([v['balanced_accuracy'] for v in ms[1:]])),
                     AvgBA_all=float(np.mean([v['balanced_accuracy'] for v in ms])))
            final.append(row)
            for label in range(8):
                rs=sorted([r for r in classes if r['method']==method and r['seed']==seed and r['original_label']==label],key=lambda x:x['task'])
                eligible=len(rs)>1
                forgetting.append(dict(method=method,seed=seed,original_label=label,first_task=rs[0]['task'],eligible=eligible,
                      first_recall=rs[0]['recall'],final_recall=rs[-1]['recall'],
                      first_minus_final=rs[0]['recall']-rs[-1]['recall'] if eligible else None,
                      peak_minus_final=max(r['recall'] for r in rs)-rs[-1]['recall'] if eligible else None))
        fs=[r for r in final if r['method']==method]
        summary.append(dict(method=method,Final_BA=float(np.mean([v['balanced_accuracy'] for v in fs])),
              AvgBA_inc=float(np.mean([v['AvgBA_inc'] for v in fs])),MacroF1=float(np.mean([v['macro_f1'] for v in fs])),
              old=float(np.mean([v['old_macro_recall'] for v in fs])),current=float(np.mean([v['current_macro_recall'] for v in fs])),
              tail=float(np.mean([v['tail_recall'] for v in fs])),worst_seed_BA=min(v['balanced_accuracy'] for v in fs)))
    cols=dict(Final_BA='balanced_accuracy',MacroF1='macro_f1',old='old_macro_recall',current='current_macro_recall',tail='tail_recall',HM='HM',AvgBA_inc='AvgBA_inc')
    final_by={(v['method'],v['seed']):v for v in final};paired=[]
    for control in ('S','H','K','F_S','F_R','G','E'):
        for metric,col in cols.items():
            diffs=[];samples=[]
            for seed in SEEDS:
                a=final_by['R',seed][col];b=final_by[control,seed][col]
                if metric=='AvgBA_inc':
                    sample=np.mean([boot['R',seed,t]['BA']-boot[control,seed,t]['BA'] for t in (2,3,4)],axis=0)
                else:
                    key='BA' if metric=='Final_BA' else metric
                    sample=boot['R',seed,4][key]-boot[control,seed,4][key]
                difference=a-b;low,high=map(float,np.quantile(sample,[.025,.975]))
                diffs.append(difference);samples.append(sample)
                paired.append(dict(contrast='R-'+control,seed=seed,metric=metric,difference_pp=difference,low=low,high=high))
            low,high=map(float,np.quantile(np.mean(samples,axis=0),[.025,.975]))
            paired.append(dict(contrast='R-'+control,seed='fixed_three_mean',metric=metric,difference_pp=float(np.mean(diffs)),low=low,high=high))
    def pair(control,metric='Final_BA'):
        return next(v for v in paired if v['contrast']=='R-'+control and v['metric']==metric and v['seed']=='fixed_three_mean')
    def diff(control,metric='Final_BA'):return pair(control,metric)['difference_pp']
    positives={c:sum(final_by['R',s]['balanced_accuracy']>final_by[c,s]['balanced_accuracy'] for s in SEEDS) for c in ('S','E')}
    utility=diff('S')>=1 and positives['S']>=2 and diff('S','AvgBA_inc')>=0
    violations=[]
    for control in ('S','F_R'):
        for pc in classes:
            if pc['method']!='R' or pc['task']!=4 or pc['n_images']<20 or pc['recall']!=0:continue
            baseline=next(v for v in classes if v['method']==control and v['task']==4 and v['seed']==pc['seed'] and v['original_label']==pc['original_label'])
            if baseline['recall']>0:violations.append(dict(control=control,seed=pc['seed'],original_label=pc['original_label'],n=pc['n_images']))
    safety=not violations and all(diff(c,'old')>=-2 and diff(c,'current')>=-2 and diff(c,'tail')>=0 and
              all(final_by['R',s]['balanced_accuracy']-final_by[c,s]['balanced_accuracy']>=-2 for s in SEEDS) for c in ('S','F_R'))
    strong=all(diff(c)>=0 for c in ('H','K','F_S')) and diff('F_R')>=1
    retention=diff('G')>=.5 and diff('G','current')>=0 and diff('G','tail')>=0
    sampling=diff('E')>=.5 and positives['E']>=2
    status=dict(engineering_status='PASS',matrix_status='COMPLETE',utility_status='PASS' if utility else
                ('NOT_PASSED_EVIDENCE_INSUFFICIENT' if pair('S')['high']>=1 else 'NOT_PASSED'),
                safety_status='PASS' if safety else 'FAIL',strong_control_net_utility='PASS' if strong else 'FAIL',
                retention_signal='DEVELOPMENTAL_SIGNAL' if retention else 'NOT_ESTABLISHED',
                sampling_added_value=('PRELIMINARY_INTERVAL_CROSSES_ZERO' if pair('E')['low']<=0 else 'DEVELOPMENTAL_SIGNAL') if sampling else 'NOT_ESTABLISHED',
                frozen_comparison_status='PASS' if diff('F_R')>=1 and diff('F_S')>=0 else 'NOT_PASSED',
                complete_method_development_gate=bool(utility and safety and strong),positive_orders=positives,
                new_qualified_zero_recall=violations,zero_recall_safety_scope='Final all-seen W_final',
                NEXT_DECISION='STOP',publication_status='AWAITING_SEPARATE_AUTHORIZATION')
    tables=dict(stage_metrics=stage,class_metrics=classes,final_metrics=final,dataset_summary=summary,
                paired_differences=paired,bootstrap_intervals=paired,head_refit_diagnostics=head_diagnostics,
                error_flows=errors,forgetting=forgetting,zero_recall_classes=[v for v in classes if v['recall']==0])
    for name,rows in tables.items():
        if rows:csvwrite(pub/(name+'.csv'),rows)
    (pub/'NEXT_DECISION.json').write_text(json.dumps(status,indent=2))
    (pub/'ANALYSIS_AUDIT.json').write_text(json.dumps(dict(stage_rows=len(stage),class_rows=len(classes),bootstrap_resamples=2000,
         bootstrap_seed=64201,shared_class_component_draws=True,AvgBA_inc_tasks=[2,3,4],CPU_seconds=time.monotonic()-start,
         no_image_access=True,fixed_trained_models=True,secondary_comparisons='descriptive'),indent=2))
    lines=['# NB-RL-A1：完整四任务辅助策略梯度实验','',
       'ISIC；正常 Task1 起步，无独立基础阶段，无旧图像或旧逐样本特征回放。六训练条件、三顺序、四任务全部完成；两个冻结参照分别来自本轮 S/R 的正常 Task1。',
       '主预测头始终为 W_final，W_start 仅作预先指定的重拟合诊断。未按结果更改主候选 R、epoch 档位或训练目标。','',
       '|方法|Final BA|AvgBA_inc|Macro-F1|old|current|tail|','|---|---:|---:|---:|---:|---:|---:|']
    for v in summary:lines.append('|'+v['method']+'|'+'|'.join(f"{v[k]:.3f}" for k in ('Final_BA','AvgBA_inc','MacroF1','old','current','tail'))+'|')
    lines+=['','## 预定比较与判定','']
    for c in ('S','H','K','F_S','F_R','G','E'):
        v=pair(c);lines.append(f"- R−{c} Final BA：{v['difference_pp']:+.3f} pp，条件区间 [{v['low']:+.3f}, {v['high']:+.3f}]。")
    lines+=['',f"主效用：{status['utility_status']}；安全：{status['safety_status']}；强控制净效用：{status['strong_control_net_utility']}。",
        f"保持组合信号：{status['retention_signal']}；采样附加信号：{status['sampling_added_value']}；冻结参照：{status['frozen_comparison_status']}。",'',
        '## 解释与边界','',
        '这是 RaPO-inspired categorical contextual-bandit auxiliary，不是生成式 RaPO/GRPO 复现。解析引导训练接口是所有条件的共同改变，其收益不能归因于 RL。',
        'E 优化相同的 detached 奖励目标，不做动作抽样。只有 R 相对 E 的预定比较支持，才有采样附加收益的初步证据；E 不劣于 R 时保留“采样无额外证据”的解释。仅有准确率奖励时，期望梯度相当于按当前正确类概率缩放的 CE 梯度，不代表推理能力。',
        'R 与 F_R 的比较区分首任务学习收益和后续持续 RL 的净效用。R−G 是保持奖励组合干预，不单独验证 CTAN。重复的正确类别动作具有相同保持奖励，不能复现不同正确文本轨迹之间的排序。',
        '区间采用 2000 次按类别/身份 component 的共享配对 bootstrap，seed64201，条件于固定训练模型及三个顺序；动作、epoch 和顺序不作为独立患者样本。次比较为描述性，无独立外部确认，保留多重开发限制。',
        '未过门槛不等于证明等效或普遍无效；若主比较区间仍覆盖有意义收益，应解释为本预算证据不足。',
        '逐 epoch 的有效优势组、奖励饱和、梯度范数/夹角、clip、样本暴露和参数更新见 reward_diagnostics.jsonl 与 gradient_diagnostics.jsonl；梯度分项采样点为每 epoch 的首个真实 minibatch。所有零召回与分母均报告，不以受限类别预测替代 CIL 成绩。',
        '资源、数据访问与独立备份以对应审计和最终 ACK 为准。NEXT_DECISION=STOP；不自动延长训练、改变奖励、扩展 HK 或推送 GitHub。']
    reward=pub/'reward_diagnostics.jsonl'
    if reward.exists():
        rs=[json.loads(x) for x in reward.read_text().splitlines()]
        assert len(rs) in (216,360)
        effective=[v['means']['effective_groups'] for v in rs if v['method']=='R']
        lines+=['',f"R 每 epoch 有效优势组比例的均值为 {np.mean(effective):.6f}，最小值为 {min(effective):.6f}。"]
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return status


def synthetic_check(root):
    """End-to-end equal-method fixture: coverage, pairing and no false promotion."""
    import tempfile
    from preflight_ct1 import ISIC_ORDERS
    with tempfile.TemporaryDirectory(dir=Path(root)/'private') as directory:
        r=Path(directory);(r/'public').mkdir();(r/'private/sealed').mkdir(parents=True);entries=[]
        for method in METHODS:
            for seed in SEEDS:
                for task in range(1,5):
                    order=np.array(ISIC_ORDERS[seed][:2*task]);original=np.repeat(sorted(order),3)
                    y=np.array([list(order).index(v) for v in original]);raw=np.eye(len(order))[y]
                    ids=np.array([f'synthetic_{int(c)}_{i}' for c in sorted(order) for i in range(3)])
                    p=r/'private/sealed'/f'{method}_{seed}_{task}.npz'
                    np.savez_compressed(p,raw=raw,start_raw=raw,y=y,order=order,original=original,ids=ids,component=ids)
                    entries.append(dict(method=method,seed=seed,task=task,prediction_file=p.name,prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        result=report(entries,r)
        assert result['utility_status']=='NOT_PASSED' and not result['complete_method_development_gate']
        assert result['safety_status']=='PASS' and result['sampling_added_value']=='NOT_ESTABLISHED'
        return dict(status='PASS',synthetic_only=True,stage_rows=96,class_rows=480,equal_methods_do_not_promote=True)
