"""Paired training-seed replication; identity bootstrap stays conditional on fitted models."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from report_nb_rl_a1 import report as analyze
from report_locked_holdout_r1 import csvwrite

METHODS = ('A', 'C', 'F_S')
ORDERS = (1993, 1994, 1995)
RUNS = {o*100+i: dict(order_seed=o, training_seed=73000+i) for o in ORDERS for i in (1,2)}
CONTRASTS = (('C','A'), ('C','F_S'), ('A','F_S'))


def report(entries, root):
    pub = Path(root)/'public'
    tables = analyze(entries, root, methods=METHODS, contrasts=CONTRASTS,
                     analysis_only=True, seeds=tuple(RUNS), aggregate_label='fixed_six_mean')
    final = {(r['method'],r['seed']):r for r in tables['final_metrics']}
    paired = tables['paired_differences']
    per_order = []
    for candidate,control in CONTRASTS:
        for order in ORDERS:
            for metric in ('Final_BA','AvgBA_inc','old','current','tail','MacroF1','HM'):
                rows = [r for r in paired if r['contrast']==candidate+'-'+control
                        and r['metric']==metric and r['seed'] in (order*100+1,order*100+2)]
                values = [r['difference_pp'] for r in rows]
                per_order.append(dict(contrast=candidate+'-'+control,order_seed=order,metric=metric,
                    training_seed_73001_pp=values[0],training_seed_73002_pp=values[1],
                    difference_pp=float(np.mean(values)),minimum_pp=min(values),maximum_pp=max(values),
                    positive_training_seeds=sum(v>0 for v in values)))
    tables['per_order_differences'] = per_order
    for rows in tables.values():
        for row in rows:
            if row.get('seed') in RUNS:
                row.update(run_id=row['seed'],**RUNS[row['seed']])
    def pair(control,metric='Final_BA'):
        return next(r for r in paired if r['contrast']=='C-'+control and r['metric']==metric and r['seed']=='fixed_six_mean')
    positive_orders = sum(r['difference_pp']>0 for r in per_order if r['contrast']=='C-A' and r['metric']=='Final_BA')
    positive_pairs = sum(final['C',s]['balanced_accuracy']>final['A',s]['balanced_accuracy'] for s in RUNS)
    zeros = []
    for r in tables['class_metrics']:
        if r['method']!='C' or r['task']!=4 or r['n_images']<20 or r['recall']!=0: continue
        for control in ('A','F_S'):
            b=next(v for v in tables['class_metrics'] if v['method']==control and v['seed']==r['seed']
                   and v['task']==4 and v['original_label']==r['original_label'])
            if b['recall']>0: zeros.append(dict(control=control,run_id=r['seed'],original_label=r['original_label'],n=r['n_images']))
    utility = pair('A')['difference_pp']>=1 and positive_orders>=2 and pair('A','AvgBA_inc')['difference_pp']>=0
    safety = not zeros and pair('A','old')['difference_pp']>=-2 and pair('A','current')['difference_pp']>=-2 \
             and pair('A','tail')['difference_pp']>=0 and all(final['C',s]['balanced_accuracy']-final['A',s]['balanced_accuracy']>=-2 for s in RUNS)
    status=dict(engineering_status='PASS',matrix_status='COMPLETE',utility='PASS' if utility else 'NOT_PASSED',
        safety='PASS' if safety else 'FAIL',development_gate=bool(utility and safety),positive_orders=positive_orders,
        positive_training_pairs=positive_pairs,training_pairs=6,new_qualified_zero_recall=zeros,
        frozen_BA_delta_pp=pair('F_S')['difference_pp'],frozen_conditional_interval=[pair('F_S')['low'],pair('F_S')['high']],
        independent_confirmation=False,validation_role='training-randomness replication on previously used validation data',
        NEXT_DECISION='STOP',publication_status='AUTHORIZED_PENDING_DELIVERY')
    for name,rows in tables.items():
        if rows: csvwrite(pub/(name+'.csv'),rows)
        else: (pub/(name+'.csv')).write_text('method,seed,task,original_label,n_images,recall\n')
    (pub/'NEXT_DECISION.json').write_text(json.dumps(status,indent=2)+'\n')
    (pub/'ANALYSIS_AUDIT.json').write_text(json.dumps(dict(stage_rows=72,class_rows=360,training_pairs=6,
        orders=ORDERS,training_seeds=[73001,73002],bootstrap_resamples=2000,bootstrap_seed=64201,
        shared_identity_component_draws=True,fixed_trained_models=True,no_image_access=True,
        no_independent_patient_replication_from_seeds=True,independent_confirmation=False),indent=2)+'\n')
    lines=['# NB-RL-A3：学习率收益的训练随机性复验','',
        '三个固定类别顺序 × 两个新训练种子；A/C 各 6 条完整轨迹，F_S 来自对应 A 的 Task1。每任务 3 epoch。',
        'A/C 均 FD10；Task2–4 学习率分别为 3e-4 / 1e-4；Task1 统一 CE、3e-4。不启用辅助奖励。','',
        '|条件|Final BA|AvgBA_inc|Macro-F1|old|current|tail|','|---|---:|---:|---:|---:|---:|---:|']
    for r in tables['dataset_summary']:
        lines.append('|'+r['method']+'|'+'|'.join(f"{r[k]:.3f}" for k in ('Final_BA','AvgBA_inc','MacroF1','old','current','tail'))+'|')
    lines+=['','## 固定比较','']
    for control in ('A','F_S'):
        r=pair(control)
        lines.append(f"- C−{control} Final BA：{r['difference_pp']:+.3f} pp，固定模型条件区间 [{r['low']:+.3f}, {r['high']:+.3f}]。")
    lines += [f"- C−A：效用 {status['utility']}；安全 {status['safety']}；正向顺序 {positive_orders}/3，正向训练配对 {positive_pairs}/6。",'',
        '|顺序|训练种子 73001 的 C−A BA|训练种子 73002 的 C−A BA|两种子均值|','|---|---:|---:|---:|']
    for r in per_order:
        if r['contrast']=='C-A' and r['metric']=='Final_BA':
            lines.append(f"|{r['order_seed']}|{r['training_seed_73001_pp']:+.3f}|{r['training_seed_73002_pp']:+.3f}|{r['difference_pp']:+.3f}|")
    lines += ['','## 解释边界','',
        '沿用开发效用门槛：C−A Final BA 至少 +1 pp，至少 2/3 顺序的两训练种子均值正向，AvgBA_inc 不下降。',
        '安全：相对 A 的平均 old/current 至少 -2 pp、tail 不下降，每个训练配对的 Final BA 至少 -2 pp；不得新增相对 A 或 F_S、n>=20 的最终零召回。',
        '聚合安全通过不等于每个顺序的每个类群均无退化；逐顺序、逐训练种子的 old/current/tail 差距完整保留。冻结参照差距单独报告。',
        '2000 次类别/身份 component 配对 bootstrap 使用共享抽样权重；区间条件于六组已训练模型，不估计训练随机性总体分布。两个训练种子的差异以原值和范围报告。',
        '验证集已参与 A1/A2 开发；新训练种子不能替代独立数据确认，六组模型不能当作六份独立患者样本。未作多重比较校正。',
        '主头始终 W_final；W_start 仅作诊断。无旧图像回放、无 test/reserved 访问。所有结果保留，NEXT_DECISION=STOP，不自动增加条件、epoch 或种子。']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return status


def synthetic_check(root):
    from preflight_ct1 import ISIC_ORDERS
    with tempfile.TemporaryDirectory(dir=Path(root)/'private') as directory:
        r=Path(directory);(r/'public').mkdir();(r/'private/sealed').mkdir(parents=True);entries=[]
        for method in METHODS:
            for seed,spec in RUNS.items():
                for task in range(1,5):
                    order=np.array(ISIC_ORDERS[spec['order_seed']][:2*task]);original=np.repeat(sorted(order),3)
                    y=np.array([list(order).index(v) for v in original]);raw=np.eye(len(order))[y]
                    ids=np.array([f'synthetic_{int(c)}_{i}' for c in sorted(order) for i in range(3)])
                    p=r/'private/sealed'/f'{method}_{seed}_{task}.npz'
                    np.savez_compressed(p,raw=raw,start_raw=raw,y=y,order=order,original=original,ids=ids,component=ids)
                    entries.append(dict(method=method,seed=seed,task=task,prediction_file=p.name,prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        result=report(entries,r)
        assert result['utility']=='NOT_PASSED' and result['safety']=='PASS' and not result['development_gate']
        assert result['positive_orders']==result['positive_training_pairs']==0
        return dict(status='PASS',stage_rows=72,class_rows=360,equal_methods_do_not_promote=True,paired_seeds=6)
