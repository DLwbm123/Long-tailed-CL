"""Prespecified A2 developmental comparisons using the A1 paired analyzer."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from report_nb_rl_a1 import report as analyze
from report_locked_holdout_r1 import csvwrite

METHODS = ('A', 'B', 'C', 'D', 'F_S')
CONTRASTS = (('B','A'), ('C','A'), ('D','A'), ('A','F_S'), ('B','F_S'), ('C','F_S'), ('D','F_S'))


def report(entries, root):
    start = time.monotonic(); root = Path(root); pub = root / 'public'
    tables = analyze(entries, root, methods=METHODS, contrasts=CONTRASTS, analysis_only=True)
    for name, rows in tables.items():
        if rows: csvwrite(pub / (name + '.csv'), rows)
        else: (pub / (name + '.csv')).write_text('method,seed,task,original_label,n_images,recall\n')
    final = {(v['method'],v['seed']): v for v in tables['final_metrics']}
    classes = tables['class_metrics']
    def pair(method, control='A', metric='Final_BA'):
        return next(v for v in tables['paired_differences'] if v['contrast']==method+'-'+control
                    and v['seed']=='fixed_three_mean' and v['metric']==metric)
    decisions = {}
    for method in ('B','C','D'):
        margin = .5 if method == 'D' else 1.
        delta = pair(method); positives = sum(final[method,s]['balanced_accuracy'] > final['A',s]['balanced_accuracy'] for s in (1993,1994,1995))
        zeros = []
        for v in classes:
            if v['method']!=method or v['task']!=4 or v['n_images']<20 or v['recall']!=0: continue
            for control in ('A','F_S'):
                baseline = next(c for c in classes if c['method']==control and c['seed']==v['seed']
                                and c['task']==4 and c['original_label']==v['original_label'])
                if baseline['recall']>0: zeros.append(dict(control=control,seed=v['seed'],original_label=v['original_label'],n=v['n_images']))
        safety = not zeros and pair(method,metric='old')['difference_pp']>=-2 and pair(method,metric='current')['difference_pp']>=-2 \
            and pair(method,metric='tail')['difference_pp']>=0 and all(final[method,s]['balanced_accuracy']-final['A',s]['balanced_accuracy']>=-2 for s in (1993,1994,1995))
        utility = delta['difference_pp']>=margin and positives>=2 and pair(method,metric='AvgBA_inc')['difference_pp']>=0
        decisions[method] = dict(utility='PASS' if utility else 'NOT_PASSED',safety='PASS' if safety else 'FAIL',
            development_gate=bool(utility and safety),required_BA_gain_pp=margin,positive_orders=positives,
            delta_BA_pp=delta['difference_pp'],conditional_interval=[delta['low'],delta['high']],
            frozen_delta_BA_pp=pair(method,'F_S')['difference_pp'],new_qualified_zero_recall=zeros)
    status = dict(engineering_status='PASS',matrix_status='COMPLETE',comparisons=decisions,
                  independent_confirmation=False,validation_role='adaptive development after NB-RL-A1',
                  sampling_added_value='NOT_TESTED',NEXT_DECISION='STOP',publication_status='AUTHORIZED_PENDING_DELIVERY')
    (pub/'NEXT_DECISION.json').write_text(json.dumps(status,indent=2)+'\n')
    (pub/'ANALYSIS_AUDIT.json').write_text(json.dumps(dict(stage_rows=60,class_rows=300,bootstrap_resamples=2000,
        bootstrap_seed=64201,shared_class_component_draws=True,AvgBA_inc_tasks=[2,3,4],no_image_access=True,
        fixed_trained_models=True,CPU_seconds=time.monotonic()-start,comparisons='developmental, no multiplicity adjustment'),indent=2)+'\n')
    lines = ['# NB-RL-A2：保持强度与精确奖励的开发性实验','',
        '四训练条件、三个顺序、四任务；所有 Task1 均为相同 CE 训练。F_S 来自本轮 A 的 Task1。',
        'A：FD10；B：FD30；C：FD10 且后续学习率 1e-4；D：FD10 加 0.1 精确期望辅助奖励（从 Task2 开始）。其余条件学习率 3e-4；每任务 3 epoch。','',
        '|条件|Final BA|AvgBA_inc|Macro-F1|old|current|tail|','|---|---:|---:|---:|---:|---:|---:|']
    for v in tables['dataset_summary']:
        lines.append('|'+v['method']+'|'+'|'.join(f"{v[k]:.3f}" for k in ('Final_BA','AvgBA_inc','MacroF1','old','current','tail'))+'|')
    lines += ['','## 锁定比较','']
    for method,control in CONTRASTS:
        v=pair(method,control)
        lines.append(f"- {method}−{control} Final BA：{v['difference_pp']:+.3f} pp，条件区间 [{v['low']:+.3f}, {v['high']:+.3f}]。")
    for method,v in decisions.items():
        lines.append(f"- {method} 对 A：效用 {v['utility']}；安全 {v['safety']}；正向顺序 {v['positive_orders']}/3。")
    lines += ['','## 解释边界','',
        '本轮由已观察的 A1 验证结果驱动，属于方法开发；三个顺序及共享身份 component bootstrap 不能替代独立确认。',
        '区间采用 2000 次类别/身份 component 配对重采样，seed64201，条件于固定训练模型及三个顺序；未作多重比较校正。',
        'B/C 检验保持强度与学习率；D−A 检验同一强保持基线上的组合奖励增量，不单独验证保持奖励、归一化或策略采样。',
        '所有条件均不采样动作。D 即使通过开发门槛，也不能证明采样有价值，不能自动启动新的采样条件。',
        '主头为 W_final；W_start 仅用于重拟合诊断。所有零召回、逐类分母、梯度/奖励诊断与冻结差距均保留。',
        'B/C 效用阈值为相对 A 最终 BA +1 pp，D 为 +0.5 pp；至少 2/3 顺序正向且 AvgBA_inc 不下降。',
        '安全要求相对 A 的 old/current 不低于 -2 pp、tail 不下降、每顺序 BA 不低于 -2 pp；不得新增相对 A 或 F_S、n>=20 的最终零召回。',
        '结果无论正负均停止本轮并公开交付；不自动扩大 epoch、种子、方法或数据集。']
    (pub/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return status


def synthetic_check(root):
    import tempfile
    from preflight_ct1 import ISIC_ORDERS
    with tempfile.TemporaryDirectory(dir=Path(root)/'private') as directory:
        r=Path(directory);(r/'public').mkdir();(r/'private/sealed').mkdir(parents=True);entries=[]
        for method in METHODS:
            for seed in (1993,1994,1995):
                for task in range(1,5):
                    order=np.array(ISIC_ORDERS[seed][:2*task]);original=np.repeat(sorted(order),3)
                    y=np.array([list(order).index(v) for v in original]);raw=np.eye(len(order))[y]
                    ids=np.array([f'synthetic_{int(c)}_{i}' for c in sorted(order) for i in range(3)])
                    p=r/'private/sealed'/f'{method}_{seed}_{task}.npz'
                    np.savez_compressed(p,raw=raw,start_raw=raw,y=y,order=order,original=original,ids=ids,component=ids)
                    entries.append(dict(method=method,seed=seed,task=task,prediction_file=p.name,prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        result=report(entries,r)
        assert all(v['utility']=='NOT_PASSED' and v['safety']=='PASS' and not v['development_gate'] for v in result['comparisons'].values())
        return dict(status='PASS',synthetic_only=True,stage_rows=60,class_rows=300,equal_methods_do_not_promote=True)
