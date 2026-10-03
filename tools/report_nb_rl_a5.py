"""Prespecified A5 reward comparisons using the A1 paired analyzer."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from report_nb_rl_a1 import report as analyze
from report_locked_holdout_r1 import csvwrite

METHODS = ('A', 'G', 'R', 'E', 'F_S')
CONTRASTS = (('R','A'), ('R','G'), ('R','E'), ('R','F_S'), ('G','A'), ('E','A'), ('A','F_S'), ('G','F_S'), ('E','F_S'))


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
    for method in ('G','R','E'):
        margin = 1.
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
            development_gate=bool(utility and safety),
            frozen_net_gain=bool(pair(method,'F_S')['difference_pp']>0 and sum(final[method,s]['balanced_accuracy']>final['F_S',s]['balanced_accuracy'] for s in (1993,1994,1995))>=2),required_BA_gain_pp=margin,positive_orders=positives,
            delta_BA_pp=delta['difference_pp'],conditional_interval=[delta['low'],delta['high']],
            frozen_delta_BA_pp=pair(method,'F_S')['difference_pp'],new_qualified_zero_recall=zeros)
    sampling=pair('R','E'); retention=pair('R','G')
    sampling_pass=bool(sampling['difference_pp']>=.5 and sampling['low']>0 and sum(final['R',s]['balanced_accuracy']>final['E',s]['balanced_accuracy'] for s in (1993,1994,1995))>=2)
    retention_pass=bool(retention['difference_pp']>=.5 and sum(final['R',s]['balanced_accuracy']>final['G',s]['balanced_accuracy'] for s in (1993,1994,1995))>=2 and pair('R','G','current')['difference_pp']>=0 and pair('R','G','tail')['difference_pp']>=0)
    status = dict(engineering_status='PASS',matrix_status='COMPLETE',comparisons=decisions,
                  independent_confirmation=False,validation_role='adaptive development after A1-A3, new training seed74001',
                  primary_comparison='R-A', sampling_added_value='PRELIMINARY_PASS' if sampling_pass else 'NOT_PASSED', retention_added_value='PRELIMINARY_PASS' if retention_pass else 'NOT_PASSED',NEXT_DECISION='STOP',publication_status='AUTHORIZED_PENDING_DELIVERY')
    (pub/'NEXT_DECISION.json').write_text(json.dumps(status,indent=2)+'\n')
    (pub/'ANALYSIS_AUDIT.json').write_text(json.dumps(dict(stage_rows=60,class_rows=300,bootstrap_resamples=2000,
        bootstrap_seed=64201,shared_class_component_draws=True,AvgBA_inc_tasks=[2,3,4],no_image_access=True,
        fixed_trained_models=True,CPU_seconds=time.monotonic()-start,comparisons='developmental, no multiplicity adjustment'),indent=2)+'\n')
    lines = ['# NB-RL-A5：强蒸馏基线上的采样奖励','',
        'A/G/R/E × 三顺序，训练种子74001；共同FD10、lr3e-4、3epoch。Task1相同CE；F_S来自本轮A Task1。',
        'Task2起：A无奖励；G准确率采样奖励；R准确率+0.5保持代理；E同R的精确期望。奖励项系数均0.5。','',
        '|条件|Final BA|AvgBA_inc|Macro-F1|old|current|tail|','|---|---:|---:|---:|---:|---:|---:|']
    for v in tables['dataset_summary']:
        lines.append('|'+v['method']+'|'+'|'.join(f"{v[k]:.3f}" for k in ('Final_BA','AvgBA_inc','MacroF1','old','current','tail'))+'|')
    lines += ['','## 锁定比较','']
    for method,control in CONTRASTS:
        v=pair(method,control)
        lines.append(f"- {method}−{control} Final BA：{v['difference_pp']:+.3f} pp，条件区间 [{v['low']:+.3f}, {v['high']:+.3f}]。")
    for method,v in decisions.items():
        lines.append(f"- {method} 对 A：效用 {v['utility']}；安全 {v['safety']}；正向顺序 {v['positive_orders']}/3。")
    lines += ['',f"采样附加价值：{status['sampling_added_value']}；保持代理附加价值：{status['retention_added_value']}。",'',
        '唯一主要比较为R−A；效用要求BA>=+1pp、2/3顺序正向和AvgBA_inc不下降。安全要求old/current>=-2pp、tail>=0、各顺序BA>=-2pp且无新增n>=20零召回。',
        '采样证据R−E>=.5pp、2/3正向且条件区间下界>0；保持证据R−G>=.5pp、2/3正向且current/tail不下降。',
        '冻结净收益另列于NEXT_DECISION.json，要求候选−F_S均值>0且2/3正向；仅相对A改善不能证明优于冻结表示。',
        '开发性比较：固定训练模型、单一新训练种子及此前使用的三个顺序和验证集；2000次类别/身份component配对bootstrap，seed64201，未校正多重比较。',
        '保持奖励仅是当前图像上的教师一致性代理，不能替代旧类真实表现。分类contextual bandit辅助目标不是生成式长程RL。',
        '主头W_final，W_start仅作头重拟合诊断。所有负结果、分母、逐类零召回、梯度、奖励与冻结差距均保留。',
        '系数0.5为开发后选定单点，跨轮变化不能单独归因于强蒸馏；本轮相同系数的R−E隔离采样。',
        '完成后STOP；不自动扩大种子、系数、数据集或轮次，不能据此宣称达到顶尖论文要求。']
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
        assert result['sampling_added_value']==result['retention_added_value']=='NOT_PASSED'
        assert not any(v['frozen_net_gain'] for v in result['comparisons'].values())
        return dict(status='PASS',synthetic_only=True,stage_rows=60,class_rows=300,equal_methods_do_not_promote=True)
