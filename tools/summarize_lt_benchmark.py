"""Publishable aggregates for the frozen matrix; no sample IDs or private paths."""
import json
from pathlib import Path
import sys

from run_prototype_single import save


def summarize(root, jobs):
    root = Path(root); results = {}; comparisons = []
    lines = ['# 三数据集 ordered / shuffled：原始完整方案验证','',
        '同一设置内使用相同划分、初始化、训练步数和两轮适配器训练；每臂只运行 seed74002。',
        '完整方案 cf_group；cf_linear 为闭式头基线，cf_prototype 增加多原型证据，cf_weighted 使用直接梯度权重控制。',
        '医学数据使用原有开发验证集；CIFAR 使用冻结模型的官方测试集。无验证集选模型、选参数或提前停止。',
        '单种子、短训练预算只能支持本轮设置内的比较，不能作为充分收敛、独立临床确认或论文表格复现的结论。','',
        '|数据/顺序/方法|平均 Acc %|最终 Acc %|最终 BA %|任务遗忘 pp|策略更新|非均匀权重轮数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for spec in jobs:
        if spec['id']=='PREFLIGHT': continue
        name=spec['id']; path=root/spec.get('result_id',name)
        status=json.loads((path/'STATUS.json').read_text())
        if status['status']!='COMPLETE': raise ValueError('Incomplete trajectory')
        metrics=json.loads((path/'metrics.json').read_text())
        diagnostics=json.loads((path/'diagnostics.json').read_text())
        activation=dict(policy_updates=sum(d['controller']['optimizer_steps'] for d in diagnostics),
            weighting_epochs=sum(d['activation']['weight_l1_change']>1e-10 for d in diagnostics),
            accepted_epochs=sum(d['controller']['accepted'] for d in diagnostics),epochs=len(diagnostics),
            prototype_active_epochs=sum(d['activation']['prototype_metric_relative_change']>1e-10 for d in diagnostics))
        results[name]=dict(status=status,metrics=metrics,activation=activation,diagnostics=diagnostics)
        m=metrics;a=activation
        lines.append(f"|{name}|{100*m['average_incremental_accuracy']:.3f}|{100*m['final_accuracy']:.3f}|{100*m['final_balanced_accuracy']:.3f}|{100*m['task_forgetting']:.3f}|{a['policy_updates']}|{a['weighting_epochs']}/{a['epochs']}|")
    for dataset in ('ISIC','HK','CIFAR100LT'):
        for order in ('ordered','shuffled'):
            full=results[f'{dataset}_{order}_cf_group']
            for control in ('cf_linear','cf_prototype','cf_weighted'):
                other=results[f'{dataset}_{order}_{control}']
                if full['status']['steps']!=other['status']['steps']: raise ValueError('Unmatched adapter updates')
                comparisons.append(dict(dataset=dataset,order=order,control=control,
                    final_accuracy_delta=full['metrics']['final_accuracy']-other['metrics']['final_accuracy'],
                    final_ba_delta=full['metrics']['final_balanced_accuracy']-other['metrics']['final_balanced_accuracy'],
                    task_forgetting_delta=full['metrics']['task_forgetting']-other['metrics']['task_forgetting']))
    lines += ['', '非均匀权重轮数为零时，不能宣称 RL 权重对训练实际生效；策略有更新与最终动作被接受分开统计。',
        '原型证据折叠为线性头中的先验，并未增加非线性分类边界；旧类共同平移仍是近似。',
        'many >100、medium 20–100、few <20 按冻结训练样本数定义，空组记 null。每类召回及任务准确率矩阵见 RESULTS.json。','']
    save(root/'public/RESULTS.json',dict(records=results,paired_comparisons=comparisons))
    (root/'public/REPORT_ZH.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    c=json.load(sys.stdin);summarize(c['root'],c['jobs'])
