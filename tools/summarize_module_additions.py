"""Anonymous aggregate comparisons; no private artifacts are copied."""
import csv
import json
from pathlib import Path
import sys


def summarize(snapshot,destination):
    destination.mkdir(parents=True,exist_ok=False)
    fields=('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall',
            'forgetting','final_old_recall','final_new_recall')
    rows=[]
    for dataset in ('ISIC','HK'):
        base=snapshot['runs'][dataset+'_base']['metrics']
        for arm in ('base','hierarchy','local','paced','fusion'):
            run=snapshot['runs'][dataset+'_'+arm];m=run['metrics']
            if run['status']['status']!='COMPLETE' or m['test_accessed']:
                raise ValueError('Incomplete or invalid aggregate input')
            if [s['per_class_n'] for s in m['stages']] != [s['per_class_n'] for s in base['stages']]:
                raise ValueError('Class denominator mismatch')
            delta={f:100*(m[f]-base[f]) for f in fields}
            passed=(arm!='base' and delta['final_balanced_accuracy']>=1 and
                    delta['final_tail_recall']>=-.5 and delta['forgetting']<=1 and
                    delta['final_new_recall']>=-1 and delta['final_old_recall']>=-1)
            rows.append(dict(dataset=dataset,module=arm,**{f:100*m[f] for f in fields},
                             delta_pp=delta,development_screen_passed=passed))
    (destination/'RESULTS.json').write_text(json.dumps(dict(rows=rows,runs=snapshot['runs'],
        costs=snapshot['program']['costs'],independent_confirmation=False,test_accessed=False),indent=2,allow_nan=False))
    with (destination/'COMPARISONS.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['dataset','module',*fields,'screen_passed'])
        for row in rows:writer.writerow([row['dataset'],row['module'],*[row[f] for f in fields],row['development_screen_passed']])
    lines=['# 单模块增量实验报告','',
           '全部10条固定轨迹完成。结果是单种子、单顺序、已反复查看开发集上的增量比较，不是独立确认；四个模块是论文思想迁移，不是原论文复现。','',
           '|数据集|模块|最终BA %|尾类 %|遗忘 pp|旧类 %|新类 %|ΔBA pp|开发门槛|',
           '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append('|'+ '|'.join([r['dataset'],r['module'],*[f"{r[f]:.4f}" for f in
            ('final_balanced_accuracy','final_tail_recall','forgetting','final_old_recall','final_new_recall')],
            f"{r['delta_pp']['final_balanced_accuracy']:+.4f}",str(r['development_screen_passed'])])+'|')
    seconds=sum(x['residence_seconds'] for x in snapshot['program']['costs'])
    lines+=['',f'本轮所有成功/失败进程驻留合计 {seconds:.3f} 秒，约 {seconds/3600:.4f} GPU小时；是进程驻留而非纯GPU计算。历史成本不重置。',
            '', '层级与局部证据修改原型先验，保持线性推理；步幅控制基于历史参数更新向量秩；融合使用当前fit的batch梯度平方曲率近似。详见冻结协议。',
            '', '只发布协议、源码、匿名类级聚合、模块激活与成本；影像、身份、模型权重、特征、原型向量、私有配置和原始日志不公开。',
            '', '完成本矩阵后停止，不按开发结果自动组合、调参、换种子或继续训练。']
    (destination/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    config=json.load(sys.stdin);summarize(json.loads(Path(config['snapshot']).read_text()),Path(config['output']))
