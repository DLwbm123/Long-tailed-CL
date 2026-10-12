"""All eight methods, all positive/negative comparisons and reconstruction limits."""
import json
from pathlib import Path
from summarize_module_additions import summarize
from summarize_external_methods import FIELDS

METHODS=('base','local_transport','residual_drift','local_transport_drift','AIR','APART_full','DAT_paper','Hao_paper')
NEW=METHODS[-2:]


def report(s,out):
    p=s['program'];lock=s['lock']
    if p['status']!='COMPLETE' or p['active'] or p['failures'] or len(p['costs'])!=12:raise ValueError('Twelve terminal GPU operations')
    if any(c['returncode'] or c['gpu'] not in (0,1) for c in p['costs']):raise ValueError('Unsuccessful/unauthorized cost')
    for ds in ('ISIC','HK'):
        d=lock['data'][ds]
        for method in NEW:
            name=ds+'_'+method;v=s['runs'][name];m=v['metrics'];status=v['status'];f=s['preflights'][name]
            if status['status']!='COMPLETE' or status['steps']!=d[method]['total_updates'] or status['stages']!=len(d['task_sizes']) or status['policy_updates'] or m['test_accessed']:
                raise ValueError('Formal support/budget differs')
            if not m['nonofficial_paper_reimplementation'] or m['module']!=method or m['evaluation_split']!='development_validation':raise ValueError('Method identity')
            if f['status']!='PASS' or f['optimizer_updates'] or f['test_accessed'] or not f['parameters_unchanged'] or f['gradient_norm']<=0:raise ValueError('Real preflight')
            if len(v['scalar_diagnostics'])!=lock[method]['epochs']*len(d['task_sizes']):raise ValueError('All frozen epochs required')
            for t in m['stages']:
                seen=d['order'][:sum(d['task_sizes'][:t['task']])]
                if t['seen']!=seen or t['per_class_n']!={str(c):d['val_counts'][str(c)] for c in seen}:raise ValueError('Class denominator differs')
    out=Path(out);s.update(modules=list(METHODS),datasets=['ISIC','HK'],description='DAT/Hao非官方论文重实现；所有旧控制与AIR/APART完整复用，回放/预训练/预算差异单列。')
    summarize(s,out)
    cost=sum(c['residence_seconds'] for c in p['costs']);ledger=s['closed_round_cost_ledger'].copy();ledger[s['round_name']]=cost
    a=dict(status='PASS',source_commit=s['source_commit'],formal_complete=4,formal_updates=lock['budgets']['total_optimizer_updates'],
        real_zero_update_preflights=4,policy_updates=0,test_accessed=False,successful_cost_records=12,failures=[],dispatch_gpus=[0,1],
        round_gpu_process_residence_seconds=cost,cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),
        closed_round_cost_ledger=ledger,parallel_wall_seconds=p['elapsed_seconds'],whole_round_wall_seconds=p.get('whole_round_wall_seconds'),
        resource_wait_seconds=p.get('resource_wait_seconds'),cpu_checks=s['cpu_checks'],nonofficial_paper_reimplementation=True,
        reused_control_updates_and_costs=0,engineering_events=s.get('engineering_events',[]))
    (out/'AUDIT.json').write_text(json.dumps(a,indent=2)+'\n');(out/'PREFLIGHT_CHECKS.json').write_text(json.dumps(s['preflights'],indent=2)+'\n')
    results=json.loads((out/'RESULTS.json').read_text());results['audit']=a
    (out/'RESULTS.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    lines=['# DAT/Hao：ISIC/HK非官方论文重实现完整对比','',
      '按用户明确授权补全未披露细节，全部事先冻结；不是作者代码、不是原论文benchmark精确复现。Hao允许旧图像herding回放，DAT无旧图像回放；各方法预训练、更新、监督曝光不等，不能作等预算因果归因。test/reserved封存，单seed/order反复开发集仅开发证据。','',
      '|数据集|方法|最终BA|平均BA|尾类|遗忘↓|旧类|新类|最终Acc|平均Acc|macro F1|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ('ISIC','HK'):
        for method in METHODS:
            m=s['runs'][ds+'_'+method]['metrics'];values=[m[k] for k in (*FIELDS,'final_accuracy','average_incremental_accuracy')]
            f=m.get('final_macro_F1');lines.append('|'+ds+'|'+method+'|'+'|'.join(f'{100*v:.6f}' for v in values)+'|'+('未提供' if f is None else f'{100*f:.6f}')+'|')
    lines+=['','## 全部相对六个既有控制增量（pp）','', '|数据集|比较|BA|平均BA|尾类|遗忘|旧类|新类|','|---|---|---:|---:|---:|---:|---:|---:|'];contributions=[]
    for ds in ('ISIC','HK'):
        for method in NEW:
            m=s['runs'][ds+'_'+method]['metrics']
            for control in METHODS[:-2]:
                b=s['runs'][ds+'_'+control]['metrics'];delta={k:100*(m[k]-b[k]) for k in FIELDS}
                contributions.append(dict(dataset=ds,method=method,control=control,delta_pp=delta))
                lines.append('|'+ds+'|'+method+'−'+control+'|'+'|'.join(f'{delta[k]:+.6f}' for k in FIELDS)+'|')
    lines+=['','## 所有阶段与每类支持','', '|数据集|方法|stage|n|Acc|BA|macro F1|','|---|---|---:|---:|---:|---:|---:|']
    for ds in ('ISIC','HK'):
        for method in METHODS:
            for t in s['runs'][ds+'_'+method]['metrics']['stages']:
                f=t.get('macro_F1');lines.append(f'|{ds}|{method}|{t["task"]}|{t["validation_n"]}|{100*t["accuracy"]:.6f}|{100*t["balanced_accuracy"]:.6f}|'+('未提供' if f is None else f'{100*f:.6f}')+'|')
        lines+=['',f'### {ds}最终全部类召回','', '|类|train n|val n|'+'|'.join(METHODS)+'|','|---|---:|---:|'+'|'.join(['---:']*len(METHODS))+'|']
        for c in lock['data'][ds]['order']:
            values=[100*s['runs'][ds+'_'+method]['metrics']['stages'][-1]['per_class_recall'][str(c)] for method in METHODS]
            lines.append('|'+str(c)+'|'+str(lock['data'][ds]['train_counts'][str(c)])+'|'+str(lock['data'][ds]['val_counts'][str(c)])+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
    lines+=['','## 资源与边界','',f'本轮GPU12次驻留{cost:.6f}秒，历史独立round累加{a["cumulative_gpu_process_residence_seconds"]:.6f}秒；复用控制费用0，CPU/wall独立列出。',
       'DAT每task40epoch/.03 Adam/batch24、常数lr，当前任务平衡支路重采样；Hao100epoch/.1 SGD/60,80衰减/batch32，实际bank ISIC160/HK413张不同旧训练图。HK原类5/6/19/21不足20张分别3/6/6/18，不补造不同图像。',
       '所有缺失细节和预训练差异见PROTOCOL_LOCK，数学、兼容或预检PASS仅表示执行有效，不能称方法有效；所有正负及小分母保留。']
    (out/'CONTRIBUTIONS.json').write_text(json.dumps(contributions,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return a
