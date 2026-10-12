"""All eight methods, all positive/negative comparisons and reconstruction limits."""
import csv
import json
from pathlib import Path
from summarize_module_additions import summarize
from summarize_external_methods import FIELDS

METHODS=('base','local_transport','residual_drift','local_transport_drift','AIR','APART_full','DAT_paper','Hao_paper')
NEW=METHODS[-2:]
PROJECT=METHODS[:4]
EXTERNAL=METHODS[4:]


def unified_comparison(snapshot, out, lines):
    spec=snapshot['resource_comparison'];runs=snapshot['runs'];lock=snapshot['lock']
    if spec['primary_project_method']!='local_transport' or spec['posthoc_best_variant_selection']:
        raise ValueError('No posthoc project row selection')
    resource_rows=[]
    for row in spec['resource_rows']:
        row=row.copy();name=row['dataset']+'_'+row['method'];v=runs[name];model=v.get('model_cost') or {}
        costs=snapshot['cost_sources'].get(row['method'],[])
        for operation,field in [('train','training_GPU_residence_seconds'),('evaluate','evaluation_GPU_residence_seconds')]:
            matched=[c for c in costs if c.get('job')==name and c.get('operation')==operation]
            row[field]=sum(c['residence_seconds'] for c in matched) if matched else None
        matched=[c for c in costs if c.get('job')==name+'_preflight']
        row['matched_preflight_GPU_residence_seconds']=sum(c['residence_seconds'] for c in matched) if matched else None
        row['optimizer_trainable_parameters']=model.get('trainable_parameters',model.get('trainable_parameters_peak'))
        row['checkpoint_bytes']=model.get('checkpoint_bytes');row['peak_GPU_bytes']=model.get('peak_gpu_bytes')
        exposures=row['exact_optimizer_image_exposures']
        row['old_replay_optimizer_image_exposures']=(None if exposures is None else
            exposures-row['epochs_per_task']*sum(lock['data'][row['dataset']]['train_counts'].values())
            if row['old_image_replay'] else 0)
        if row['old_replay_optimizer_image_exposures'] is not None and row['old_replay_optimizer_image_exposures']<0:
            raise ValueError('Replay exposure is inconsistent with current training support')
        resource_rows.append(row)
    (out/'RESOURCE_COMPARISON.json').write_text(json.dumps(dict(rows=resource_rows,notes=spec['notes'],reused_costs_charged_now=0),indent=2,ensure_ascii=False)+'\n')
    with (out/'RESOURCE_COMPARISON.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(resource_rows[0]));writer.writeheader();writer.writerows(resource_rows)
    lines+=['','## 单列资源区别（不等预训练、监督曝光或计算预算）','',
       '|数据集|方法|预训练|epoch/遍历|batch|optimizer更新|解析批|更新/我方|回放不同图bank|可训练参数|checkpoint GiB|GPU峰值GiB|train驻留秒|eval驻留秒|',
       '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def value(x,precision=6):return '未提供' if x is None else f'{x:.{precision}f}'
    for r in resource_rows:
        lines.append('|'+r['dataset']+'|'+r['method']+'|'+r['pretraining']+'|'+str(r['epochs_per_task'] if r['epochs_per_task'] is not None else '1遍fit')+'|'+str(r['batch_size'])+'|'+str(r['optimizer_steps'])+'|'+('不适用' if not r['analytic_batches_applicable'] else str(r['analytic_batches']))+'|'+value(r['optimizer_step_ratio_to_local_transport'],3)+'|'+str(r['old_image_bank_unique'])+'|'+('未提供' if r['optimizer_trainable_parameters'] is None else str(r['optimizer_trainable_parameters']))+'|'+value(None if r['checkpoint_bytes'] is None else r['checkpoint_bytes']/1024**3,3)+'|'+value(None if r['peak_GPU_bytes'] is None else r['peak_GPU_bytes']/1024**3,3)+'|'+value(r['training_GPU_residence_seconds'],3)+'|'+value(r['evaluation_GPU_residence_seconds'],3)+'|')
    lines+=['','我方四格使用原固定80/20 fit/meta，optimizer只fit、边界全current train；AIR全train单遍、APART全train10epoch、DAT全train40epoch另有平衡重采样、Hao全train100epoch另有旧图回放。精确公开的图像曝光与优化器日程见资源CSV；原fit影像精确数量未提供，不按名义比例捏造。AIR0optimizer不等于0计算或0存储。',
       '旧类存储：我方聚合充分统计/原型（局部或漂移配置另外留聚合量）、AIR逐类A/C/count、APART双池/router/head、DAT缓存hyper-adapter/heads/keys、Hao旧训练图像引用+旧模型。旧未提供的参数/峰值留缺失；不能仅凭step比例称计算成本按同比例减少。旧共用/不明确匹配的预检不乱分摊，不重复收取旧成本。']
    contrasts=[]
    lines+=['','## 当前主方法local_transport相对全部四外部（pp）','',
       '|数据集|我方−外部|BA|平均BA|尾类|遗忘↓|旧类|新类|','|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in spec['datasets']:
        project=runs[ds+'_local_transport']['metrics']
        for external in EXTERNAL:
            other=runs[ds+'_'+external]['metrics'];delta={k:100*(project[k]-other[k]) for k in FIELDS}
            better=[k for k,d in delta.items() if (-d if k=='forgetting' else d)>1e-9]
            worse=[k for k,d in delta.items() if (-d if k=='forgetting' else d)<-1e-9]
            contrasts.append(dict(dataset=ds,project='local_transport',external=external,delta_pp=delta,favourable_metrics=better,unfavourable_metrics=worse))
            lines.append('|'+ds+'|local_transport−'+external+'|'+'|'.join(f'{delta[k]:+.6f}' for k in FIELDS)+'|')
    verdict={}
    for ds in spec['datasets']:
        group=[r for r in contrasts if r['dataset']==ds];wins=sum(r['delta_pp']['final_balanced_accuracy']>1e-9 for r in group);losses=sum(r['delta_pp']['final_balanced_accuracy']<-1e-9 for r in group)
        verdict[ds]=dict(final_BA_higher_external_count=wins,final_BA_lower_external_count=losses,final_BA_ties=4-wins-losses,statistical_superiority_claimed=False)
        lines+=['',f'{ds}：local_transport最终BA高于{wins}/4个外部方法、低于{losses}/4、持平{4-wins-losses}/4。逐项有利和不利指标见上表及COMPETITIVENESS.json，不能由最终BA排名抹掉tail、forget、old/new取舍，也不能在此后挑另一配置替代主方法。']
    (out/'COMPETITIVENESS.json').write_text(json.dumps(dict(primary_method='local_transport',all_project_variants_retained=True,primary_contrasts=contrasts,descriptive_verdict=verdict,notes=spec['notes']),indent=2,ensure_ascii=False)+'\n')
    with (out/'STAGE_CLASS_RECALLS.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['dataset','method','stage','class','train_n','val_n','correct_n','recall_percent'])
        for ds in spec['datasets']:
            for method in METHODS:
                for t in runs[ds+'_'+method]['metrics']['stages']:
                    for c,n in t['per_class_n'].items():
                        recall=t['per_class_recall'][c];w.writerow([ds,method,t['task'],c,lock['data'][ds]['train_counts'][c],n,round(n*recall),100*recall])
    with (out/'FINAL_CLASS_DIFFERENCES.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['dataset','external','project','class','val_n','external_recall_percent','project_recall_percent','external_minus_project_pp','net_correct_n_difference'])
        for ds in spec['datasets']:
            for external in EXTERNAL:
                a=runs[ds+'_'+external]['metrics']['stages'][-1]
                for project in PROJECT:
                    b=runs[ds+'_'+project]['metrics']['stages'][-1]
                    for c,n in a['per_class_n'].items():
                        x=a['per_class_recall'][c];y=b['per_class_recall'][c]
                        w.writerow([ds,external,project,c,n,100*x,100*y,100*(x-y),round(n*x)-round(n*y)])
    lines+=['','最终31类、四外部×四我方全部496条类级差异写入FINAL_CLASS_DIFFERENCES.csv；所有已见stage类的n/recall写入STAGE_CLASS_RECALLS.csv。正、零、负全部保留。净正确数差异不等于逐样本rescue/harm，未获取逐样本分数或额外推理。',
       '竞争力结论只限冻结医学迁移配置的单seed/order开发集。DAT/Hao为非官方重实现，完整APART保留作者实际日程/路由/辅助损失，AIR采用共享预训练迁移；预训练/回放/训练量不同。没有统计显著性、独立患者、临床或封存test证据。']




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
    lines+=['','## 四个外部相对全部四个我方配置，以及新外部相对AIR/APART（pp）','', '|数据集|比较|BA|平均BA|尾类|遗忘|旧类|新类|','|---|---|---:|---:|---:|---:|---:|---:|'];contributions=[]
    for ds in ('ISIC','HK'):
        for method in EXTERNAL:
            m=s['runs'][ds+'_'+method]['metrics']
            controls=PROJECT+METHODS[4:6] if method in NEW else PROJECT
            for control in controls:
                b=s['runs'][ds+'_'+control]['metrics'];delta={k:100*(m[k]-b[k]) for k in FIELDS}
                contributions.append(dict(dataset=ds,method=method,control=control,delta_pp=delta))
                lines.append('|'+ds+'|'+method+'−'+control+'|'+'|'.join(f'{delta[k]:+.6f}' for k in FIELDS)+'|')
    lines+=['','## 所有阶段与每类支持']
    for ds in ('ISIC','HK'):
        lines+=['',f'### {ds}各阶段','', '|数据集|方法|stage|n|Acc|BA|macro F1|','|---|---|---:|---:|---:|---:|---:|']
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
    if len(contributions)!=40:raise ValueError('All forty signed contrasts required')
    unified_comparison(s,out,lines)
    a['reporting_comparison_version']=s['resource_comparison']['version']
    a['complete_signed_comparisons']=40
    a['primary_project_method']='local_transport'
    a['reporting_cpu_checks']=s.get('reporting_cpu_checks')
    a['reporting_source_commit']=s.get('reporting_source_commit')
    (out/'AUDIT.json').write_text(json.dumps(a,indent=2)+'\n')
    results['audit']=a
    (out/'RESULTS.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    (out/'CONTRIBUTIONS.json').write_text(json.dumps(contributions,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return a
