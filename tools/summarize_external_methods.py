"""Complete native-schedule external methods versus all frozen project controls."""
import json
from pathlib import Path
import sys

from summarize_module_additions import summarize

METHODS=('base','local_transport','residual_drift','local_transport_drift','AIR','APART_full')
FIELDS=('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall',
        'forgetting','final_old_recall','final_new_recall')


def report(snapshot, output):
    p=snapshot['program'];lock=snapshot['lock']
    if p['status']!='COMPLETE' or p['active'] or p['failures'] or len(p['costs'])!=12:
        raise ValueError('Full twelve-process terminal barrier required')
    if any(c['returncode']!=0 or c['gpu'] not in (0,1) for c in p['costs']):
        raise ValueError('Failed/unauthorized cost record')
    for ds in ('ISIC','HK'):
        d=lock['data'][ds]
        for method in ('AIR','APART_full'):
            v=snapshot['runs'][ds+'_'+method];s=v['status'];m=v['metrics']
            steps=0 if method=='AIR' else d['APART_total_updates'];batches=d['AIR_total_analytic_batches'] if method=='AIR' else 0
            if (s['status']!='COMPLETE' or s['steps']!=steps or s['analytic_batches']!=batches or
                    s['stages']!=len(d['task_sizes']) or s['policy_updates'] or m['test_accessed'] or
                    m['evaluation_split']!='development_validation' or m['module']!=method):
                raise ValueError('Method budget/permission differs')
            f=snapshot['preflights'][ds+'_'+method]
            if f['status']!='PASS' or f['optimizer_updates'] or f['policy_updates'] or f['test_accessed']:
                raise ValueError('Missing real zero-update preflight')
            if method=='AIR' and (f['projection_dim']!=8192 or f['max_solve_residual']>1e-8 or
                    v['model_cost']['max_solve_residual']>1e-8):raise ValueError('AIR dense solve audit failed')
            if method=='APART_full' and (not f['parameters_unchanged'] or not f['dual_pool_router_head_gradients'] or
                    f['gradient_norm']<=0 or len(v['scalar_diagnostics'])!=10*len(d['task_sizes'])):
                raise ValueError('APART native compatibility/epoch audit failed')
            for stage in m['stages']:
                seen=d['order'][:sum(d['task_sizes'][:stage['task']])]
                if stage['seen']!=seen or stage['per_class_n']!={str(c):d['val_counts'][str(c)] for c in seen}:
                    raise ValueError('Frozen seen class denominators differ')
    out=Path(output);snapshot.update(modules=list(METHODS),datasets=['ISIC','HK'],
        description='AIR/完整APART：同数据与预训练、作者实际日程；原四格控制全部严格复用，非等更新量比较。')
    summarize(snapshot,out)
    cost=sum(c['residence_seconds'] for c in p['costs']);ledger=snapshot['closed_round_cost_ledger'].copy()
    ledger[snapshot['round_name']]=cost
    audit=dict(status='PASS',source_commit=snapshot['source_commit'],formal_complete=4,
        APART_formal_updates=5360,AIR_analytic_batches=802,AIR_optimizer_updates=0,
        policy_updates=0,test_accessed=False,real_zero_update_preflights=4,successful_cost_records=12,
        failures=[],dispatch_gpus=[0,1],round_gpu_process_residence_seconds=cost,
        cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),
        closed_round_cost_ledger=ledger,parallel_wall_seconds=p['elapsed_seconds'],
        whole_round_wall_seconds=p.get('whole_round_wall_seconds'),resource_wait_seconds=p.get('resource_wait_seconds'),
        cpu_checks=snapshot['cpu_checks'],method_specific_schedules_not_equal_budget=True,
        reused_control_updates_and_costs=0,engineering_events=snapshot.get('engineering_events',[]))
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    (out/'PREFLIGHT_CHECKS.json').write_text(json.dumps(snapshot['preflights'],indent=2)+'\n')
    results=json.loads((out/'RESULTS.json').read_text());results['audit']=audit
    (out/'RESULTS.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    lines=['# AIR/完整APART：ISIC/HK完整原生日程比较','',
        '四格项目控制全部复用公开结果，无重训、重新评价或重复计费。AIR与完整APART使用相同既有预训练/数据/order，但训练日程及optimizer监督曝光不同，不能称等预算比较。test/reserved封存，单seed反复开发集，不是临床验证。','',
        '|数据集|方法|最终BA|平均BA|尾类|遗忘↓|旧类|新类|最终Acc|平均Acc|macro F1|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ('ISIC','HK'):
        for method in METHODS:
            m=snapshot['runs'][ds+'_'+method]['metrics']
            values=[m[k] for k in (*FIELDS,'final_accuracy','average_incremental_accuracy')]
            f1=m.get('final_macro_F1');lines.append('|'+ds+'|'+method+'|'+'|'.join(f'{100*x:.6f}' for x in values)+'|'+('未提供' if f1 is None else f'{100*f1:.6f}')+'|')
    contributions=[]
    lines+=['','## 每个外部方法相对全部四控制的六指标增量（pp）','',
        '|数据集|比较|BA|平均BA|尾类|遗忘|旧类|新类|','|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in ('ISIC','HK'):
        for method in ('AIR','APART_full'):
            a=snapshot['runs'][ds+'_'+method]['metrics']
            for control in METHODS[:4]:
                b=snapshot['runs'][ds+'_'+control]['metrics'];d={k:100*(a[k]-b[k]) for k in FIELDS}
                contributions.append(dict(dataset=ds,method=method,control=control,delta_pp=d))
                lines.append('|'+ds+'|'+method+'−'+control+'|'+'|'.join(f'{d[k]:+.6f}' for k in FIELDS)+'|')
    lines+=['','## 全stage Acc/BA/F1与每类n/recall','',
        '|数据集|方法|task|n|Acc|BA|macro F1|','|---|---|---:|---:|---:|---:|---:|']
    for ds in ('ISIC','HK'):
        for method in METHODS:
            for s in snapshot['runs'][ds+'_'+method]['metrics']['stages']:
                f=s.get('macro_F1');lines.append(f'|{ds}|{method}|{s["task"]}|{s["validation_n"]}|{100*s["accuracy"]:.6f}|{100*s["balanced_accuracy"]:.6f}|'+('未提供' if f is None else f'{100*f:.6f}')+'|')
        lines+=['',f'### {ds}全部最终类','', '|类|n|'+'|'.join(METHODS)+'|','|---|---:|'+'|'.join(['---:']*6)+'|']
        for c,n in lock['data'][ds]['val_counts'].items():
            v=[100*snapshot['runs'][ds+'_'+method]['metrics']['stages'][-1]['per_class_recall'][c] for method in METHODS]
            lines.append('|'+c+'|'+str(n)+'|'+'|'.join(f'{x:.6f}' for x in v)+'|')
    lines+=['',f'本轮12成功GPU子进程驻留{cost:.6f}秒，全部历史闭合{audit["cumulative_gpu_process_residence_seconds"]:.6f}秒；CPU和wall另列，复用控制0费用。',
        'AIR每任务一次train解析fit/gamma2000/8192维；完整APART10epoch/batch48、作者实际首task cosine/后续active恒定学习率及effective WD.01。数学或真实预检PASS不代表效用，小分母与缺失的历史F1完整披露。']
    (out/'CONTRIBUTIONS.json').write_text(json.dumps(contributions,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit


if __name__=='__main__':
    c=json.load(sys.stdin);print(json.dumps(report(json.loads(Path(c['snapshot']).read_text()),c['output']),indent=2))
