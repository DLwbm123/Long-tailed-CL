"""Matched main-grid, AIR gap and every new-class gain/loss from scalar results."""
import csv
import json
from pathlib import Path
from summarize_module_additions import summarize
from summarize_residual_drift import FIELDS,delta,gates
from run_lca_supported_cycle import STEPS,SIZES,ORDERS

MODULES=('base','local_transport','lca_supported','local_transport_lca_supported','AIR')
NEW=MODULES[2:4]


def report(s,out):
    p=s['program'];residual=[]
    if p['status']!='COMPLETE' or p['active'] or p['failures'] or len(p['costs'])!=12:
        raise ValueError('Twelve successful operations and empty active set required')
    if any(c['returncode'] or c['gpu'] not in (0,1) for c in p['costs']):raise ValueError('GPU scope/cost failure')
    for ds in ORDERS:
        for arm in NEW:
            v=s['runs'][ds+'_'+arm];st=v['status'];m=v['metrics'];f=s['preflights'][ds+'_'+arm]
            if st['status']!='COMPLETE' or st['steps']!=sum(STEPS[ds]) or st['policy_updates'] or st['stages']!=len(SIZES[ds]):
                raise ValueError('Exact new trajectory budget differs')
            if m['source_commit']!=s['source_commit'] or m['lca_version']!='supported_old_margin_v1' or m['module']!=arm or m['test_accessed'] or not m['checkpoint_head_rebuild_verified']:
                raise ValueError('Scientific metadata/head rebuild differs')
            if f['status']!='PASS' or f['optimizer_updates'] or f['policy_updates'] or not f['parameters_unchanged'] or not f['gradient_norm']>0 or f['test_accessed']:
                raise ValueError('Real zero-update preflight failed')
            residual.extend([st['max_solve_residual'],f['max_solve_residual']])
            if len(v['scalar_diagnostics'])!=2*len(SIZES[ds]):raise ValueError('Missing full epochs')
            for d in v['scalar_diagnostics']:
                residual.append(d['max_batch_solve_residual'])
                for key in ('lca_supported','boundary_lca_supported'):
                    if key in d:
                        a=d[key]
                        if not a['detached'] or a['old_samples_read'] or a['generated_features'] or a['actual_relative_bound']>.100000001 or a['overlap_leakage']>1e-8:
                            raise ValueError('Risk/support/cap audit differs')
                if 'boundary_solve' in d:residual.append(d['boundary_solve']['relative_residual'])
            for i,t in enumerate(m['stages']):
                seen=ORDERS[ds][:sum(SIZES[ds][:i+1])]
                if t['seen']!=seen or t['per_class_n']!={str(c):s['lock']['data'][ds]['val_counts'][str(c)] for c in seen}:
                    raise ValueError('Label mapping/denominator mismatch')
    if max(residual)>1e-8:raise ValueError('Analytic residual exceeds tolerance')
    s.update(modules=list(MODULES),datasets=['ISIC','HK'],description='LCA-inspired main four-grid, reused base/local_transport/AIR, all signs, no test')
    summarize(s,out);out=Path(out)
    comparisons=[];gaps=[];screens=[];classes=[]
    lines=['# 新版LCA-inspired：主四格与AIR差距','',
        'A0/AL与AIR严格复用已公开结果；AB/AP为新模块单独/组合。不是原版LCA复现。单seed/order反复development val，test/reserved封存。主方法固定local_transport及其组合，不按val选配置。','',
        '|数据集|方法|最终BA|平均BA|尾类|遗忘↓|旧类|新类|最终Acc|平均Acc|AIR−方法 BA gap|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ORDERS:
        m={a:s['runs'][ds+'_'+a]['metrics'] for a in MODULES}
        for a in MODULES:
            x=m[a];gap=100*(m['AIR'][FIELDS[0]]-x[FIELDS[0]])
            lines.append('|'+ds+'|'+a+'|'+'|'.join(f'{100*x[k]:.6f}' for k in (*FIELDS,'final_accuracy','average_incremental_accuracy'))+f'|{gap:+.6f}|')
        for a,b in [(NEW[0],'base'),(NEW[1],'local_transport'),(NEW[1],NEW[0]),(NEW[1],'base'),(NEW[0],'AIR'),(NEW[1],'AIR')]:
            comparisons.append(dict(dataset=ds,contrast=a+'−'+b,delta_pp=delta(m[a],m[b])))
        left=delta(m[NEW[1]],m['local_transport']);right=delta(m[NEW[0]],m['base'])
        comparisons.append(dict(dataset=ds,contrast='interaction',delta_pp={k:left[k]-right[k] for k in FIELDS}))
        for a,b in [(NEW[0],'base'),(NEW[1],'local_transport')]:
            gaps.append(dict(dataset=ds,candidate=a,control=b,AIR_gap_before_pp=100*(m['AIR'][FIELDS[0]]-m[b][FIELDS[0]]),AIR_gap_after_pp=100*(m['AIR'][FIELDS[0]]-m[a][FIELDS[0]]),gap_shrink_pp=100*(m[a][FIELDS[0]]-m[b][FIELDS[0]])))
            screens.append(dict(dataset=ds,candidate=a,base_relative=gates(delta(m[a],m['base'])),own_control_BA_ge_1=delta(m[a],m[b])[FIELDS[0]]>=1,new_mean_protection_ge_minus1=delta(m[a],m[b])[FIELDS[5]]>=-1))
        for i in range(len(SIZES[ds])):
            for a in MODULES:
                t=m[a]['stages'][i]
                for c,n in t['per_class_n'].items():classes.append(dict(dataset=ds,method=a,stage=i+1,original_class=c,train_n=s['lock']['data'][ds]['train_counts'][c],val_n=n,correct_n=round(n*t['per_class_recall'][c]),recall_percent=100*t['per_class_recall'][c]))
    lines+=['','## 全部六指标贡献（pp；遗忘负值有利）','','|数据集|比较|BA|平均BA|tail|forget|old|new|','|---|---|---:|---:|---:|---:|---:|---:|']
    for c in comparisons:lines.append('|'+c['dataset']+'|'+c['contrast']+'|'+'|'.join(f'{c["delta_pp"][k]:+.6f}' for k in FIELDS)+'|')
    for ds in ORDERS:
        lines+=['',f'## {ds}阶段Acc/BA','','|方法|stage|n|Acc|BA|','|---|---:|---:|---:|---:|']
        for a in MODULES:
            for t in s['runs'][ds+'_'+a]['metrics']['stages']:lines.append(f'|{a}|{t["task"]}|{t["validation_n"]}|{100*t["accuracy"]:.6f}|{100*t["balanced_accuracy"]:.6f}|')
        lines+=['',f'## {ds}最终每类与全部新类得失','','|类|train n|val n|base|local_transport|AB|AP|AIR|AB−base|AP−local|AP−AIR|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for c in ORDERS[ds]:
            v={a:100*s['runs'][ds+'_'+a]['metrics']['stages'][-1]['per_class_recall'][str(c)] for a in MODULES}
            lines.append('|'+str(c)+'|'+str(s['lock']['data'][ds]['train_counts'][str(c)])+'|'+str(s['lock']['data'][ds]['val_counts'][str(c)])+'|'+'|'.join(f'{v[a]:.6f}' for a in MODULES)+'|'+f'{v[NEW[0]]-v["base"]:+.6f}|{v[NEW[1]]-v["local_transport"]:+.6f}|{v[NEW[1]]-v["AIR"]:+.6f}|')
    cost=sum(c['residence_seconds'] for c in p['costs']);ledger=s['closed_round_cost_ledger'].copy();ledger[s['round_name']]=cost
    audit=dict(status='PASS',source_commit=s['source_commit'],formal_complete=4,formal_adapter_updates=1288,real_zero_update_preflights=4,
        policy_updates=0,test_accessed=False,successful_cost_records=12,failures=[],dispatch_gpus=[0,1],
        max_solve_residual=max(residual),reused_updates_and_costs_charged=0,round_gpu_process_residence_seconds=cost,
        cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),closed_round_cost_ledger=ledger,
        parallel_wall_seconds=p['elapsed_seconds'],whole_round_wall_seconds=p.get('whole_round_wall_seconds'),cpu_checks=s['cpu_checks'],
        control_matching=s['control_matching'],engineering_events=s['engineering_events'],screens=screens,
        module_target_met=all(all(v['base_relative'].values()) and v['own_control_BA_ge_1'] and v['new_mean_protection_ge_minus1'] for v in screens),
        nonzero_prior_is_not_utility=True,new_recall_guarantee=False)
    for name,value in [('AUDIT.json',audit),('CONTRIBUTIONS.json',comparisons),('AIR_GAPS.json',gaps),('PREFLIGHT_CHECKS.json',s['preflights'])]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    with (out/'STAGE_CLASS_RECALLS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(classes[0]),lineterminator='\n');w.writeheader();w.writerows(classes)
    result=json.loads((out/'RESULTS.json').read_text());result['audit']=audit;(out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines+=['','## 费用与限制','',f'本轮GPU驻留{cost:.6f}秒，复用费用0；CPU准备及全部失败另见AUDIT。',
        'S/M非零只表示激活。旧类漂移是由当前fit与既有mu/Q外推，支持覆盖低时不能称可靠测量。去重同时降低能量；M仅旧类margin，新增项对新列直接力为零不是新类recall保证。',
        '主四格不能分别证明旧类估计、去重和输出保护的因果贡献；本轮不含AC/AO/AD，不增加消融、seed或自动失败重跑。',
        '逐类损伤即使平均门内也公开。ISIC0/3、HK8/3及所有阶段/全部31类支持度完整保留。旧结果未提供F1，不填0，不读旧模型补评价。AIR预训练相同但解析/监督曝光和资源不同，不作等计算预算SOTA或临床结论。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit
