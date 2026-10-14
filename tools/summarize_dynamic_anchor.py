"""All signs of main grid, matched static controls, class damage and AIR gaps."""
import csv
import json
from pathlib import Path

from summarize_module_additions import summarize
from summarize_residual_drift import FIELDS,delta,gates
from run_dynamic_anchor_cycle import ORDERS,SIZES,STEPS,label

NEW=('dynamic_anchor','local_transport_dynamic_anchor','static_anchor','local_transport_static_anchor')
MODULES=('base','local_transport',*NEW,'AIR')


def report(s,out):
    p=s['program'];residual=[]
    if p['status']!='COMPLETE' or p['active'] or p['failures'] or len(p['costs'])!=24:raise ValueError('24 successful operations and empty active required')
    if any(c['returncode'] or c['gpu'] not in (0,1) for c in p['costs']):raise ValueError('GPU cost/scope failure')
    for ds in ORDERS:
        for arm in NEW:
            v=s['runs'][ds+'_'+arm];st=v['status'];m=v['metrics'];f=s['preflights'][ds+'_'+arm]
            if st['status']!='COMPLETE' or st['steps']!=sum(STEPS[ds]) or st['stages']!=len(SIZES[ds]) or st['policy_updates']:
                raise ValueError('Exact new budget differs')
            if m['source_commit']!=s['source_commit'] or m['anchor_version']!=s['lock']['version'] or m['module']!=arm or m['test_accessed'] or not m['checkpoint_head_rebuild_verified'] or not m['paired_control_inference_recall_verified']:
                raise ValueError('Source/head/control/eval audit differs')
            if f['receipt_status']!='PASS' or not f['receipt_complete'] or f['optimizer_updates'] or f['policy_updates'] or not f['parameters_unchanged'] or f['test_accessed'] or not f['gradient_norm']>0:raise ValueError('Zero-update preflight failed')
            residual.extend([st['max_solve_residual'],f['max_solve_residual']])
            if len(v['scalar_diagnostics'])!=2*len(SIZES[ds]):raise ValueError('Missing complete epochs')
            for d in v['scalar_diagnostics']:
                residual.append(d['max_batch_solve_residual'])
                if not d['dynamic_anchor']['fit_only'] or not d['dynamic_anchor']['detached']:raise ValueError('Target detachment/split differs')
                if d['epoch']==2 and not d['boundary_dynamic_anchor']['meta_excluded_from_target']:raise ValueError('Meta entered target')
            for i,t in enumerate(m['stages']):
                seen=ORDERS[ds][:sum(SIZES[ds][:i+1])]
                if t['seen']!=seen or t['per_class_n']!={str(c):s['lock']['data'][ds]['val_counts'][str(c)] for c in seen}:raise ValueError('Class denominators/order differs')
                residual.extend([t['solve_residual'],t['checkpoint_rebuild_relative_error']])
                for row in t['paired']['classes']:
                    if row['n']!=t['per_class_n'][str(row['label'])] or row['correct_on']-row['correct_control']!=row['rescued']-row['damaged']:raise ValueError('Paired counts inconsistent')
    if max(residual)>1e-8:raise ValueError('Analytic residual failure')
    s.update(modules=list(MODULES),datasets=list(ORDERS),description='Dynamic anchor main grid and matched static anchor controls; all signs, test sealed')
    out=Path(out);summarize(s,out)
    contrasts=[];class_rows=[];paired_rows=[];gaps=[]
    lines=['# 动态锚定：主四格、静态贡献对照及AIR差距','','主方法固定local_transport。控制指标及AIR复用；动态/静态每臂2epoch、原fit/meta与参数不变。静态与动态使用同强度定义，绝对lambda取各自起点G0；训练轨迹分化后不保证数值lambda相等。','','|数据|方法|finalBA|avgBA|tail|forget↓|old|new|Acc|avgAcc|macroF1|AIR−方法 BA|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ORDERS:
        m={a:s['runs'][ds+'_'+a]['metrics'] for a in MODULES}
        for arm in MODULES:
            x=m[arm];value=x.get('final_macro_F1',x.get('final_macroF1'));f1='未提供' if value is None else f'{100*value:.6f}'
            lines.append('|'+ds+'|'+arm+'|'+'|'.join(f'{100*x[k]:.6f}' for k in (*FIELDS,'final_accuracy','average_incremental_accuracy'))+f'|{f1}|{100*(m["AIR"][FIELDS[0]]-x[FIELDS[0]]):+.6f}|')
        for a,b in [(NEW[0],'base'),(NEW[1],'local_transport'),(NEW[2],'base'),(NEW[3],'local_transport'),(NEW[0],NEW[2]),(NEW[1],NEW[3]),(NEW[1],NEW[0]),(NEW[3],NEW[2]),(NEW[1],'base'),*[ (a,'AIR') for a in NEW]]:
            contrasts.append(dict(dataset=ds,contrast=a+'−'+b,delta_pp=delta(m[a],m[b])))
        for mode in ('dynamic','static'):
            a=label('local_transport',mode);b=label('base',mode);x=delta(m[a],m['local_transport']);y=delta(m[b],m['base'])
            contrasts.append(dict(dataset=ds,contrast=mode+'_interaction',delta_pp={k:x[k]-y[k] for k in FIELDS}))
        for arm in NEW:
            control='local_transport' if arm.startswith('local_transport') else 'base'
            gaps.append(dict(dataset=ds,method=arm,control=control,AIR_gap_before_pp=100*(m['AIR'][FIELDS[0]]-m[control][FIELDS[0]]),AIR_gap_after_pp=100*(m['AIR'][FIELDS[0]]-m[arm][FIELDS[0]]),shrink_pp=100*(m[arm][FIELDS[0]]-m[control][FIELDS[0]])))
        for arm in MODULES:
            for stage in m[arm]['stages']:
                for cl,n in stage['per_class_n'].items():class_rows.append(dict(dataset=ds,method=arm,stage=stage['task'],label=int(cl),train_n=s['lock']['data'][ds]['train_counts'][cl],val_n=n,correct_n=round(n*stage['per_class_recall'][cl]),recall_percent=100*stage['per_class_recall'][cl]))
        for arm in NEW:
            for t in m[arm]['stages']:
                for row in t['paired']['classes']:paired_rows.append(dict(dataset=ds,method=arm,stage=t['task'],**row,net=row['rescued']-row['damaged']))
    lines+=['','## 全部六指标差异（pp；forget负值有利）','','|数据|比较|BA|avgBA|tail|forget|old|new|','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in contrasts:lines.append('|'+row['dataset']+'|'+row['contrast']+'|'+'|'.join(f'{row["delta_pp"][k]:+.6f}' for k in FIELDS)+'|')
    for ds in ORDERS:
        lines+=['',f'## {ds}最终每类及新类全部救回/破坏','','|方法|类|角色|n|救回|破坏|净|control间隔|ON间隔|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for row in paired_rows:
            if row['dataset']==ds and row['stage']==len(SIZES[ds]):lines.append(f'|{row["method"]}|{row["label"]}|{row["role"]}|{row["n"]}|{row["rescued"]}|{row["damaged"]}|{row["net"]:+d}|{row["true_margin_mean_control"]:.6f}|{row["true_margin_mean_on"]:.6f}|')
    cost=sum(c['residence_seconds'] for c in p['costs']);ledger=s['closed_round_cost_ledger'].copy();ledger[s['round_name']]=cost
    a=dict(status='PASS',source_commit=s['source_commit'],formal_complete=8,formal_adapter_updates=2576,real_zero_update_preflights=8,successful_cost_records=24,
        policy_updates=0,test_accessed=False,dispatch_gpus=[0,1],max_solve_residual=max(residual),failures=[],cost_records=p['costs'],
        round_gpu_process_residence_seconds=cost,closed_round_cost_ledger=ledger,cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),
        reused_control_training_updates_and_costs=0,new_paired_control_inference_included_in_eval=True,parallel_wall_seconds=p['elapsed_seconds'],whole_round_wall_seconds=p.get('whole_round_wall_seconds'),
        cpu_checks=s['cpu_checks'],engineering_events=s['engineering_events'],new_recall_guarantee=False,nonzero_activation_is_not_utility=True)
    for name,val in [('AUDIT.json',a),('CONTRIBUTIONS.json',contrasts),('AIR_GAPS.json',gaps),('PREFLIGHT_CHECKS.json',s['preflights'])]:
        (out/name).write_text(json.dumps(val,indent=2,allow_nan=False)+'\n')
    for name,rows in [('STAGE_CLASS_RECALLS.csv',class_rows),('PAIRED_CORRECTIONS.csv',paired_rows)]:
        with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines+=['','## 费用和限制','',f'本轮GPU进程驻留{cost:.6f}秒；24条成功operation逐次保留，旧控制训练不收费，配对控制新推理包含在eval费用。CPU及失败见AUDIT。',
        '全stage逐类n/recall在STAGE_CLASS_RECALLS.csv，全stage救回/破坏与间隔在PAIRED_CORRECTIONS.csv，全部混淆矩阵与频数组在RESULTS。',
        '旧类方向仍为统计运输估计；支持/年龄不是校准概率。锚定可能与FD/proximal重复或伤害新类，不把激活当收益，不以平均保护门称逐类无损。',
        '单seed/order反复development val，非患者独立/临床/test证据；AIR解析批数、训练曝光和资源不同，不作等预算SOTA结论。所有正零负保留，不追加seed、强度、预算臂或失败重跑。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return a
