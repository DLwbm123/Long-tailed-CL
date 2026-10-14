"""PMT main grid, independent regional contribution, class damage and AIR gaps."""
import csv
import json
from pathlib import Path

from summarize_module_additions import summarize
from summarize_residual_drift import FIELDS,delta
from run_lca_supported_cycle import ORDERS,SIZES,STEPS

NEW=('pmt','local_transport_pmt','local_transport_pmt_global_only')
MODULES=('base','local_transport',*NEW,'AIR')


def report(s,out):
    p=s['program'];residual=[]
    if p['status']!='COMPLETE' or p['active'] or p['failures'] or len(p['costs'])!=18:raise ValueError('18 successful operations and empty active required')
    if any(c['returncode'] or c['gpu'] not in (0,1) for c in p['costs']):raise ValueError('GPU cost/scope failure')
    for ds in ORDERS:
        for arm in NEW:
            v=s['runs'][ds+'_'+arm];st=v['status'];m=v['metrics'];f=s['preflights'][ds+'_'+arm]
            if st['status']!='COMPLETE' or st['steps']!=sum(STEPS[ds]) or st['stages']!=len(SIZES[ds]) or st['policy_updates']:
                raise ValueError('Exact new budget differs')
            if m['source_commit']!=s['source_commit'] or m['pmt_version']!=s['lock']['version'] or m['module']!=arm or m['test_accessed'] or not m['checkpoint_head_rebuild_verified'] or not m['paired_control_inference_recall_verified']:
                raise ValueError('Source/head/control/eval audit differs')
            if not f.get('synthetic_highest_bank',{}).get('synthetic_only') or f.get('synthetic_highest_bank',{}).get('dimension')!=1536:raise ValueError('Highest-bank GPU shadow missing')
            if f['receipt_status']!='PASS' or not f['receipt_complete'] or f['optimizer_updates'] or f['policy_updates'] or not f['parameters_unchanged'] or f['test_accessed'] or not f['gradient_norm']>0:raise ValueError('Zero-update preflight failed')
            residual.extend([st['max_solve_residual'],f['max_solve_residual']])
            if len(v['scalar_diagnostics'])!=2*len(SIZES[ds]):raise ValueError('Missing complete epochs')
            for d in v['scalar_diagnostics']:
                residual.append(d['max_batch_solve_residual'])
                if not d['paired_transport']['fit_only'] or not d['paired_transport']['detached']:raise ValueError('Target detachment/split differs')
                if d['epoch']==2 and not d['boundary_paired_transport']['current_train_including_meta']:raise ValueError('Native all-current boundary exposure differs')
            if len(m['stages'])!=len(SIZES[ds]):raise ValueError('Complete stage metrics required')
            for i,t in enumerate(m['stages']):
                seen=ORDERS[ds][:sum(SIZES[ds][:i+1])]
                if t['seen']!=seen or t['per_class_n']!={str(c):s['lock']['data'][ds]['val_counts'][str(c)] for c in seen}:raise ValueError('Class denominators/order differs')
                residual.extend([t['solve_residual'],t['checkpoint_rebuild_relative_error']])
                for row in t['paired']['classes']:
                    if row['n']!=t['per_class_n'][str(row['label'])] or row['correct_on']-row['correct_control']!=row['rescued']-row['damaged']:raise ValueError('Paired counts inconsistent')
    if not all(__import__('math').isfinite(float(x)) for x in residual):raise ValueError('Nonfinite residual')
    if max(residual)>1e-8:raise ValueError('Analytic residual failure')
    expected={(prefix+ds+'_'+arm,operation) for ds in ORDERS for arm in NEW for prefix,operation in [('pre_','train'),('','train'),('','evaluate')]}
    from collections import Counter
    if Counter((c['job'],c['operation']) for c in p['costs'])!=Counter({key:1 for key in expected}):raise ValueError('Every pre/train/eval operation must be preserved once')
    s.update(modules=list(MODULES),datasets=list(ORDERS),description='PMT main grid and regional mechanism control; all signs, test sealed')
    out=Path(out);summarize(s,out)
    contrasts=[];class_rows=[];paired_rows=[];gaps=[];map_rows=[]
    lines=['# 配对矩运输：主四格、独立区域贡献及AIR差距','','主方法固定local_transport。控制指标及AIR复用；新臂每task 2epoch、原fit/meta与参数不变。PMT更新旧全局矩，组合另拟合四个区域映射；global_only保留原区域平移。当前类使用真实统计，不套旧映射。','','|数据|方法|finalBA|avgBA|tail|forget↓|old|new|Acc|avgAcc|macroF1|AIR−方法 BA|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ORDERS:
        m={a:s['runs'][ds+'_'+a]['metrics'] for a in MODULES}
        for arm in MODULES:
            x=m[arm];value=x.get('final_macro_F1',x.get('final_macroF1'));f1='未提供' if value is None else f'{100*value:.6f}'
            lines.append('|'+ds+'|'+arm+'|'+'|'.join(f'{100*x[k]:.6f}' for k in (*FIELDS,'final_accuracy','average_incremental_accuracy'))+f'|{f1}|{100*(m["AIR"][FIELDS[0]]-x[FIELDS[0]]):+.6f}|')
        for a,b in [(NEW[0],'base'),(NEW[1],'local_transport'),(NEW[2],'local_transport'),(NEW[1],NEW[2]),(NEW[1],NEW[0]),(NEW[2],NEW[0]),(NEW[1],'base'),*[(a,'AIR') for a in NEW]]:
            contrasts.append(dict(dataset=ds,contrast=a+'−'+b,delta_pp=delta(m[a],m[b])))
        for arm in NEW[1:]:
            x=delta(m[arm],m['local_transport']);y=delta(m[NEW[0]],m['base'])
            contrasts.append(dict(dataset=ds,contrast=arm+'_interaction',delta_pp={k:x[k]-y[k] for k in FIELDS}))
        for arm in NEW:
            control='local_transport' if arm.startswith('local_transport') else 'base'
            gaps.append(dict(dataset=ds,method=arm,control=control,AIR_gap_before_pp=100*(m['AIR'][FIELDS[0]]-m[control][FIELDS[0]]),AIR_gap_after_pp=100*(m['AIR'][FIELDS[0]]-m[arm][FIELDS[0]]),shrink_pp=100*(m[arm][FIELDS[0]]-m[control][FIELDS[0]])))
        for arm in MODULES:
            for stage in m[arm]['stages']:
                for cl,n in stage['per_class_n'].items():class_rows.append(dict(dataset=ds,method=arm,stage=stage['task'],label=int(cl),train_n=s['lock']['data'][ds]['train_counts'][cl],val_n=n,correct_n=round(n*stage['per_class_recall'][cl]),recall_percent=100*stage['per_class_recall'][cl]))
        for arm in NEW:
            for diagnostic in s['runs'][ds+'_'+arm]['scalar_diagnostics']:
                for phase,key in [('fit','paired_transport'),('boundary','boundary_paired_transport')]:
                    if key in diagnostic:map_rows.append(dict(dataset=ds,method=arm,task=diagnostic.get('task'),epoch=diagnostic['epoch'],phase=phase,**diagnostic[key]))
            for t in m[arm]['stages']:
                for row in t['paired']['classes']:paired_rows.append(dict(dataset=ds,method=arm,stage=t['task'],**row,net=row['rescued']-row['damaged']))
    lines+=['','## 全部六指标差异（pp；forget负值有利）','','|数据|比较|BA|avgBA|tail|forget|old|new|','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in contrasts:lines.append('|'+row['dataset']+'|'+row['contrast']+'|'+'|'.join(f'{row["delta_pp"][k]:+.6f}' for k in FIELDS)+'|')
    for ds in ORDERS:
        lines+=['',f'## {ds}最终每类及新类全部救回/破坏','','|方法|类|角色|n|救回|破坏|净|control间隔|ON间隔|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for row in paired_rows:
            if row['dataset']==ds and row['stage']==len(SIZES[ds]):lines.append(f'|{row["method"]}|{row["label"]}|{row["role"]}|{row["n"]}|{row["rescued"]}|{row["damaged"]}|{row["net"]:+d}|{row["true_margin_mean_control"]:.6f}|{row["true_margin_mean_on"]:.6f}|')
    if not all(__import__('math').isfinite(float(x)) for x in residual):raise ValueError('Nonfinite residual')
    cost=sum(c['residence_seconds'] for c in p['costs']);ledger=s['closed_round_cost_ledger'].copy();ledger[s['round_name']]=cost
    a=dict(status='PASS',source_commit=s['source_commit'],formal_complete=6,formal_adapter_updates=1932,real_zero_update_preflights=6,successful_cost_records=18,
        policy_updates=0,test_accessed=False,dispatch_gpus=[0,1],max_solve_residual=max(residual),failures=[],cost_records=p['costs'],
        round_gpu_process_residence_seconds=cost,closed_round_cost_ledger=ledger,cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),
        reused_control_training_updates_and_costs=0,new_paired_control_inference_included_in_eval=True,parallel_wall_seconds=p['elapsed_seconds'],whole_round_wall_seconds=p.get('whole_round_wall_seconds'),preflight_gate_wait_wall_seconds=p.get('preflight_gate_wait_wall_seconds'),resource_admission_revised_before_GPU=p.get('resource_admission_revised_before_GPU',False),
        cpu_checks=s['cpu_checks'],engineering_events=s['engineering_events'],new_recall_guarantee=False,nonzero_activation_is_not_utility=True)
    for name,val in [('AUDIT.json',a),('CONTRIBUTIONS.json',contrasts),('AIR_GAPS.json',gaps),('PREFLIGHT_CHECKS.json',s['preflights'])]:
        (out/name).write_text(json.dumps(val,indent=2,allow_nan=False)+'\n')
    for name,rows in [('STAGE_CLASS_RECALLS.csv',class_rows),('PAIRED_CORRECTIONS.csv',paired_rows),('MAP_AUDIT.csv',map_rows)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines+=['','## 训练映射审计','','|数据|方法|task|epoch|阶段|rank|配对MSE|线性变化范数|区域rank和|旧未知半径数|半径上界最大|','|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in map_rows:
        lines.append('|'+ '|'.join(str(row.get(k,'未提供')) for k in ('dataset','method','task','epoch','phase','rank','paired_weighted_MSE','linear_delta_frobenius','regional_rank_sum','old_radius_unknown','old_radius_upper_max'))+'|')
    lines+=['','## 费用和限制','',f'本轮GPU进程驻留{cost:.6f}秒；18条成功operation逐次保留，旧控制训练不收费，配对控制新推理包含在eval费用。CPU及失败见AUDIT。',
        '全stage逐类n/recall在STAGE_CLASS_RECALLS.csv，全stage救回/破坏与间隔在PAIRED_CORRECTIONS.csv，全部混淆矩阵与频数组在RESULTS。',
        '旧类仿射推广是假设：未观察方向保持残差identity不代表无漂移。旧样本不重新归一化，未知半径仅报告界。不能把拟合残差或映射激活当分类收益，更不保证新类无损。',
        '单seed/order反复development val，非患者独立/临床/test证据；AIR解析批数、训练曝光和资源不同，不作等预算SOTA结论。所有正零负保留，不追加seed、强度、预算臂或失败重跑。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return a
