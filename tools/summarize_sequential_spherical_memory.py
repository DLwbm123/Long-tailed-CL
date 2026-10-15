"""Scalar-only terminal report; never reruns memory construction or reads checkpoints."""
import csv
import json
import sys
from pathlib import Path


def write_csv(path,rows,fields):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text())
    p=v['program'];lock=json.loads((r/'DESIGN_LOCK.json').read_text())
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    assert [c['id'] for c in p['cases']]==[c['id'] for c in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(c['status']!='NOT_RUN_OVERALL_CAP' for c in p['cases'])
    rows=[];classes=[];stages=[];checks={}
    for result in v['results']:
        if result['id']=='sequence_arithmetic':
            checks['arithmetic']=result.get('status')=='PASS';continue
        if 'stages' not in result:
            rows.append(dict(case=result['id'],status=result['status']));continue
        assert len(result['stages'])==len(result['case']['task_sizes'])
        checks[result['id']]=result['finite_query_screen_pass']
        for s in result['stages']:
            assert {a['mode'] for a in s['rows']}==set(lock['modes'])
            base=dict(case=result['id'],task=s['task'],seen=s['seen'],old=s['old'])
            stages.append(dict(**base,old_memory_exact=s['old_memory_exact'],old_fixed_query_error=s['old_fixed_query_error'],
                ETF_error=s['rescaled_ETF_old_logit_error'],checkpoint_exact=s['checkpoint_exact'],
                checkpoint_bytes=s['checkpoint_bytes'],screen_pass=s['finite_query_screen_pass']))
            for a in s['rows']:
                rows.append(dict(**base,status=result['status'],**{k:a[k] for k in (
                    'mode','mean_abs_error','max_abs_error','random_max','stress_max','old_mean_abs_error',
                    'old_max_abs_error','new_mean_abs_error','new_max_abs_error','logical_memory_bytes','min_effective_nodes')}))
                assert len(a['per_class_mean'])==len(a['per_class_max'])==len(a['second_trace'])==s['seen']
                for j in range(s['seen']):
                    classes.append(dict(**base,mode=a['mode'],class_index=j,group='old' if j<s['old'] else 'new',
                        mean_abs_error=a['per_class_mean'][j],max_abs_error=a['per_class_max'][j],second_trace=a['second_trace'][j]))
    passed=all(checks.get(k,False) for k in ('arithmetic','ISIC_scale','HK_scale')) and not p['failures'] and len(costs)==4
    total=sum(c['CPU_process_residence_seconds'] for c in costs)
    audit=dict(status='FINITE_SEQUENCE_QUERY_SIGNAL' if passed else 'NO_COMPLETE_SEQUENCE_SIGNAL',checks=checks,
        scientific_source_commit=p['source_commit'],costs=costs,CPU_residence_seconds=total,
        CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),
        peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in costs),default=None),
        supervisor_wall_seconds=p['supervisor_wall_seconds'],failures=p['failures'],
        prior_design_CPU_seconds=lock['prior_CPU_residence_seconds'],combined_design_CPU_seconds=lock['prior_CPU_residence_seconds']+total,
        new_GPU_seconds=0,closed_historical_GPU_seconds=400036.74851310626,
        client_collection_residence_seconds=None,medical_effect_evidence=False,production_integrated=False,
        classifier_training_tested=False,lossy_recompression_tested=False,uniform_certificate=False)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    fields=['case','task','seen','old','status','mode','mean_abs_error','max_abs_error','random_max','stress_max',
        'old_mean_abs_error','old_max_abs_error','new_mean_abs_error','new_max_abs_error','logical_memory_bytes','min_effective_nodes']
    write_csv(r/'CPU_SUMMARY.csv',rows,fields)
    write_csv(r/'CLASS_ERRORS.csv',classes,['case','task','seen','old','mode','class_index','group','mean_abs_error','max_abs_error','second_trace'])
    write_csv(r/'INTERFACE_AUDIT.csv',stages,['case','task','seen','old','old_memory_exact','old_fixed_query_error','ETF_error','checkpoint_exact','checkpoint_bytes','screen_pass'])
    report=['# 序列球面记忆 CPU 结果','',f"结论：{audit['status']}；冻结门 {checks}。",'',
        '|规模|stage|方法|平均误差|最大误差|旧类均误差|新类均误差|','|---|---:|---|---:|---:|---:|---:|']
    for a in rows:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','task','mode','mean_abs_error','max_abs_error','old_mean_abs_error','new_mean_abs_error'))+'|')
    report+=['',f"CPU独立驻留{total}秒；user {audit['CPU_user_seconds']} / system {audit['CPU_system_seconds']}；peak {audit['peak_RSS_MiB']} MiB。父wall {audit['supervisor_wall_seconds']}秒另列。内部构造/评价/operation已包含，不相加。",'',
        f"既有五轮{lock['prior_CPU_residence_seconds']}秒，本轮后合计{audit['combined_design_CPU_seconds']}秒。失败{p['failures']}；缺失null不补0。",'',
        '|case|退出码|驻留秒|user|system|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for c in costs:report.append('|'+ '|'.join(str(c[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','0 GPU/医学图像/医学checkpoint/val/test/正式更新。没有分类训练或损失性重压缩；合成旧支持仅独立oracle评价。',
        'ETF恒等仅固定输入旧旧差，不保护新类竞争。共同stress不是独立population检验或sup证书。每类16节点的总容量随类增长，非固定全局预算。',
        '本轮主候选预声明为sphere；不得回写上一轮sphere_weight失败。原Core/LT不变，不从本CPU直接进入医学GPU。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    (r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(cases=len(p['cases']),costs=len(costs),rows=len(rows),class_rows=len(classes),stages=len(stages),checks=checks,missing_preserved=True),indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
