"""Report frozen CPU cases without loading features or rerunning science."""
import csv
import json
import sys
from pathlib import Path


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text());p=v['program']
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    lock=json.loads((r/'DESIGN_LOCK.json').read_text())
    assert [c['id'] for c in p['cases']]==[c['id'] for c in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(c['status']!='NOT_RUN_OVERALL_CAP' for c in p['cases'])
    rows=[];checks={}
    for result in v['results']:
        if result['id']=='arithmetic':checks['arithmetic']=result.get('status')=='PASS'
        if 'rows' not in result:
            if result['id']!='arithmetic':rows.append(dict(case=result['id'],mode=None,status=result['status']))
            continue
        assert {a['mode'] for a in result['rows']}==set(lock['modes'])
        for a in result['rows']:
            rows.append(dict(case=result['id'],mode=a['mode'],status=result['status'],
               mean_abs_error=a['mean_abs_error'],max_abs_error=a['max_abs_error'],
               random_max=a['random_query_max_error'],stress_max=a['stress_query_max_error'],
               moment_objective=a['moment_objective'],memory_bytes=a['logical_memory_bytes']))
        if result['id'] in ('ISIC_scale','HK_scale'):checks[result['id']]=result['finite_query_screen_pass']
    passed=all(checks.get(k,False) for k in ('arithmetic','ISIC_scale','HK_scale')) and not p['failures'] and len(costs)==4
    audit=dict(status='FINITE_QUERY_SIGNAL' if passed else 'NO_COMPLETE_FINITE_QUERY_SIGNAL',
        scientific_source_commit=p['source_commit'],checks=checks,
        costs=costs,CPU_residence_seconds=sum(c['CPU_process_residence_seconds'] for c in costs),
        CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),
        CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),
        peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in costs),default=None),
        supervisor_wall_seconds=p['supervisor_wall_seconds'],failures=p['failures'],
        new_GPU_seconds=0,closed_historical_GPU_seconds=400036.74851310626,
        uniform_certificate=False,medical_effect_evidence=False,
        client_and_collection_residence_seconds=None,formal_training_performed=False,
        future_recompression_tested=False,production_integrated=False)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    fields=['case','mode','status','mean_abs_error','max_abs_error','random_max','stress_max','moment_objective','memory_bytes']
    with (r/'CPU_SUMMARY.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report=['# 判别记忆 CPU 第一轮结果','',f"结论：{audit['status']}。仅有限查询误差与资源检查，不是医学效果或全族误差证书。",'',
            '|规模|记忆|均值绝对误差|最大绝对误差|随机最大|stress最大|字节|','|---|---|---:|---:|---:|---:|---:|']
    for a in rows:
        report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','mode','mean_abs_error','max_abs_error','random_max','stress_max','memory_bytes'))+'|')
    report+=['',f"冻结门：{checks}；失败：{p['failures']}。缺失为null，不补0。",'',
      f"CPU独立驻留 {audit['CPU_residence_seconds']} 秒，user {audit['CPU_user_seconds']}，system {audit['CPU_system_seconds']}，peak RSS {audit['peak_RSS_MiB']} MiB。父wall {audit['supervisor_wall_seconds']} 秒另列。mode/operation包含于驻留，不相加。",'',
      '## 逐次成本','', '|case|退出码|驻留秒|user秒|system秒|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for c in costs:report.append('|'+ '|'.join(str(c[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','0 GPU/医学图像/医学checkpoint/val/test/正式更新。旧GPU历史400036.74851310626秒不变。',
      '未验证完整序列训练/重压缩累计误差/真实分类收益。均值基线占位16槽但只有1唯一点，random仅合成fixture真实点，不可自动转为医学原始特征回放。',
      '所有查询评价只支持有限测试信号；stress为共同针对四种方法的发现过程，不是统一sup上界。','',
      '下一步须先公开交付、解释完整正负；失败不自动增加节点、预算、seed或降低门槛。持续授权只允许有机制理由的新协议轮。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    (r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(case_count=len(p['cases']),cost_count=len(costs),rows=len(rows),checks=checks,missing_preserved=True,counts_checked=True),indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
