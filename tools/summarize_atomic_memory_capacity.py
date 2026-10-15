"""Report scalar capacity witnesses; no reconstruction or experiment execution."""
import csv
import json
import sys
from pathlib import Path


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text());p=v['program']
    lock=json.loads((r/'DESIGN_LOCK.json').read_text())
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    assert [c['id'] for c in p['cases']]==[c['id'] for c in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(c['status']!='NOT_RUN_OVERALL_CAP' for c in p['cases'])
    rows=[];checks={}
    for a in v['results']:
        checks[a['id']]=a.get('status')=='PASS' if a['id']=='capacity_arithmetic' else a.get('all_inequalities_verified',False)
        for z in a.get('rows',[]):rows.append(dict(case=a['id'],**z))
    total=sum(c['CPU_process_residence_seconds'] for c in costs)
    audit=dict(status='CAPACITY_ARITHMETIC_CHECKED' if all(checks.get(c['id'],False) for c in lock['cases']) and not p['failures'] else 'INCOMPLETE_CAPACITY_CHECK',
        checks=checks,source_commit=p['source_commit'],costs=costs,failures=p['failures'],
        CPU_residence_seconds=total,CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),
        CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in costs),default=None),
        supervisor_wall_seconds=p['supervisor_wall_seconds'],prior_CPU_seconds=lock['prior_CPU_residence_seconds'],
        combined_CPU_seconds=lock['prior_CPU_residence_seconds']+total,new_GPU_seconds=0,
        historic_GPU_seconds=400036.74851310626,client_collection_residence_seconds=None,
        medical_evidence=False,causal_explanation_of_prior_fixture=False,method_improvement_tested=False)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    fields=list(rows[0]) if rows else ['case','mode']
    with (r/'CPU_SUMMARY.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    report=['# 固定原子记忆容量反例 CPU 结果','',str(audit['status']), '',
        '|维数|构造|rank|trace|谱下界|仅trace下界|实际见证误差|','|---:|---|---:|---:|---:|---:|---:|']
    for a in rows:report.append('|'+ '|'.join(str(a[k]) for k in ('dimension','mode','rank','trace','spectrum_lower_bound','trace_only_lower_bound','witness_max_abs_error'))+'|')
    report+=['',f"独立CPU驻留{total}秒，user{audit['CPU_user_seconds']}，system{audit['CPU_system_seconds']}，peak RSS{audit['peak_RSS_MiB']}MiB；父wall{audit['supervisor_wall_seconds']}秒另列。内部operation包含其中不相加。",'',
        f"前六轮{lock['prior_CPU_residence_seconds']}秒，累计{audit['combined_CPU_seconds']}秒。失败{p['failures']}；缺失null不补0。",'',
        '|case|退出码|驻留秒|user|system|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for c in costs:report.append('|'+ '|'.join(str(c[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','这是各向同性参考的存在性反例和解析不等式数值检查，不是所有16节点记忆无效或前轮失败原因的证明。',
        'identity adapter仍归一化非零节点，因此球内缩短非零节点不会逃离本归一化查询的单位方向容量限制。零节点仅退化负对照。',
        '0医学images/features/checkpoints/val/test/GPU/optimizer，未提出或验证新分类改进，不能放宽旧门或直接启动GPU。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    (r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(case_count=len(p['cases']),cost_count=len(costs),rows=len(rows),checks=checks,missing_preserved=True),indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
