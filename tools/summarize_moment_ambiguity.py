"""Scalar-only closeout of a deterministic representation counterexample."""
import csv
import json
import sys
from pathlib import Path


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text());p=v['program'];lock=json.loads((r/'DESIGN_LOCK.json').read_text())
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    assert [c['id'] for c in p['cases']]==[c['id'] for c in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(c['status']!='NOT_RUN_OVERALL_CAP' for c in p['cases'])
    rows=[];batches=[];checks=[]
    for a in v['results']:
        checks.append(dict(case=a['id'],status=a['status'],checks=a.get('checks')))
        if 'rows' in a:
            assert len(a['rows'])==8 and len(a['batches'])==128
            rows.extend(dict(case=a['id'],**b) for b in a['rows']);batches.extend(dict(case=a['id'],**b) for b in a['batches'])
        elif a['id']!='ambiguity_arithmetic':rows.append(dict(case=a['id'],status=a['status']))
    passed=len(costs)==3 and not p['failures'] and len(v['results'])==3 and all(a['status']==('PASS' if a['id']=='ambiguity_arithmetic' else 'COMPLETED') for a in v['results'])
    total=sum(c['CPU_process_residence_seconds'] for c in costs)
    audit=dict(status='COUNTEREXAMPLE_CHECKS_PASS' if passed else 'INCOMPLETE_COUNTEREXAMPLE_CHECKS',method_effectiveness_claim=False,source_commit=p['source_commit'],checks=checks,costs=costs,failures=p['failures'],CPU_residence_seconds=total,CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in costs),default=None),supervisor_wall_seconds=p['supervisor_wall_seconds'],prior_CPU_seconds=lock['prior_CPU_residence_seconds'],combined_CPU_seconds=lock['prior_CPU_residence_seconds']+total,new_GPU_seconds=0,historical_GPU_seconds=400036.74851310626,client_collection_residence_seconds=None,medical_evidence=False)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');write_csv(r/'CPU_SUMMARY.csv',rows);write_csv(r/'SAMPLING_AUDIT.csv',batches)
    report=['# 同矩歧义 CPU反例审计','',audit['status'],'','|规模|均值长度|查询|参考A损失|参考B损失|至少一个参考误差下界|','|---|---:|---|---:|---:|---:|']
    for a in rows:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','mean_length','query','reference_A_loss','reference_B_loss','common_prediction_error_lower_bound'))+'|')
    report+=['',f"CPU驻留{total}秒/user{audit['CPU_user_seconds']}/system{audit['CPU_system_seconds']}/peakRSS{audit['peak_RSS_MiB']}MiB；父wall{audit['supervisor_wall_seconds']}另列。内部operation含驻留不加。",f"前14轮{lock['prior_CPU_residence_seconds']}，累计{audit['combined_CPU_seconds']}秒。失败{p['failures']}，missing不补0。",'','|case|退出码|驻留秒|user|system|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for c in costs:report.append('|'+ '|'.join(str(c[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','三角下界只是共同预测至少一个参考的误差，不是两者均失败/所有查询上界。两个旋转单纯形同5点、同一二阶矩，与旧HK fixture不同。','0memory/query optimizer、0医学images/features/checkpoint/val/test/GPU。不是新方法有效性或前轮失败因果证明，不允许据此放宽旧门。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');(r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(cases=len(p['cases']),costs=len(costs),rows=len(rows),sampling_rows=len(batches),arithmetic_complete=passed,missing_preserved=True),indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
