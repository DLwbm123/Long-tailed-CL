"""Scalar report for shared-query domain compatibility; no effectiveness inference."""
import csv
import json
import sys
from pathlib import Path


def table(path,rows):
    keys=list(dict.fromkeys(k for a in rows for k in a))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text());p=v['program'];lock=json.loads((r/'DESIGN_LOCK.json').read_text())
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    assert [c['id'] for c in p['cases']]==[c['id'] for c in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(c['status']!='NOT_RUN_OVERALL_CAP' for c in p['cases'])
    rows=[];pairs=[];controls=[];unit=next((a for a in v['results'] if a['id']=='query_compatibility_arithmetic'),None)
    for a in v['results']:
        if a.get('status')=='COMPLETED':
            cfg=a['case'];assert len(a['stages'])==len(cfg['task_sizes']) and len(a['pairs'])==cfg['classes']*(cfg['classes']-1)//2
            rows.extend(dict(case=a['id'],**b) for b in a['stages']);pairs.extend(dict(case=a['id'],**b) for b in a['pairs']);controls.append(dict(case=a['id'],**a['shared_anchor_algebraic_control']))
        elif a['id']!='query_compatibility_arithmetic':rows.append(dict(case=a['id'],status=a['status']))
    complete=len(costs)==4 and not p['failures'] and unit is not None and unit.get('status')=='PASS' and len(rows)==12 and len(pairs)==287
    total=sum(c['CPU_process_residence_seconds'] for c in costs)
    audit=dict(status='DOMAIN_AUDIT_COMPLETE' if complete else 'INCOMPLETE_DOMAIN_AUDIT',source_commit=p['source_commit'],unit=unit,costs=costs,failures=p['failures'],stage_status_counts={s:sum(a['status']==s for a in rows) for s in ('EMPTY_INTERSECTION_CERTIFIED','NONEMPTY_WITNESS','UNRESOLVED_BY_PAIRWISE_TEST')},CPU_residence_seconds=total,CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in costs),default=None),supervisor_wall_seconds=p['supervisor_wall_seconds'],prior_CPU_seconds=lock['prior_CPU_residence_seconds'],combined_CPU_seconds=lock['prior_CPU_residence_seconds']+total,new_GPU_seconds=0,historical_GPU_seconds=400036.74851310626,client_collection_residence_seconds=None,method_effectiveness_claim=False,medical_evidence=False)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');table(r/'CPU_SUMMARY.csv',rows);table(r/'PAIR_GEOMETRY.csv',pairs);table(r/'SHARED_ANCHOR_CONTROL.csv',controls)
    report=['# 共享查询域兼容性审计','',audit['status'],'','|规模|阶段|已见类|结论|不兼容pair数|共同半径必要下界|均值center半径上界|','|---|---:|---:|---|---:|---:|---:|']
    for a in rows:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','task','seen_classes','status','incompatible_pairs','necessary_common_radius_lower_bound','mean_center_radius_upper_bound'))+'|')
    report+=['','原radius .125不改。pair距>2r足以证明交集为空；pair全部不违反不是充分可行条件。shared-anchor代数control不代表loss或分类有效。',f"CPU驻留{total}秒/user{audit['CPU_user_seconds']}/system{audit['CPU_system_seconds']}/peak{audit['peak_RSS_MiB']}MiB；父wall{audit['supervisor_wall_seconds']}另列。内部operation含驻留不加。",f"前16轮{lock['prior_CPU_residence_seconds']}秒，累计{audit['combined_CPU_seconds']}。失败{p['failures']}，缺失不补0。",'','|case|退出码|驻留秒|user|system|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for c in costs:report.append('|'+ '|'.join(str(c[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','仅针对同一个v需同时调用多类局部记忆的参数域要求，不推广到不同参数耦合的所有多类方法。没有分类训练或loss评价，不改判旧局部精度或完整查询族失败。0医学images/features/checkpoint/val/test/GPU。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');(r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(cases=len(p['cases']),costs=len(costs),stage_rows=len(rows),pair_rows=len(pairs),control_rows=len(controls),audit_complete=complete,missing_preserved=True),indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
