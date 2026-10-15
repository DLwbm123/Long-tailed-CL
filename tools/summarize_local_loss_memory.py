"""Report local loss-jet checks without substituting them for full-query effectiveness."""
import csv
import json
import sys
from pathlib import Path


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text());p=v['program'];lock=json.loads((r/'DESIGN_LOCK.json').read_text())
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    assert [a['id'] for a in p['cases']]==[a['id'] for a in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(a['status']!='NOT_RUN_OVERALL_CAP' for a in p['cases'])
    unit=next((a for a in v['results'] if a['id']=='local_loss_arithmetic'),None)
    results={a['id']:a for a in v['results']};rows=[];classes=[];geometry=[];resources=[]
    for cfg in lock['cases'][1:]:
        a=results.get(cfg['id'],{})
        if a.get('status')=='COMPLETED':
            assert len(a['rows'])==4 and len(a['class_rows'])==4*cfg['classes'] and len(a['geometry'])==cfg['classes']
            rows.extend(dict(case=cfg['id'],**row) for row in a['rows']);classes.extend(dict(case=cfg['id'],**row) for row in a['class_rows']);geometry.extend(dict(case=cfg['id'],**row) for row in a['geometry'])
            resources.append(dict(case=cfg['id'],**{k:a.get(k) for k in ('local_bound_checks_pass','local_finite_query_screen_pass','original_full_query_gate_pass','max_value_bound_violation','max_gradient_bound_violation','max_value_bound','max_gradient_bound','candidate_memory_bytes','baseline_memory_bytes','candidate_optimizer_updates','baseline_optimizer_updates','baseline_initial_objective','baseline_final_objective','baseline_construction_seconds','jet_construction_seconds','evaluation_seconds','CPU_pipeline_seconds','operation_seconds_including_import')}))
        else:rows.append(dict(case=cfg['id'],status=a.get('status','MISSING')))
    complete=len(costs)==4 and len(results)==4 and not p['failures'] and unit is not None and unit.get('status')=='PASS' and len(resources)==3 and all(a['local_bound_checks_pass'] for a in resources)
    local_pass=complete and all(results[c]['local_finite_query_screen_pass'] for c in ('ISIC_scale','HK_scale'))
    total=sum(a['CPU_process_residence_seconds'] for a in costs)
    audit=dict(status='LOCAL_BOUND_CHECKS_PASS' if complete else 'INCOMPLETE_LOCAL_CHECKS',local_finite_signal_pass=local_pass,original_full_query_gate_pass=None,source_commit=p['source_commit'],unit=unit,costs=costs,failures=p['failures'],CPU_residence_seconds=total,CPU_user_seconds=sum(a['CPU_user_seconds'] for a in costs),CPU_system_seconds=sum(a['CPU_system_seconds'] for a in costs),peak_RSS_MiB=max((a['peak_RSS_MiB'] for a in costs),default=None),supervisor_wall_seconds=p['supervisor_wall_seconds'],prior_CPU_seconds=lock['prior_CPU_residence_seconds'],combined_CPU_seconds=lock['prior_CPU_residence_seconds']+total,new_GPU_seconds=0,historical_GPU_seconds=400036.74851310626,client_collection_residence_seconds=None,medical_evidence=False,GPU_ready=False)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    for name,content in [('CPU_SUMMARY',rows),('CLASS_ERRORS',classes),('CURVATURE_BOUNDS',geometry),('RESOURCE_SUMMARY',resources)]:csv_write(r/(name+'.csv'),content)
    report=['# 局部损失记忆 CPU审计','',audit['status'],f'局部有限查询信号：{local_pass}；原完整查询族门：未检验，旧失败不改判。','', '|规模|方法|平均loss误差|最大loss误差|平均gradient L2误差|最大gradient L2误差|','|---|---|---:|---:|---:|---:|']
    for a in rows:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','mode','mean_abs_error','max_abs_error','mean_gradient_l2_error','max_gradient_l2_error'))+'|')
    report+=['','|规模|loss界最大值|gradient界最大值|局部有限门|memory bytes|','|---|---:|---:|---|---:|']
    for a in resources:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','max_value_bound','max_gradient_bound','local_finite_query_screen_pass','candidate_memory_bytes'))+'|')
    report+=['',f"CPU驻留{total}秒/user{audit['CPU_user_seconds']}/system{audit['CPU_system_seconds']}/peak{audit['peak_RSS_MiB']}MiB。父wall{audit['supervisor_wall_seconds']}另列。内部operation/eigh/eval包含驻留不相加。",f"前十五轮{lock['prior_CPU_residence_seconds']}秒，加本轮累计{audit['combined_CPU_seconds']}。失败{p['failures']}。缺失不补0。",'','|case|退出码|驻留秒|user|system|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for a in costs:report.append('|'+ '|'.join(str(a[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','局部loss/gradient均是固定特征、identity adapter、每类当前均值anchor邻域的合成结果；不是共享多类分类轨迹或encoder训练的保证。class-specific anchor/radius需生产机制另行保持，本轮没有验证。0医学/val/test/GPU，非GPU-ready。','完整逐类误差、曲率和成本见CLASS_ERRORS、CURVATURE_BOUNDS和RESOURCE_SUMMARY。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    (r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(cases=len(p['cases']),costs=len(costs),rows=len(rows),class_rows=len(classes),geometry_rows=len(geometry),local_engineering_complete=complete,local_finite_signal_pass=local_pass,full_query_not_checked=True,missing_preserved=True),indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
