"""Aggregate frozen scalar results; batches are never averaged into the primary outcome."""
import csv
import json
import sys
from pathlib import Path


def write_csv(path,rows,fields):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text());p=v['program']
    lock=json.loads((r/'DESIGN_LOCK.json').read_text());assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    assert [c['id'] for c in p['cases']]==[c['id'] for c in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(c['status']!='NOT_RUN_OVERALL_CAP' for c in p['cases'])
    checks={};rows=[];classes=[];diagnostics=[]
    for a in v['results']:
        if a['id']=='spectral_arithmetic':checks['arithmetic']=a.get('status')=='PASS';continue
        if 'rows' not in a:
            rows.append(dict(case=a['id'],status=a['status']));continue
        assert len(a['batches'])==2*lock['audit_batches'] and len(a['rows'])==2
        checks[a['id']]=a['finite_query_screen_pass']
        for b in a['rows']+a['batches']:
            base=dict(case=a['id'],status=a['status'],mode=b['mode'],batch=b.get('batch'))
            rows.append(dict(**base,**{k:b[k] for k in ('mean_abs_error','max_abs_error','random_max','stress_max')}))
            for j in range(a['case']['classes']):classes.append(dict(**base,class_index=j,mean_abs_error=b['per_class_mean'][j],max_abs_error=b['per_class_max'][j]))
        diagnostics.extend(dict(case=a['id'],**b) for b in a['averaged_prediction_diagnostic'])
    (r/'SAMPLING_DIAGNOSTICS.json').write_text(json.dumps([dict(case=a['id'],variance=a.get('prediction_variance_diagnostic'),pairing_seconds=a.get('pairing_seconds'),pair_cache_bytes_per_method=a.get('pair_cache_bytes_per_method'),pair_mean_sq_distance=a.get('pair_mean_sq_distance'),base_memory_bytes=a.get('base_memory_bytes'),candidate_memory_bytes=a.get('candidate_memory_bytes'),fresh_batch_bytes=a.get('fresh_batch_bytes'),isotropic_control_memory_bytes=a.get('isotropic_control_memory_bytes'),baseline_initial_objective=a.get('baseline_initial_objective'),baseline_final_objective=a.get('baseline_final_objective'),baseline_construction_seconds=a.get('baseline_construction_seconds'),spectral_construction_seconds=a.get('spectral_construction_seconds'),isotropic_memory_bytes=a.get('isotropic_memory_bytes'),analytic_geometry=a.get('analytic_geometry'),case_total_memory_updates=a.get('case_total_memory_updates')) for a in v['results'] if 'rows' in a],indent=2)+'\n')
    passed=all(checks.get(k,False) for k in ('arithmetic','ISIC_scale','HK_scale')) and not p['failures'] and len(costs)==4
    total=sum(c['CPU_process_residence_seconds'] for c in costs)
    audit=dict(status='FINITE_FRESH_BATCH_SIGNAL' if passed else 'NO_COMPLETE_FRESH_BATCH_SIGNAL',checks=checks,
        source_commit=p['source_commit'],costs=costs,failures=p['failures'],CPU_residence_seconds=total,
        CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),
        peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in costs),default=None),supervisor_wall_seconds=p['supervisor_wall_seconds'],
        prior_CPU_seconds=lock['prior_CPU_residence_seconds'],combined_CPU_seconds=lock['prior_CPU_residence_seconds']+total,
        new_GPU_seconds=0,historic_GPU_seconds=400036.74851310626,client_collection_residence_seconds=None,
        medical_evidence=False,uniform_post_draw_query_certificate=False,unbiased_to_true_distribution=False,
        multiple_batches_averaged_for_primary=False)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    write_csv(r/'CPU_SUMMARY.csv',rows,['case','status','mode','batch','mean_abs_error','max_abs_error','random_max','stress_max'])
    write_csv(r/'CLASS_ERRORS.csv',classes,['case','status','mode','batch','class_index','mean_abs_error','max_abs_error'])
    (r/'AVERAGING_DIAGNOSTIC_NOT_PRIMARY.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    report=['# 同预算谱分布记忆 CPU 结果','',audit['status'],f"冻结门：{checks}",'',
        '|规模|方法|批次|平均误差|最大误差|','|---|---|---:|---:|---:|']
    for a in rows:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','mode','batch','mean_abs_error','max_abs_error'))+'|')
    report+=['',f"CPU独立驻留{total}秒，user{audit['CPU_user_seconds']}，system{audit['CPU_system_seconds']}，peak RSS{audit['peak_RSS_MiB']}MiB；父wall{audit['supervisor_wall_seconds']}另列。内部构造/eval/operation包含于驻留，不相加。",'',
        f"先前十一轮{lock['prior_CPU_residence_seconds']}秒，累计{audit['combined_CPU_seconds']}秒。失败{p['failures']}；缺失null不补0。",'',
        '|case|退出码|驻留秒|user|system|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for c in costs:report.append('|'+ '|'.join(str(c[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','0GPU/医学images/features/checkpoints/val/test/正式更新。每次只使用16点，8批仅审计，不平均替代主门。',
        '新查询先固定再独立采样；不能承诺观察当次样本后再自适应选择的所有查询。生成分布估计无偏不等真实旧分布无偏。',
        '归一化可改变均值，原二阶残差不保证各向同性或PSD。谱表示与原moment存储精确相等，基线48步、候选无优化器但eig成本全部计费；raw矩不等于归一化后的矩或高阶查询准确。尚无完整序列/分类训练/生产预算证据。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    (r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(cases=len(p['cases']),costs=len(costs),rows=len(rows),class_rows=len(classes),checks=checks,missing_preserved=True),indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
