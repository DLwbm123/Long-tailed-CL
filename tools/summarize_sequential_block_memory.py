"""Report scalar-only sequence results including partial failed stages, without rerunning science."""
import csv
import json
import sys
from pathlib import Path


def write_csv(path,rows,fields):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text());p=v['program'];lock=json.loads((r/'DESIGN_LOCK.json').read_text())
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    assert [a['id'] for a in p['cases']]==[a['id'] for a in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(a['status']!='NOT_RUN_OVERALL_CAP' for a in p['cases'])
    rows=[];classes=[];interfaces=[];diagnostics=[];checks={}
    metrics=['mean_abs_error','max_abs_error','random_max','stress_max','old_mean_abs_error','old_max_abs_error','new_mean_abs_error','new_max_abs_error']
    for a in v['results']:
        if a['id']=='sequence_block_arithmetic':checks['arithmetic']=a.get('status')=='PASS';continue
        stages=a.get('stages')
        complete=stages is not None and a.get('status')=='COMPLETED'
        if stages is None:stages=(a.get('last_progress') or {}).get('completed_stage_results',[])
        checks[a['id']]=complete and a.get('finite_query_screen_pass',False)
        if not stages:rows.append(dict(case=a['id'],status=a.get('status')))
        for s in stages:
            assert len(s['rows'])==18 and {b['mode'] for b in s['rows']}==set(lock['modes'])
            base=dict(case=a['id'],task=s['task'],seen=s['seen'],old=s['old'])
            interfaces.append(dict(**base,old_memory_exact=s['old_memory_exact'],old_draw_exact=s['old_draw_exact'],old_fixed_query_error=s['old_fixed_query_error'],ETF_error=s['rescaled_ETF_old_logit_error'],checkpoint_exact=s['checkpoint_exact'],checkpoint_bytes=s['checkpoint_bytes'],baseline_memory_bytes=s['baseline_memory_bytes'],candidate_memory_bytes=s['candidate_memory_bytes'],screen_pass=s['finite_query_screen_pass']))
            diagnostics.append(dict(**base,construction_costs=s['construction_costs'],analytic_geometry=s['analytic_geometry'],evaluation_seconds=s['evaluation_seconds'],prediction_variance=s['prediction_variance_diagnostic'],averaged_prediction_DIAGNOSTIC_ONLY=s['averaged_prediction_diagnostic']))
            for b in s['rows']:
                rows.append(dict(**base,status=a['status'],mode=b['mode'],batch=b['batch'],**{k:b[k] for k in metrics}))
                assert len(b['per_class_mean'])==len(b['per_class_max'])==s['seen']
                for j in range(s['seen']):classes.append(dict(**base,mode=b['mode'],batch=b['batch'],class_index=j,group='old' if j<s['old'] else 'new',mean_abs_error=b['per_class_mean'][j],max_abs_error=b['per_class_max'][j]))
    passed=all(checks.get(k,False) for k in ('arithmetic','ISIC_scale','HK_scale')) and not p['failures'] and len(costs)==4
    total=sum(c['CPU_process_residence_seconds'] for c in costs)
    audit=dict(status='FINITE_SEQUENCE_FRESH_BATCH_SIGNAL' if passed else 'NO_COMPLETE_SEQUENCE_SIGNAL',checks=checks,scientific_source_commit=p['source_commit'],costs=costs,failures=p['failures'],CPU_residence_seconds=total,CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in costs),default=None),supervisor_wall_seconds=p['supervisor_wall_seconds'],prior_CPU_seconds=lock['prior_CPU_residence_seconds'],combined_CPU_seconds=lock['prior_CPU_residence_seconds']+total,new_GPU_seconds=0,historical_GPU_seconds=400036.74851310626,client_collection_residence_seconds=None,medical_evidence=False,classifier_training_tested=False,lossy_recompression_tested=False,uniform_certificate=False)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');(r/'SEQUENCE_DIAGNOSTICS.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    write_csv(r/'CPU_SUMMARY.csv',rows,['case','task','seen','old','status','mode','batch',*metrics]);write_csv(r/'CLASS_ERRORS.csv',classes,['case','task','seen','old','mode','batch','class_index','group','mean_abs_error','max_abs_error'])
    fields=['case','task','seen','old','old_memory_exact','old_draw_exact','old_fixed_query_error','ETF_error','checkpoint_exact','checkpoint_bytes','baseline_memory_bytes','candidate_memory_bytes','screen_pass'];write_csv(r/'INTERFACE_AUDIT.csv',interfaces,fields)
    report=['# 序列谱球面记忆 CPU结果','',audit['status'],str(checks),'','|规模|stage|方法|批次|mean|max|old mean|new mean|','|---|---:|---|---:|---:|---:|---:|---:|']
    for a in rows:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','task','mode','batch','mean_abs_error','max_abs_error','old_mean_abs_error','new_mean_abs_error'))+'|')
    report+=['',f"CPU独立驻留{total}秒；user{audit['CPU_user_seconds']} / system{audit['CPU_system_seconds']}；peak RSS{audit['peak_RSS_MiB']}MiB。父wall{audit['supervisor_wall_seconds']}另列，内部操作不相加。",f"前十三轮{lock['prior_CPU_residence_seconds']}秒，累计{audit['combined_CPU_seconds']}秒。失败{p['failures']}，missing不补0。",'','|case|退出码|驻留秒|user|system|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for c in costs:report.append('|'+ '|'.join(str(c[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','0新GPU/医学数据/val/test/正式更新。每类16d+16参数随seen增长，非固定全局容量。旧支持仅合成oracle评价，构造不访问。','固定输入ETF恒等不保证新类竞争或encoder更新；没有分类训练或有损重压缩。stress随stage变化，不等旧memory漂移。八批平均诊断不代替逐stage逐批门。','主候选block_spherical，Gaussian配对控制全部正负保留，不从CPU通过直接称医学贡献或GPU-ready。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');(r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(cases=len(p['cases']),costs=len(costs),rows=len(rows),class_rows=len(classes),stages=len(interfaces),checks=checks,missing_preserved=True,partial_completed_stages_preserved=True),indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
