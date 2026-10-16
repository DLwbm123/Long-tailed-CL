"""All-arm synthetic sequence report; retains domain-control costs and negative stages."""
import csv
import json
import sys
from pathlib import Path


def table(path,rows):
    fields=list(dict.fromkeys(k for a in rows for k in a))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    r=Path(sys.argv[1]);v=json.loads((r/'CPU_RESULTS.json').read_text());p=v['program'];lock=json.loads((r/'DESIGN_LOCK.json').read_text())
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    assert [a['id'] for a in p['cases']]==[a['id'] for a in lock['cases']]
    costs=p['costs'];assert len(costs)==sum(a['status']!='NOT_RUN_OVERALL_CAP' for a in p['cases'])
    unit=next((a for a in v['results'] if a['id']=='shared_zero_arithmetic'),None);results={a['id']:a for a in v['results']}
    outputs={k:[] for k in ('rows','class_rows','training','curvature','construction_costs','bound_diagnostics','stage_contributions')};resources=[]
    for cfg in lock['cases'][1:]:
        a=results.get(cfg['id'],{})
        if a.get('status')=='COMPLETED':
            stages=len(cfg['task_sizes']);seen=0;class_count=0
            for n in cfg['task_sizes']:seen+=n;class_count+=seen
            assert len(a['rows'])==4*stages and len(a['class_rows'])==4*class_count
            assert len(a['training'])==4*stages and len(a['bound_diagnostics'])==stages
            assert a['total_head_updates']==4*48*stages and a['all_training_sealed_before_evaluation']
            for k in outputs:outputs[k].extend(dict(case=cfg['id'],**row) for row in a[k])
            resources.append(dict(case=cfg['id'],**{k:a.get(k) for k in ('synthetic_sequence_signal_pass','candidate_memory_bytes','moment_memory_bytes','head_updates_per_arm','total_head_updates','baseline_compression_updates','candidate_compression_updates','CPU_pipeline_seconds','operation_seconds_including_import','candidate_old_support_reads_during_training')}))
        else:outputs['rows'].append(dict(case=cfg['id'],status=a.get('status','MISSING')))
    complete=len(costs)==4 and not p['failures'] and unit is not None and unit.get('status')=='PASS' and len(resources)==3
    signal=complete and all(results[c]['synthetic_sequence_signal_pass'] for c in ('ISIC_scale','HK_scale'))
    total=sum(c['CPU_process_residence_seconds'] for c in costs)
    audit=dict(status='SYNTHETIC_SEQUENCE_CHECKS_COMPLETE' if complete else 'INCOMPLETE_SYNTHETIC_SEQUENCE',synthetic_classification_signal_pass=signal,original_full_query_gate_pass=None,medical_evidence=False,GPU_ready=False,source_commit=p['source_commit'],unit=unit,costs=costs,failures=p['failures'],CPU_residence_seconds=total,CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in costs),default=None),supervisor_wall_seconds=p['supervisor_wall_seconds'],prior_CPU_seconds=lock['prior_CPU_residence_seconds'],combined_CPU_seconds=lock['prior_CPU_residence_seconds']+total,new_GPU_seconds=0,historical_GPU_seconds=400036.74851310626,client_collection_residence_seconds=None)
    (r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    names=dict(rows='CPU_SUMMARY',class_rows='CLASS_RECALL',training='TRAINING_COSTS',curvature='CURVATURE',construction_costs='CONSTRUCTION_COSTS',bound_diagnostics='BOUND_DIAGNOSTICS',stage_contributions='CONTRIBUTIONS')
    for k,name in names.items():table(r/(name+'.csv'),outputs[k])
    table(r/'RESOURCE_SUMMARY.csv',resources)
    report=['# 共同零点记忆：完整合成分类序列','',audit['status'],f'预声明合成分类信号：{signal}。不是医学/ViT/Core复现，也不改变旧完整查询族失败。','', '|规模|阶段|方法|Acc|BA|旧BA|新BA|','|---|---:|---|---:|---:|---:|---:|']
    for a in outputs['rows']:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','task','mode','accuracy','balanced_accuracy','old_balanced_accuracy','new_balanced_accuracy'))+'|')
    report+=['','|规模|阶段|candidate−moment BA|candidate−local oracle BA|local−wide oracle BA|全指标无损|存在严格收益|','|---|---:|---:|---:|---:|---|---|']
    for a in outputs['stage_contributions']:report.append('|'+ '|'.join(str(a.get(k)) for k in ('case','task','candidate_minus_moment_BA','candidate_minus_local_oracle_BA','local_minus_wide_oracle_BA','no_harm_to_moment','strict_gain_over_moment'))+'|')
    report+=['',f"CPU驻留{total}秒/user{audit['CPU_user_seconds']}/system{audit['CPU_system_seconds']}/peak{audit['peak_RSS_MiB']}MiB。父wall{audit['supervisor_wall_seconds']}另列，内部拟合/头训练/评价/operation含驻留不相加。",f"前17轮{lock['prior_CPU_residence_seconds']}秒，累计{audit['combined_CPU_seconds']}。失败{p['failures']}，missing不补0。",'','|case|退出码|驻留秒|user|system|RSS MiB|timeout|','|---|---:|---:|---:|---:|---:|---|']
    for c in costs:report.append('|'+ '|'.join(str(c[k]) for k in ('id','returncode','CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB','timeout'))+'|')
    report+=['','四臂同线性架构/48步/固定合成特征；候选和moment训练仅当前支持+旧memory，完整旧支持仅用于oracle对照及训练封存后的诊断。独立合成评价seed+10000固定，不访问医学val/test。','每stage不能被末stage覆盖；shared domain可行不等分类有效。无encoder更新或正式GPU预算证据。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');(r/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(cases=len(p['cases']),costs=len(costs),rows=len(outputs['rows']),class_rows=len(outputs['class_rows']),training_rows=len(outputs['training']),bound_rows=len(outputs['bound_diagnostics']),engineering_complete=complete,synthetic_signal_pass=signal,missing_preserved=True),indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
