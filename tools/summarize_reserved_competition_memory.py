"""Render anonymous frozen factorial results; reporting only, never scientific execution."""
import json
import sys
from pathlib import Path


def save(root,name,value):
    (root/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(root):
    lock=json.loads((root/'DESIGN_LOCK.json').read_text())
    data=json.loads((root/'CPU_RESULTS.json').read_text());program=data['program'];results=data['results']
    by={v['id']:v for v in results}; unit=by.get(lock['cases'][0]['id'],{})
    expected=[c for c in lock['cases'] if c['kind']=='scale']
    complete=unit.get('status')=='PASS' and all(by.get(c['id'],{}).get('status')=='COMPLETED' for c in expected)
    collected={}
    for name,key in [('CPU_SUMMARY','rows'),('CLASS_RECALL','class_rows'),('PAIRED_PREDICTIONS','paired_predictions'),('TRAINING_COSTS','training'),('CONSTRUCTION_COSTS','construction'),('ASSIGNMENT_COSTS','assignment_costs'),('MEMORY_DIAGNOSTICS','diagnostics'),('CONTRIBUTIONS','contributions')]:
        rows=[dict(case=v['id'],**row) for v in results for row in v.get(key,[])]
        save(root,name,rows);collected[name]=rows
    if complete:
        assert len(collected['CPU_SUMMARY'])==48 and len(collected['CLASS_RECALL'])==540
        assert len(collected['PAIRED_PREDICTIONS'])==180 and len(collected['CONTRIBUTIONS'])==12
        assert len(collected['CONSTRUCTION_COSTS'])==36 and len(collected['TRAINING_COSTS'])==48
        for v in results:
            assert not v.get('GPU_initialized',False) and not v.get('test_accessed',False) and not v.get('val_accessed',False)
    total=sum(c['CPU_process_residence_seconds'] for c in program['costs'])
    signal=complete and all(by[c]['joint_mechanism_signal_pass'] for c in ('ISIC_scale','HK_scale'))
    audit=dict(status='SYNTHETIC_FACTORIAL_COMPLETE' if complete else 'INCOMPLETE',joint_mechanism_signal_pass=bool(signal),
        source_commit=program['source_commit'],CPU_residence_seconds=total,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],
        cumulative_CPU_residence_seconds=lock['prior_CPU_residence_seconds']+total,
        CPU_user_seconds=sum(c['CPU_user_seconds'] for c in program['costs']),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in program['costs']),
        peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in program['costs']),default=None),parent_wall_seconds_separate=program.get('supervisor_wall_seconds'),
        client_collector_residence_seconds=None,new_GPU_seconds=0,historical_GPU_seconds_unchanged=lock['historical_GPU_seconds'],
        failures=program['failures'],costs=program['costs'],GPU_ready=False,medical_effect_established=False,
        original_full_query_gate=None,internal_timings_included_not_added=True)
    save(root,'AUDIT',audit)
    save(root,'CPU_REPORT_CHECK',dict(report_consistency_pass=True,scientific_cases_complete=complete,unit_pass=unit.get('status')=='PASS',counts={k:len(v) for k,v in collected.items()},missing_filled_with_zero=False))
    lines=['# 稳定接口、预留类别竞争与判别记忆：合成 CPU 结果','',
           '状态：'+audit['status']+'；冻结联合信号门：'+str(bool(signal))+'。这是新同架构 2×2 合成验证，不是医学 Core/LT 或 GPU-ready 结果。','',
           '|规模|stage|方法|Acc %|BA %|旧 BA %|新 BA %|','|---|---:|---|---:|---:|---:|---:|']
    def pct(v):return 'null' if v is None else f'{100*v:.6f}'
    for r in collected['CPU_SUMMARY']:
        lines.append('|'+ '|'.join([r['case'],str(r['task']),r['mode'],pct(r['accuracy']),pct(r['balanced_accuracy']),pct(r['old_balanced_accuracy']),pct(r['new_balanced_accuracy'])])+'|')
    lines+=['','全部类别见 CLASS_RECALL；各组救回/破坏/错换错和错误流向见 PAIRED_PREDICTIONS；分配、记忆简单效应及交互见 CONTRIBUTIONS；同状态记忆误差见 MEMORY_DIAGNOSTICS。没有遗漏的失败补零。','',
            f'独立子进程 CPU 驻留 {total:.9f} 秒；加此前一次累计 {audit["cumulative_CPU_residence_seconds"]:.9f} 秒。父 wall {program.get("supervisor_wall_seconds")} 秒另列；峰值 RSS {audit["peak_RSS_MiB"]} MiB。内部训练、压缩、分配、评价含在驻留内不再相加。新增 GPU 0，历史 GPU 不变。','',
            '失败条目：'+json.dumps(program['failures'],ensure_ascii=False)+'。具体匿名失败说明见 FAILURES_ANON（若有）。','',
            '固定输入空间与旧 slot/记忆不变不代表适配器后特征或分类不变。有限 sliced-W1 和两快照梯度匹配没有任意未来查询保证。固定接口与教师保持为共同条件，本轮不单独证明其贡献。无生产 checkpoint/真实 encoder/医学数据验证。所有系数与数据事前冻结，不据结果改门或追加预算。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main(Path(sys.argv[1]))
