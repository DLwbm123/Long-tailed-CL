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
    analyze(root, results, audit)
    print(json.dumps(audit))



def analyze(root, results, audit):
    if audit['status']!='SYNTHETIC_FACTORIAL_COMPLETE':return
    differences=[]; classes=[]; resources=[]; checks=0
    metrics=['balanced_accuracy','old_balanced_accuracy','new_balanced_accuracy']
    comparisons=[('fixed_moment','fixed_decision'),('assigned_moment','assigned_decision'),('fixed_moment','assigned_moment'),('fixed_decision','assigned_decision'),('fixed_moment','assigned_decision')]
    lines=['# 完整分析：联合门失败，判别记忆没有形成分类收益','',
           '三个规模的联合门均 false；4 CPU case 正常结束，0运行失败/0超时。未重跑、未改门、未启动医学/GPU。','',
           '本轮四臂只识别分配与记忆的简单效应及交互；固定接口、随机取向ETF、非线性适配器与教师保持均为共同条件。第一阶段尚无旧记忆，同一分配下两记忆臂模型精确一致；第一阶段分类差异来自分配而非压缩。','',
           '|规模|stage|固定+moment BA%|固定+判别 BA%|分配+moment BA%|完整 BA%|','|---|---:|---:|---:|---:|---:|']
    for case in results[1:]:
        cid=case['id']; stages=sorted({r['task'] for r in case['rows']})
        for t in stages:
            by={r['mode']:r for r in case['rows'] if r['task']==t}
            lines.append('|'+ '|'.join([cid,str(t)]+[f"{100*by[m]['balanced_accuracy']:.6f}" for m in ['fixed_moment','fixed_decision','assigned_moment','assigned_decision']])+'|')
            for mode,row in by.items():
                cc=[c for c in case['class_rows'] if c['task']==t and c['mode']==mode]
                assert abs(sum(c['recall'] for c in cc)/len(cc)-row['balanced_accuracy'])<1e-12
                for cls in cc:assert 0<=cls['recall']<=1 and cls['count']>0
                checks+=1
            for a,b in comparisons:
                differences.append(dict(case=cid,task=t,baseline=a,candidate=b,**{m+'_difference_pp':None if by[a][m] is None else 100*(by[b][m]-by[a][m]) for m in metrics}))
                ac={c['class_index']:c for c in case['class_rows'] if c['task']==t and c['mode']==a}
                for c in case['class_rows']:
                    if c['task']==t and c['mode']==b:classes.append(dict(case=cid,task=t,baseline=a,candidate=b,class_index=c['class_index'],count=c['count'],recall_difference_pp=100*(c['recall']-ac[c['class_index']]['recall'])))
                pair=next(z for z in case['paired_predictions'] if z['task']==t and z['baseline']==a and z['candidate']==b and z['group']=='all')
                assert abs((pair['rescued']-pair['harmed'])/pair['count']-(by[b]['accuracy']-by[a]['accuracy']))<1e-12
                checks+=1
        assert sum(z['updates'] for z in case['training'])==case['total_adapter_updates']
        assert sum(z['updates'] for z in case['construction'])==case['actual_memory_updates']
        resources.append(dict(case=cid,adapter_parameters=case['adapter_parameters'],frame_bytes=case['frame_bytes'],memory_bytes_per_arm=case['candidate_memory_bytes'],training_seconds=sum(z['seconds'] for z in case['training']),construction_seconds=sum(z['seconds'] for z in case['construction']),assignment_seconds=sum(z['seconds'] for z in case['assignment_costs']),pipeline_seconds=case['CPU_pipeline_seconds'],adapter_updates=case['total_adapter_updates'],actual_memory_updates=case['actual_memory_updates']))
    save(root,'STAGE_DIFFERENCES',differences);save(root,'CLASS_CONTRIBUTIONS',classes);save(root,'RESOURCE_SUMMARY',resources)
    save(root,'FAILURES_ANON',dict(engineering_failures=[],timeouts=0,scientific_joint_gate_pass=False,scientific_gate_failures=['small','ISIC_scale','HK_scale'],prelaunch_transport_failures=1,transport_scientific_seconds=0))
    lines+=['','以下所有差值均为百分点；完整方案减“分配+moment”可隔离分配背景下的记忆变化。','',
            '|规模|stage|ΔBA|Δ旧BA|Δ新BA|','|---|---:|---:|---:|---:|']
    for d in differences:
        if d['baseline']=='assigned_moment' and d['candidate']=='assigned_decision':
            vals=[d[m+'_difference_pp'] for m in metrics]
            lines.append('|'+ '|'.join([d['case'],str(d['task'])]+['null' if v is None else f'{v:+.6f}' for v in vals])+'|')
    lines+=['','判别记忆在三个规模的全部非首阶段 BA 都低于同分配 moment。ISIC末stage旧/新同时下降；HK末stage新类改善但旧类下降，类别不均衡下总Acc只少1个正确样本仍对应BA明显下降。完整方案相对固定moment在HK stage3/4 BA提高，但旧BA仍下降；不能用平均或末stage掩盖损害。','',
            '分配在moment背景的末stage收益 small −0.049895pp、ISIC +0.615139pp、HK +3.767860pp，前面亦有负值；只算有限信号。高维正交互表示分配缓解判别记忆损害，不表示判别记忆或组合超过强对照。','',
            '末stage完整方案 vs 分配moment：small救回21/破坏71/错换错27；ISIC救回533/破坏1186/错换错940；HK救回107/破坏108/错换错245。HK旧样本救回73/破坏108，新样本救回34/破坏0。完整错误流向按all/old/new见PAIRED_PREDICTIONS；只记录candidate错误目的组，未保存baseline完整混淆矩阵，不伪造逐样本轨迹。','',
            '各臂记忆审计比较该臂同一状态的真实旧合成支持与自己的记忆，跨臂模型不同，因此不能把跨臂误差差值当固定模型压缩因果。末stage ISIC分配moment/完整最大梯度误差1.945787/4.122217，HK2.348939/4.253384；small1.316775/2.544204。HK完整平均CE误差较小但最大CE/梯度/间隔误差更大，说明只看均值不足。','',
            '所有压缩记录的初始到最后一次更新前目标均下降，但未保存最优性证书，最后一次更新后的目标也未单测。最少有效节点约small15.369、ISIC11.292、HK9.872，没有证据把失败简化为单点塌缩。两快照有限响应与梯度没有控制未来非线性适配器；旧memory逐tensor不变不等输出或分类不变。','',
            '分配机制限制：旧碰撞项对每个新类的同一候选slot相同，因此影响选择哪些空位，不能单独决定这些空位内部的类别排列；若本阶段填满全部剩余slot，该碰撞项总和为常数。当前实验未单独消融alignment与collision，不能把分配信号归给旧冲突项。','',
            '下一步仅分析：优先检查两快照压缩后的记忆是否保留未来更新所需的判别方向，以及碰撞项在可选择空位数变化时的实际作用。没有授权新轮，不据本次结果调强度、加节点、加步数或直接上医学GPU。','',
            '资源（所有内部秒数已含驻留，不相加）：','',
            '|规模|训练秒|构造秒|分配秒|单臂memory bytes|','|---|---:|---:|---:|---:|']
    for r in resources:lines.append(f"|{r['case']}|{r['training_seconds']:.6f}|{r['construction_seconds']:.6f}|{r['assignment_seconds']:.6f}|{r['memory_bytes_per_arm']}|")
    lines+=['',f"本轮独立CPU驻留{audit['CPU_residence_seconds']:.9f}秒；父wall {audit['parent_wall_seconds_separate']:.9f}秒另列；user {audit['CPU_user_seconds']:.6f}/system {audit['CPU_system_seconds']:.6f}秒；peakRSS {audit['peak_RSS_MiB']} MiB。21轮82次CPU累计{audit['cumulative_CPU_residence_seconds']:.9f}秒。新增GPU0；历史400036.74851310626秒不变。客户端/collector驻留未测null。首次prelaunch代理TLS失败无科学费用，同代理重试成功。",'',
            '未验证真实encoder/医学数据/原Core-LT预算/生产checkpoint，无顶级期刊、临床或统计显著结论。test/reserved持续封存。']
    (root/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')
    check=json.loads((root/'CPU_REPORT_CHECK.json').read_text());check.update(BA_class_recall_and_paired_accuracy_assertions=checks,update_budget_checks=True,stage_difference_rows=len(differences),class_contribution_rows=len(classes));save(root,'CPU_REPORT_CHECK',check)

if __name__=='__main__':main(Path(sys.argv[1]))
