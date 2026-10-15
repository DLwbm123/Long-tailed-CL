"""Public scalar-only report for the prospective joint full-fit CPU supplement."""
import csv,json,subprocess
from pathlib import Path


def fmt(x):return 'missing' if x is None else f'{x:.8g}' if isinstance(x,float) else str(x)


def summarize(path):
    root=Path(path);data=json.loads((root/'CPU_RESULTS.json').read_text());p=data['program']
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
    lock=json.loads((root/'DESIGN_LOCK.json').read_text());assert len(p['cases'])==4
    cfgs={c['id']:c for c in lock['cases']};rows=[]
    for record,cost in zip(data['results'],p['costs']):
        cid=record.get('case',{}).get('id',record.get('id'));assert cid==cost['id'] and cid in cfgs
        cfg=cfgs[cid];base=record.get('base',{})
        rows.append(dict(id=cid,N=cfg['fit_n'],K=cfg['classes'],status=record['status'],
            base_active=base.get('positive_duals'),minimum_slack=base.get('minimum_slack'),
            minus_status=record.get('perturbed',[{},{}])[0].get('status'),
            plus_status=record.get('perturbed',[{},{}])[1].get('status'),
            finite_difference=record.get('finite_difference'),correct_implicit_derivative=record.get('correct_implicit_derivative'),
            full_refresh_error=record.get('full_refresh_error'),missing_reference_error=record.get('missing_reference_derivative_error'),
            naive_error=record.get('naive_error'),support_correction=record.get('support_derivative_correction'),
            reference_correction=record.get('reference_derivative_correction'),tangent_adjoint_error=record.get('tangent_adjoint_error'),
            CPU_process_residence_seconds=cost['CPU_process_residence_seconds'],CPU_user_seconds=cost['CPU_user_seconds'],
            CPU_system_seconds=cost['CPU_system_seconds'],peak_RSS_MiB=cost['peak_RSS_MiB']))
    assert len(rows)==4 and len({r['id'] for r in rows})==4
    with (root/'JOINT_GRADIENTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    old=json.loads(Path('docs/full_fit_guard_design_20261015/AUDIT.json').read_text())
    costs=p['costs'];passed=all(r['status']=='PASS' for r in rows)
    audit=dict(status='CPU_JOINT_GRADIENT_PASS_NOT_GPU_READY' if passed else 'CPU_JOINT_GRADIENT_INCOMPLETE_NOT_GPU_READY',
        design_source_commit=p['source_commit'],reporting_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cases=4,passed_cases=sum(r['status']=='PASS' for r in rows),all_positive_and_negative_cases_kept=True,
        CPU_process_residence_seconds=sum(v['CPU_process_residence_seconds'] for v in costs),
        CPU_user_seconds=sum(v['CPU_user_seconds'] for v in costs),CPU_system_seconds=sum(v['CPU_system_seconds'] for v in costs),
        peak_RSS_MiB=max(v['peak_RSS_MiB'] for v in costs),supervisor_wall_seconds=p['supervisor_wall_seconds'],
        prior_full_fit_CPU_process_residence_seconds=old['CPU_process_residence_seconds'],
        combined_19_plus_4_residence_seconds=old['CPU_process_residence_seconds']+sum(v['CPU_process_residence_seconds'] for v in costs),
        pipeline_operation_in_process_residence=True,user_system_not_added_to_wall=True,
        failures=p['failures'],ISIC_T3_cap_still_unresolved=True,
        full_actual_ViT_encoder_refresh_checked=False,a_R_old_previous_refresh_checked=False,
        formal_experiment=False,GPU_processes=0,new_GPU_seconds=0.,medical_images_read=0,medical_checkpoints_read=0,
        val_reads=0,test_accessed=False,optimizer_updates=0,production_integrated=False,GPU_ready=False,
        closed_historical_GPU_seconds=391490.24710153416)
    (root/'AUDIT.json').write_text(json.dumps(audit,indent=2)+chr(10))
    lines=['# 全fit支持移动和参考头刷新的联合CPU检查','',
        '**独立CPU方向导数检查完成，仍不启动GPU。** 这是首批19项全fit CPU结果的补充，不是新医学实验。完整native batch64目标、全部fit支持和OFF参考重解共同随theta变化；a/R0/R/old/previous在块内固定。所有特征合成。','',
        f'冻结源 `{p["source_commit"]}` 已在CPU前公开DESIGN并核实HTTP200/NAS source；同状态base活跃数及三项目标与已公开首批结果在1e-8内一致。','',
        '| 背景 | N/K | 状态 | 活跃 | naive误差 | 漏参考刷新误差 | 完整刷新误差 | CPU驻留秒 | 峰值MiB |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join([r['id'],f'{r["N"]}/{r["K"]}',r['status'],fmt(r['base_active']),fmt(r['naive_error']),fmt(r['missing_reference_error']),fmt(r['full_refresh_error']),fmt(r['CPU_process_residence_seconds']),fmt(r['peak_RSS_MiB'])])+' |')
    lines+=['',f'通过{audit["passed_cases"]}/4。全部theta=0/±1e-5的可行性、逐atom标量、原目标值、导数分项与cost见CPU_RESULTS.json、JOINT_GRADIENTS.csv；missing不补0，cap或失败不当identity。','',
        '完整解析方向导数=.5<W,Hdot W>−<Bdot,W>−Σλ xdot_i^T(Δw−ΔWref)+<Z,Bdot0−Hdot0 Wref>，Z=H0^-1 A*λ；reference切向量与adjoint结果另行互检。全fit点作同一正交旋转，保持单位范数，batch64 indexed子集同步变动；不是只更新batch支持。有限差分每次重新解完整guard及OFF参考，未对PCG迭代图反传。','',
        '## 资源与完整负结果','',
        f'本补充4子进程CPU驻留{audit["CPU_process_residence_seconds"]:.6f}秒，user{audit["CPU_user_seconds"]:.6f}/system{audit["CPU_system_seconds"]:.6f}秒，峰值{audit["peak_RSS_MiB"]:.3f}MiB；父wall{p["supervisor_wall_seconds"]:.6f}秒。前一19项707.928139761秒，独立23次合计{audit["combined_19_plus_4_residence_seconds"]:.6f}秒。pipeline/operation已包含，不再相加；多线程user/system与wall分列。','',
        '首批最大ISIC T3两背景在64atom仍违约约.0026，未解结果继续保留；本次选已有全点可行的满rank阶段和最大旧bank阶段检验联合导数，不能据4例PASS宣布所有阶段可解。未放宽约束/加cap/调参数/用val挑类。','',
        '## 限制与决定','',
        '这验证一条前瞻固定的全支持正交旋转方向，不是任意真实ViT参数方向的全Jacobian证明。a/R/previous/旧bank若也刷新，当前公式覆盖范围改变。没有测真实encoder、图像加载、实际模型训练回放、GPU时间或显存，也没有医学净纠错证据。首批12个非identity可行例的合成分类均0救回0破坏，接近100%原二次目标保留不等于性能贡献。','',
        '生产关键仍是全fit支持刷新时序与预算：每一步用新encoder重提全部fit才对应当前约束语义；若只缓存epoch特征，不能继续声称保护所有当前encoder下的fit点。ISIC T3 N9434/batch64约147.4个batch的全量前向量级，不能把本次廉价合成旋转当真实刷新成本。原预算不允许无说明追加该曝光；也不偷偷降为minibatch保护。','',
        '因此本次补齐CPU联合导数核对后，仍维持NOT_GPU_READY：最大阶段求解/预算兼容的支持刷新/生产封存与实际零更新预检均待解决。后续先设计这些问题，不继续调强度或机械开医学轮。test/reserved封存、val未读、新GPU0秒，历史闭合391490.24710153416秒不变。']
    (root/'REPORT_ZH.md').write_text(chr(10).join(lines)+chr(10))
    return audit


if __name__=='__main__':
    import sys
    print(json.dumps(summarize(sys.argv[1]),indent=2))
