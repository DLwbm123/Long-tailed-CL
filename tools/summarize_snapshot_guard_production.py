"""Scalar-only production CPU closeout, including independently measured failed attempts."""
import csv
import json
import subprocess
from pathlib import Path


def summarize(root):
    lock=json.loads((root/'PROTOCOL_LOCK.json').read_text());data=json.loads((root/'CPU_RESULTS.json').read_text());p=data['program']
    assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None and len(p['cases'])==len(data['results'])==len(lock['cases'])==8
    assert [v['id'] for v in p['cases']]==[v['id'] for v in lock['cases']]
    costs={v['id']:v for v in p['costs']};rows=[];all_pass=True
    for cfg,r in zip(lock['cases'],data['results']):
        c=costs.get(cfg['id'],{});passed=r.get('status')=='PASS';all_pass&=passed
        if passed:
            assert not r['test_accessed'] and all(r[k]==0 for k in ('GPU_processes','medical_images_read','medical_checkpoints_read','val_reads','formal_optimizer_updates'))
            if cfg['kind']=='large':
                assert r['frozen_math_certificate_status']=='CERTIFIED' and r['minimum_raw_slack']>=-2e-8
                assert r['stationarity']<=1e-8 and r['complementarity']<=1e-7 and abs(r['primal_dual_gap'])<=1e-7*(1+abs(r['objective_candidate']))
                assert r['all_constraints_n']==cfg['fit_n']*(cfg['classes']-1) and r['active_atom_cap'] is None
        rows.append(dict(id=cfg['id'],status=r.get('status','MISSING'),kind=cfg['kind'],support_n=cfg.get('fit_n'),classes=cfg.get('classes'),constraints=r.get('all_constraints_n'),positive_duals=r.get('positive_duals'),minimum_slack=r.get('minimum_raw_slack'),retained_gain=r.get('retained_gain'),identity=r.get('identity'),discriminative_delta=r.get('discriminative_delta'),checkpoint_bytes=r.get('checkpoint_bytes'),CPU_process_residence_seconds=c.get('CPU_process_residence_seconds'),CPU_user_seconds=c.get('CPU_user_seconds'),CPU_system_seconds=c.get('CPU_system_seconds'),peak_RSS_MiB=c.get('peak_RSS_MiB'),CPU_timeout=c.get('CPU_timeout')))
    cs=p['costs'];parent=2.190434727817774
    audit=dict(status='PASS_CPU_ONLY_NOT_GPU_READY' if all_pass else 'FAILED_OR_INCOMPLETE_CPU_ONLY',science_source_commit=p['source_commit'],reporting_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),current_CPU_children=len(cs),current_CPU_residence_seconds=sum(v['CPU_process_residence_seconds'] for v in cs),current_CPU_user_seconds=sum(v['CPU_user_seconds'] for v in cs),current_CPU_system_seconds=sum(v['CPU_system_seconds'] for v in cs),current_peak_RSS_MiB=max((v['peak_RSS_MiB'] for v in cs),default=None),supervisor_wall_seconds=p['supervisor_wall_seconds'],parent_failed_source='0c707f92424810d407587b39771a22d8e2dc991c',parent_failed_children=8,parent_failed_CPU_residence_seconds=parent,production_request_total_measured_CPU_residence_seconds=parent+sum(v['CPU_process_residence_seconds'] for v in cs),nested_pipeline_and_operation_in_residence=True,GPU_seconds=0,closed_GPU_history_seconds=391490.24710153416,real_GPU_preflights=0,formal_updates=0,GPU_ready=False,production_entry_connected=True,medical_images_read=0,medical_checkpoints_read=0,val_reads=0,test_accessed=False,failures=p['failures'],client_or_wrapper_CPU_residence_unmeasured=None)
    with (root/'CPU_PRODUCTION_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'CPU_PRODUCTION_CHECK.json').write_text(json.dumps(dict(status='PASS' if all_pass else 'FAILED_OR_INCOMPLETE',rows=rows,production_entry_connected=True,real_GPU_preflights=0,GPU_ready=False),indent=2)+'\n')
    (root/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (root/'CPU_REPORT_CHECK.json').write_text(json.dumps(dict(status='PASS_REPORT_INTEGRITY',cases=len(rows),all_successful_certificates_rechecked=True,missing_not_zero=True,production_CPU_all_pass=all_pass,no_medical_effect_claim=True),indent=2)+'\n')
    def fmt(v):return '未提供' if v is None else f'{v:.9g}' if isinstance(v,float) else str(v)
    lines=['# 快照守卫生产 CPU 检查完整报告','',f"科学源 `{p['source_commit']}`，当前八项CPU验收：{'全部通过' if all_pass else '存在失败/缺失，不能称通过'}。没有GPU启动或真实GPU预检。",'', '| 检查 | 状态 | 完整约束/正dual | 判别变化 | 收益保留 | 驻留秒 | 峰值MiB |','|---|---|---|---|---|---|---|']
    for row in rows:lines.append('| '+' | '.join([row['id'],row['status'],fmt(row['constraints'])+'/'+fmt(row['positive_duals']),fmt(row['discriminative_delta']),fmt(row['retained_gain']),fmt(row['CPU_process_residence_seconds']),fmt(row['peak_RSS_MiB'])])+' |')
    lines+=['',f"本轮独立CPU驻留{audit['current_CPU_residence_seconds']:.9f}秒；user {audit['current_CPU_user_seconds']:.9f}/system {audit['current_CPU_system_seconds']:.9f}；峰值{fmt(audit['current_peak_RSS_MiB'])}MiB。父wall {audit['supervisor_wall_seconds']:.9f}秒另列，pipeline/operation均包含于驻留，不相加。首轮八次解析失败2.190434728秒已77bd9e29完整交付，本生产请求两轮累计实测驻留{audit['production_request_total_measured_CPU_residence_seconds']:.9f}秒。无GPU费用，旧历史391490.24710153416秒不重计。",'', '关闭Core/LT和PMT的精确回归、全stage训练/guard-aware评分、失效/非法scope/预算/早val及guard schema结果见CPU_RESULTS。模拟eval仅合成数据，不是医学val。真实ViT两张合成图CPU零optimizer反传与高维全部fit/最大旧bank是分别记录的检查，不能合称四条真实GPU预检。', '', '预算保持新四轨迹1288更新；Core/LT及两背景PMT复用，不训练对照。54个守卫事件不加图像提取，但增加CPU原生逆、主机逻辑输入量和私有证书；未来GPU worker驻留已包含CPU等待，配对控制推理不为0。原canonical/student/teacher曝光及存储/推理预算见BUDGET_SPEC。', '', '守卫权威double头与旧存储float对照精度差异必须报告，same-state OFF/proposal不替代独立控制贡献。只保护固定encoder快照fit margin；encoder更新即失效。不保证val无损、旧旧风险识别或医学净纠错。', '', '当前生产入口已接线，CPU通过也不自动GPU-ready：未运行GPU、未读医学图像或checkpoint、未读val/test、未有新医学结果；只有明确后续GPU授权与全真实零更新预检后才能启动。所有缺失保持null，不补0。']
    if p['failures']:
        assert (root/'CPU_FAILURES_ANON.json').exists(),'Inspect private failure and sanitize before reporting'
        lines+=['','失败范围及去标识原因：',json.dumps(json.loads((root/'CPU_FAILURES_ANON.json').read_text()),ensure_ascii=False)]
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit

if __name__=='__main__':
    import sys
    print(json.dumps(summarize(Path(sys.argv[1])),ensure_ascii=False))
