"""Derive paired changes from public frozen-score diagnostic aggregates."""
import json,csv
from pathlib import Path
r=Path(__file__).parent/'results';x=json.loads((r/'metrics.json').read_text())['records'];index={(z['stage'],z['source'],z['arm'],z['subset']):z for z in x};contrasts=[]
for stage in range(6):
 for source in ['OOF','validation']:
  for arm in ['scale_joint','uniform_shrink','risk_limited_shrink']:
   for ref in ['cls768_ridge','scale_joint']:
    if arm==ref:continue
    a=index[stage,source,arm,'all'];b=index[stage,source,ref,'all'];contrasts.append(dict(stage=stage,source=source,arm=arm,reference=ref,PP_risk_delta=a['PP_risk']-b['PP_risk'],PP_accuracy_delta_pp=100*(a['PP_accuracy']-b['PP_accuracy']),BA_delta_pp=100*(a['BA']-b['BA']),image_balanced_risk_delta=a['image_balanced_risk']-b['image_balanced_risk']))
(r/'CONTRASTS.json').write_text(json.dumps(contrasts,indent=2)+'\n')
with (r/'contrasts.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(contrasts[0]));w.writeheader();w.writerows(contrasts)
v=json.loads((r/'VERIFICATION.json').read_text());c=json.loads((r/'costs.json').read_text());failure=json.loads((r/'INITIAL_FAILURE.json').read_text());seconds=c['wall_seconds']+v['wall_seconds']+failure['shell_wall_seconds']
(r/'CUMULATIVE_COSTS.json').write_text(json.dumps(dict(analysis_attempts=2,analysis_failed_attempts=1,analysis_main_seconds=c['wall_seconds'],analysis_verification_seconds=v['wall_seconds'],first_failed_attempt_shell_seconds=failure['shell_wall_seconds'],analysis_measured_seconds=seconds,prior_research_measured_seconds=4477.104266926193,cumulative_measured_seconds=4477.104266926193+seconds,new_encodes=0,new_fits=0,new_solver_calls=0,new_coefficients_selected=0,prior_research_attempts_unchanged=4,prior_scientific_solver_calls_unchanged=7,monitoring_status='PAUSED',unmeasured_overhead='some interpreter startup, transfer, report writing and publication'),indent=2)+'\n')
text=['# 冻结分数诊断数值表','', '| 来源 | 方法 | 类/患者等权平方风险 | 同权重正确率 (%) | 标准BA (%) | 类/图像等权平方风险 |','|---|---|---:|---:|---:|---:|']
for z in x:
 if z['stage']==5 and z['subset']=='all':text.append(f"| {z['source']} | {z['arm']} | {z['PP_risk']:.8f} | {100*z['PP_accuracy']:.4f} | {100*z['BA']:.4f} | {z['image_balanced_risk']:.8f} |")
text+=['','| 阶段 | 来源 | 受限相对CLS风险差 | 同权重正确率差 pp | BA差 pp |','|---|---|---:|---:|---:|']
for z in contrasts:
 if z['arm']=='risk_limited_shrink' and z['reference']=='cls768_ridge':text.append(f"| {z['stage']+1} | {z['source']} | {z['PP_risk_delta']:+.8f} | {z['PP_accuracy_delta_pp']:+.4f} | {z['BA_delta_pp']:+.4f} |")
text+=['','| 阶段 | 来源 | 统一相对尺度联合风险差 | 同权重正确率差 pp | BA差 pp |','|---|---|---:|---:|---:|']
for z in contrasts:
 if z['arm']=='uniform_shrink' and z['reference']=='scale_joint':text.append(f"| {z['stage']+1} | {z['source']} | {z['PP_risk_delta']:+.8f} | {z['PP_accuracy_delta_pp']:+.4f} | {z['BA_delta_pp']:+.4f} |")
(r/'TABLES.md').write_text('\n'.join(text)+'\n')
