"""Aggregate-only frozen support-completion report."""
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'aggregate.json').read_text());cost=json.loads((p/'COST_RECEIPT.json').read_text());co=json.loads((p/'COORDINATOR_EXIT.json').read_text())
lines=['# 等图像预算支持补全诊断','',f"状态{r['status']}，必要门={r['necessary_gate']}，{r['next_decision']}；method_success=false。",'', '## 两个预设G主对比','', '|对比|收益pp|95%区间|两项调整区间|','|---|---:|---|---|']
for k,v in r['primary'].items():lines.append(f"|{k}|{v['benefit_pp']:.4f}|{v['CI95_pp']}|{v['family2_CI_pp']}|")
lines+=['','## 全部三臂六条件','', '|臂|条件|类1组错误率%|类1图像错误率%|四类BA%|其余类宏召回%|','|---|---|---:|---:|---:|---:|']
for v in r['summaries']:lines.append(f"|{v['arm']}|{v['condition']}|{v['class1_group_error_pp']:.4f}|{v['class1_image_error_pp']:.4f}|{v['BA_pp']:.4f}|{v['other_class_macro_recall_pp']:.4f}|")
lines+=['','## P/C描述性对比','',json.dumps({k:v for k,v in r['effects'].items() if k!='G'},ensure_ascii=False,indent=2),'', '## 选组及留出审计','',json.dumps(r['fold_audit'],ensure_ascii=False,indent=2),'', '## 操纵与符号一致性','',json.dumps({k:r[k] for k in ['coverage_manipulation_pass','mean_near_minus_far_radius_ratio','head_sign_consistency']},ensure_ascii=False),'', '## 成本','',json.dumps(cost,indent=2),json.dumps(co,indent=2),'', 'BASE归一化/变换固定；72头先封存再query。全部选择仅outertrain，新增组事先已知类1属于离线诊断。每类质量1，独立组数效应不等于已分离所有有效样本量因素；全局距离覆盖不等于患者/临床覆盖。16已知组的条件区间不构成独立患者泛化保证。未按结果改选组、距离、seed、alpha、维度、层或pooling。ALL仅参照非性能上界。协调时间包含入口/正文，不相加。']
(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
