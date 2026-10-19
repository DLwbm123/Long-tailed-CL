"""Aggregate-only report for the fixed birth-geometry diagnostic."""
import json,sys
from pathlib import Path

p=Path(sys.argv[1]);r=json.loads((p/'aggregate.json').read_text())
lines=['# 出生几何的严格组外诊断','',f"执行{r['status']}，必要几何风险门={r['necessary_geometry_risk_gate']}；method_success=false，下一判定{r['next_decision']}。",'',
'仅原初始永久训练支持，四折组外预测；原永久验证与测试不参与。全部变换和头从各折训练组重估，不复用历史已拟合状态。类1仍仅20行/16已知身份组，不等于新增患者独立证据。','',
'## 六个预设主对比','', '|对比|类1组等权错误率增加pp|95%区间|六项调整区间|','|---|---|---|---|']
for name,v in r['primary'].items():lines.append(f"|{name}|{v['error_increase_pp']:.4f}|{v['CI95_pp']}|{v['family6_adjusted_CI_pp']}|")
lines+=['','## 基础强控制','', '|条件|类1组等权错误率%|类1图像错误率%|类1margin均值|四类BA%|','|---|---|---|---|---|']
for v in r['summaries']:
    if '/' not in v['condition']:lines.append(f"|{v['condition']}|{v['class1_group_error_pp']:.4f}|{v['class1_image_error_pp']:.4f}|{v['class1_margin_mean']:.6f}|{v['BA_image_pp']:.4f}|")
lines+=['','## 固定四重复全量条件','', '|条件|类1组错误率%|类1图像错误率%|类1margin均值|四类BA%|','|---|---|---|---|---|']
for v in r['summaries']:
    if '/' in v['condition']:lines.append(f"|{v['condition']}|{v['class1_group_error_pp']:.4f}|{v['class1_image_error_pp']:.4f}|{v['class1_margin_mean']:.6f}|{v['BA_image_pp']:.4f}|")
lines+=['','## 坐标删减与随机/PCA控制','',json.dumps(r['controls'],indent=2,ensure_ascii=False),'','## 非加性交互','',json.dumps(r['normalization_geometry_interaction_pp'],ensure_ascii=False),'',
'## 留出审计与成本','',json.dumps(r['fold_audit'],indent=2,ensure_ascii=False),'',json.dumps(r['fit_counts'],ensure_ascii=False),'',json.dumps(r['costs'],ensure_ascii=False),'',
'四重复是预设扰动而非可选择训练seed；区间为给定交叉拟合模型的条件性近似，不是完整学习流程或患者总体置信区间。NORM/GEOM/FULL是操作定义的分阶段反事实，不能忽略交互或声称唯一因果分摊。未按指标改alpha、rank、fold、重复或预算。模型/身份/逐例预测/特征/图像不公开。']
(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
