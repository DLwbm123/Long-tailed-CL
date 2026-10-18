"""Aggregate-only reporting; no private input or per-image outputs."""
import json,sys
from pathlib import Path

source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
r=json.loads(source.read_text());(out/'aggregate.json').write_text(json.dumps(r,indent=2,allow_nan=False))
lines=['# 历史采样误差传播诊断','',f"状态：{r['status']}；48/48单元；必要条件：{'通过' if r['gate'] else '失败'}。这不是新方法泛化成功。",'',
'固定原表示、支持划分及三点alpha；8次配对身份组bootstrap仅估计给定经验支持的采样敏感性。永久验证已历史使用，非盲测；患者隔离未知。没有读取测试。', '',
'|臂|扰动类|阶段|alpha|精确margin RMS|动态相对误差|动态/静态RMSE|验证BA均值±SD|','|---|---|---|---|---|---|---|---|']
for x in r['cells']:
    lines.append(f"|{x['arm']}|{x['source_class']}|{x['stage']}|{x['alpha']}|{x['exact_margin_RMS']:.6g}|{x['dynamic_relative_error']:.4f}|{x['dynamic_to_static_RMSE']:.4f}|{100*x['BA_mean']:.4f}±{100*x['BA_sd']:.4f}|")
lines+=['','## 成本与边界','',json.dumps(r['costs'],ensure_ascii=False),'',f"训练{r['training_rows']}行，永久验证{r['validation_rows']}行；组数{r['groups']}。",'',
'特权离线访问原支持元数据和冻结训练缓存；这不是可部署退休协议实现。冻结变换未重新拟合，所以没有测量表示学习本身的采样方差。身份组件不等同患者。来源分布差异尚未建立独立识别条件。', '',
'动态误差小只说明一阶扰动预测更接近精确重解，不说明降低了风险，也不说明某个新鲁棒分类器有效。强简单控制为出生时静态预测和精确重解；新方法仍需另行比较普通集成和匹配成本的广义ridge。', '',f"下一判定：{r['next_decision']}。不追加抽样、强度、alpha或预算。"]
(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
