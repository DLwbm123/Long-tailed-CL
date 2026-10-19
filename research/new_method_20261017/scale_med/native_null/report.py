"""Render only persisted aggregate and receipts; never read private features."""
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'aggregate.json').read_text());cost=json.loads((p/'COST_RECEIPT.json').read_text());co=json.loads((p/'COORDINATOR_EXIT.json').read_text())
assert r['heads']==24 and len(r['summaries'])==6 and sum(v['query_rows'] for v in r['fold_audit'])==2555
lines=['# 原生隐藏零空间诊断','','状态'+r['status']+'；必要门='+str(r['necessary_null_gate'])+'；method_success=false；'+r['next_decision']+'。','','|臂|类1组错误率%|类1图像错误率%|四类BA%|','|---|---:|---:|---:|']
for x in r['summaries']:lines.append(f"|{x['arm']}|{x['class1_group_error_pp']:.4f}|{x['class1_image_error_pp']:.4f}|{x['BA_pp']:.4f}|")
lines+=['','## 三个预设主对比：控制风险减NULL风险','','|控制|NULL收益pp|95%区间|三项调整区间|','|---|---:|---|---|']
for a,v in r['primary'].items():lines.append(f"|{a}|{v['null_benefit_pp']:.4f}|{v['CI95_pp']}|{v['family3_CI_pp']}|")
for name,v in [('数值投影检查',r['numerics']),('留出审计',r['fold_audit']),('正文与提取成本',r['costs']),('最终资源回执',cost),('协调回执',co)]:lines+=['','## '+name,'','```json',json.dumps(v,indent=2,ensure_ascii=False),'```']
lines+=['',f"每臂保守状态上界{r['planned_max_state_bytes']}bytes；固定1318+96，不挑层/维度/seed/alpha。",'', '本轮为InnerCIL启发机制诊断，不是新方法；旧永久验证/未来类/测试不拟合或评分。特权离线原训练支持访问不等于可部署退休实现。20行16已知组、未知患者来源；区间条件于交叉拟合模型，非新独立患者泛化认证。协调/入口/正文时间包含，不相加。所有新增投影/提取/缓存/求解均计入本轮，旧成本另保留。']
(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
