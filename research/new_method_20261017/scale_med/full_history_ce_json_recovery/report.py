"""Render persisted aggregate tables, never private scores."""
import json,csv,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'aggregate.json').read_text())
lines=['# 封存预测的完整历史CE聚合恢复','',f"科学判断：`{r['decision']}`。通过臂：{r['passing_arms']}；method_success=false。",'', '本次只读取前轮封存头和18个方法×阶段预测，无训练、头前向、特征提取或重新选参。旧失败回执保留。内部测试已研究暴露，完整历史CE仅特权离线参照。','', '|方法|初始BA%|+类4 BA%|+类5 BA%|J%|','|---|---:|---:|---:|---:|']
for n,m in r['summary'].items():lines.append('|'+n+'|'+'|'.join(f"{m[str(s)]['BA']*100:.6f}" for s in range(3))+f"|{r['J_pp'][n]:.6f}|")
lines+=['','|CE减ridge|J差pp|95%区间|九项调整区间|','|---|---:|---|---|']
for a,c in r['comparisons'].items():
 for b,v in c['vs_ridge'].items():lines.append(f"|{a}−{b}|{v['J_gain_pp']:+.6f}|{v['CI95_pp']}|{v['CI9_adjusted_pp']}|")
lines+=['','## 逐类风险','','|方法|阶段|图像召回%按类顺序|组等权BA%|','|---|---:|---|---:|']
for n,m in r['summary'].items():
 for s,v in m.items():lines.append(f"|{n}|{s}|{[round(x*100,4) for x in v['recall']]}|{v['group_equal_BA']*100:.6f}|")
lines+=['','## 预设门','','```json',json.dumps(dict(comparisons=r['comparisons'],gates=r['gates']),ensure_ascii=False,indent=2),'```','','## 成本','','```json',(p/'BUDGET_VERIFICATION.json').read_text(),'```','正文/入口包含于协调，不内部相加；历史失败CPU缺失，累计总CPU仍未知。']
(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
with (p/'RISK_TABLE.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['method','stage','BA','group_equal_BA','recall_by_class','group_recall_by_class'])
 for n,m in r['summary'].items():
  for s,v in m.items():w.writerow([n,s,v['BA'],v['group_equal_BA'],json.dumps(v['recall']),json.dumps(v['group_recall'])])
with (p/'COMPARISONS.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['CE','ridge','J_gain_pp','CI95_lower','CI95_upper','CI9_lower','CI9_upper'])
 for a,c in r['comparisons'].items():
  for b,v in c['vs_ridge'].items():w.writerow([a,b,v['J_gain_pp'],*v['CI95_pp'],*v['CI9_adjusted_pp']])
