"""Render persisted aggregate diagnostics only."""
from pathlib import Path
import json,csv,sys
p=Path(sys.argv[1]);r=json.loads((p/'aggregate.json').read_text());lines=['# 类1阈值包络诊断结果','',f"终态：`{r['decision']}`；主门={r['primary_gate']}；method_success=false。",'', '全部U均为query标签知情的限定策略族经验包络，不是可部署收益；原划分、原头及预测未变。','', '|分数族|阶段|原类1召回%|U%|包络BA%|其他类宏代价pp|状态|','|---|---:|---:|---:|---:|---:|---|']
rows=[]
for n,ms in r['envelopes'].items():
 for st,v in ms.items():
  def fmt(x):return '未决' if x is None else f'{x:.6f}'
  lines.append(f"|{n}|{st}|{v['original_recall'][1]*100:.4f}|{fmt(None if v['U'] is None else 100*v['U'])}|{fmt(None if v['envelope_BA'] is None else 100*v['envelope_BA'])}|{fmt(v['other_macro_cost_pp'])}|{v['status']}|")
  rows.append(dict(name=n,stage=st,**v))
lines+=['','## 三项初始主比较','','|表示|U_CE−U_ridge pp|可行配对重复/10000|95%区间|三项调整区间|门|','|---|---:|---:|---|---|---|']
for a,v in r['primary'].items():lines.append(f"|{a}|{v['delta_U_pp']}|{v['feasible_paired_draws']} / {v['total_draws']}|{v['CI95_pp']}|{v['CI3_adjusted_pp']}|{v['gate']}|")
lines+=['','不可行重复不丢弃，不将条件于可行子集的区间当无条件置信区间。','', '## 完整逐类风险','','|分数族|阶段|原召回%|同风险参照召回%|包络召回%|','|---|---:|---|---|---|']
for n,ms in r['envelopes'].items():
 for st,v in ms.items():
  f=lambda a:None if a is None else [round(x*100,6) for x in a]
  lines.append(f"|{n}|{st}|{f(v['original_recall'])}|{f(v['reference_recall'])}|{f(v['envelope_recall'])}|")
lines+=['','## 预算','','```json',(p/'BUDGET_VERIFICATION.json').read_text(),'```','','正文/入口包含于协调墙钟，不相加；累计历史失败CPU缺失仍保留。源commit元数据占位符更正见CODE_PROVENANCE，短运行实时ps未捕获见VERIFICATION。']
(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
with (p/'ENVELOPES.csv').open('w') as fp:
 cols=['name','stage','status','U','original_BA','reference_BA','envelope_BA','other_macro_cost_pp','tail_gain_pp','can_reach_reference_tail','feasible_states','decision_states','original_recall','reference_recall','envelope_recall'];w=csv.DictWriter(fp,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
with (p/'PRIMARY.csv').open('w') as fp:
 w=csv.DictWriter(fp,fieldnames=['arm','delta_U_pp','feasible_paired_draws','total_draws','CI95_pp','CI3_adjusted_pp','gate']);w.writeheader();w.writerows([dict(arm=a,**v) for a,v in r['primary'].items()])
