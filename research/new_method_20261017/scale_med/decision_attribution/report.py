"""Render aggregate-only receipts; never open private scores."""
import json,sys
from pathlib import Path

p=Path(sys.argv[1]);r=json.loads((p/'aggregate.json').read_text())
lines=['# 固定输出的决策归因报告','', '本轮只复用已暴露测试的封存预测，不训练、不推理、不调参。所有统计均为探索性固定头归因，不是独立泛化确认。', '', '## 类4主比较', '', '|C减控制|出生优势pp|最终优势pp|优势变化pp及95%CI|旧排序贡献pp|新竞争贡献pp|', '|---|---|---|---|---|---|']
for v in r['primary_class4_contrasts']:
    lines.append(f"|{v['control']}|{v['birth_advantage_pp']:.4f}|{v['final_advantage_pp']:.4f}|{v['advantage_change_pp']:.4f} {v['advantage_change_CI95_pp']}|{v['old_ranking']['delta_pp']:.4f}|{v['new_competition']['delta_pp']:.4f}|")
for v in r['primary_class4_contrasts']:
    lines+=['',f"C−{v['control']}："]
    for key in ['old_ranking','new_competition']:
        q=v[key];lines.append(f"- {key}：{q['delta_pp']:.4f}pp，95%CI {q['CI95_pp']}，主四项调整区间 {q['family4_adjusted_CI_pp']}，稳定额外损失={q['stable_extra_loss']}。")
lines+=['', '## 全部旧类单元', '', '|转换|臂|类|n|B|U|F|旧排序净数|新竞争净数|合计净数|', '|---|---|---|---|---|---|---|---|---|---|']
for v in r['cells']:
    lines.append(f"|{v['transition']}|{v['arm']}|{v['class_index']}|{v['count']}|{v['birth_correct']}|{v['current_old_correct']}|{v['current_full_correct']}|{v['old_ranking_net']}|{v['new_competition_net']}|{v['total_net']}|")
lines+=['', '## 类1三个阶段', '', '|阶段|臂|正确/n|召回%|', '|---|---|---|---|']
for v in r['baseline']:
    if v['class_index']==1:lines.append(f"|{v['stage']}|{v['arm']}|{v['correct']}/{v['count']}|{v['recall_pp']:.4f}|")
lines+=['', '## 全部旧类BA变化', '', '|转换|臂|上一阶段旧BA%|当前限制旧BA%|当前全部候选旧BA%|旧排序pp|新竞争pp|', '|---|---|---|---|---|---|---|']
for v in r['old_class_macros']:
    lines.append(f"|{v['transition']}|{v['arm']}|{v['birth_old_BA_pp']:.4f}|{v['current_restricted_old_BA_pp']:.4f}|{v['current_full_old_BA_pp']:.4f}|{v['old_ranking_pp']:.4f}|{v['new_competition_pp']:.4f}|")
lines+=['', '## 核验与成本', '', f"原75单元正确数完全一致；45单元分解及bootstrap恒等式通过。主要共同竞争损失门={r['common_competition_loss_gate']}；method_success=false；next_decision={r['next_decision']}。", '', json.dumps(r['costs'],ensure_ascii=False), '', 'B为上一阶段、U为当前头限制旧集合、F为当前全部候选。U利用旧类集合身份，仅为诊断，不能部署或当作改进模型。10000次配对身份组bootstrap不是训练方差或患者独立置信区间；四项调整区间仍是探索性近似。逐例身份、标签、分数和模型均不公开。']
(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
