"""Aggregate table exporter. Keep interpretation in the separate REPORT_ZH.md."""
import csv,json
from pathlib import Path
p=Path(__file__).parent;d=p/'results';cfg=json.loads((p/'config.json').read_text());results=json.loads((d/'results.json').read_text());diag=json.loads((d/'FROZEN_DIRECTION_AUDIT.json').read_text());harm=json.loads((d/'WEIGHTED_HARM.json').read_text());names={'cls768_ridge':'CLS ridge','scale_joint':'尺度联合','df_matched_joint':'df联合','plain_correction':'完整普通修正','utility_correction':'原保护','cls_plain_correction':'CLS普通修正','mass_shuffled_correction':'质量守恒打散','score_subspace':'原头分数子空间','pca_subspace':'同维PCA子空间'}
with (d/'summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['scope','arm','BA_percent','PP_accuracy_percent','head_BA_percent','middle_BA_percent','tail_BA_percent','correct_images'])
 for row in results:
  for a,m in row['metrics'].items():w.writerow([row['scope'],a,100*m['BA'],100*m['PP_accuracy'],*[100*m['frequency_BA'][n] for n in ['head','middle','tail']],m['correct']])
with (d/'per_class.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['class_id','frequency','arm','images','correct','recall_percent'])
 for i,c in enumerate(cfg['order']):
  for a,m in results[0]['metrics'].items():w.writerow([c,next(n for n,v in cfg['frequency_bins'].items() if c in v),a,m['counts'][i],m['correct_by_class'][i],100*m['recall'][i]])
with (d/'training_outer_gap.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['outer','scope','class_id','images','anchor_accuracy','plain_accuracy','anchor_PP_accuracy','plain_PP_accuracy','anchor_margin_q0','anchor_margin_q25','anchor_margin_q50','anchor_margin_q75','anchor_margin_q100','plain_margin_q0','plain_margin_q25','plain_margin_q50','plain_margin_q75','plain_margin_q100'])
 for row in diag:
  for c in row['classes']:w.writerow([row['outer'],row['scope'],cfg['order'][c['class_index']],*[c[k] for k in ['images','anchor_accuracy','plain_accuracy','anchor_PP_accuracy','plain_PP_accuracy']],*c['anchor_margin_quantiles'],*c['plain_margin_quantiles']])
with (d/'direction_decomposition.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['outer','scope','class_id','group','images','PP_mass','parallel_mean','perpendicular_mean','competitor_switch_mean','parallel_PP_mean','perpendicular_PP_mean','parallel_negative','perpendicular_negative'])
 for row in diag:
  for c in row['classes']:
   for group,a in c['groups'].items():w.writerow([row['outer'],row['scope'],cfg['order'][c['class_index']],group,*[a[k] for k in ['images','PP_mass','parallel_mean','perpendicular_mean','competitor_switch_mean','parallel_PP_mean','perpendicular_PP_mean','parallel_negative','perpendicular_negative']]])
lines=['# 固定子空间聚合表','', '| 方法 | BA% | PP% | 头类BA% | 中类BA% | 尾类BA% | CLS损害数 | BA损害pp | PP损害pp |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for a,m in results[0]['metrics'].items():h=harm[a];lines.append('| '+names[a]+' | '+' | '.join(f'{100*x:.4f}' for x in [m['BA'],m['PP_accuracy'],*[m['frequency_BA'][n] for n in ['head','middle','tail']]])+f' | {h["harm"]} | {100*h["BA_harm"]:.4f} | {100*h["PP_harm"]:.4f} |')
lines+=['','| 外层折 | 完整普通BA% | 原头子空间BA% | PCA子空间BA% | 原头尾类BA% | PCA尾类BA% |','|---|---:|---:|---:|---:|---:|']
for row in results[1:]:m=row['metrics'];lines.append('| '+row['scope']+' | '+' | '.join(f'{100*x:.4f}' for x in [m['plain_correction']['BA'],m['score_subspace']['BA'],m['pca_subspace']['BA'],m['score_subspace']['frequency_BA']['tail'],m['pca_subspace']['frequency_BA']['tail']])+' |')
lines+=['','| 候选 | 相对任一强联合头新增的零召回类 |','|---|---|']
for a in cfg['new_arms']:
 m=results[0]['metrics'];zeros=[c for i,c in enumerate(cfg['order']) if m[a]['correct_by_class'][i]==0 and any(m[b]['correct_by_class'][i]>0 for b in ['scale_joint','df_matched_joint'])];lines.append(f'| {names[a]} | {zeros or "无"} |')
(p/'TABLES_ZH.md').write_text('\n'.join(lines)+'\n')
