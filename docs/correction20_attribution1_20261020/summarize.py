"""Publish only aggregate evidence and a Chinese attribution report."""
import csv,json
from pathlib import Path
p=Path(__file__).parent;d=p/'results';cfg=json.loads((p/'config.json').read_text());rows=json.loads((d/'results.json').read_text());pooled=rows[0];audit=json.loads((d/'EXISTING_SIGNAL_AUDIT.json').read_text());v=json.loads((d/'VERIFICATION.json').read_text());cost=json.loads((d/'costs.json').read_text());names={'cls768_ridge':'CLS ridge','scale_joint':'尺度联合','df_matched_joint':'df联合','plain_correction':'联合普通修正','utility_correction':'原修复＋保留','cls_plain_correction':'CLS-only普通修正','mass_shuffled_correction':'质量守恒打散'}
with (d/'summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['scope','arm','BA_percent','PP_accuracy_percent','head_BA_percent','middle_BA_percent','tail_BA_percent','correct'])
 for row in rows:
  for a,m in row['metrics'].items():w.writerow([row['scope'],a,100*m['BA'],100*m['PP_accuracy'],*[100*m['frequency_BA'][n] for n in ['head','middle','tail']],m['correct']])
with (d/'signal_mass.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['outer','class_id','images','repair_images','keep_images','repair_weighted_fraction','keep_weighted_fraction','joint_only_weighted_fraction','fold0_patients','fold1_patients','fold2_patients','anchor_train_repair_accuracy','utility_train_repair_accuracy','anchor_train_keep_accuracy','utility_train_keep_accuracy'])
 for row in audit['inner']:
  w.writerow([row[k] for k in ['outer','class_id','images','repair_images','keep_images','repair_weighted_fraction','keep_weighted_fraction','joint_only_weighted_fraction']]+row['patient_support_by_fold']+[row['training_fit_on_signals'][arm][kind] for kind in ['repair','keep'] for arm in ['scale_joint','utility_correction']])
with (d/'per_class.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['class_id','arm','images','correct','recall_percent'])
 for i,c in enumerate(cfg['order']):
  for a,m in pooled['metrics'].items():w.writerow([c,a,m['counts'][i],m['correct_by_class'][i],100*m['recall'][i]])
lines=['# 最终20类归因：冻结信号审计与两个最小对照','', '固定原训练内部嵌套划分，复用父轮五臂，仅新增CLS-only普通残差和质量守恒打散各三头。所有新头封存后才评价，新增编码和ridge拟合均0。父轮外层结果已用于设计本轮，因此下述是开发归因证据，不能声称全新独立最终测试。','', '| 方法 | BA% | PP正确率% | 头类BA% | 中类BA% | 尾类BA% |','|---|---:|---:|---:|---:|---:|']
for a,m in pooled['metrics'].items():lines.append('| '+names[a]+' | '+' | '.join(f'{100*x:.4f}' for x in [m['BA'],m['PP_accuracy'],*[m['frequency_BA'][n] for n in ['head','middle','tail']]])+' |')
a=pooled['metrics']['plain_correction'];b=pooled['metrics']['cls_plain_correction'];u=pooled['metrics']['utility_correction'];s=pooled['metrics']['mass_shuffled_correction']
lines+=['','## 两个问题的直接比较','',f'联合普通修正减CLS-only普通修正：BA {100*(a["BA"]-b["BA"]):+.4f}个百分点，尾类BA {100*(a["frequency_BA"]["tail"]-b["frequency_BA"]["tail"]):+.4f}个百分点。CLS对照的锚点、温度、输入和残差全链路无patch；CLS块单位与残差L2规则匹配。两条管线参数数目、锚点和温度仍不同，所以这是同损失形式下的完整管线比较，不能单凭它排除容量或先验作用。','',f'原修复＋保留减质量守恒打散：BA {100*(u["BA"]-s["BA"]):+.4f}个百分点，尾类BA {100*(u["frequency_BA"]["tail"]-s["frequency_BA"]["tail"]):+.4f}个百分点。打散严格保留每类repair/keep/target联合加权分布和双边缘质量，改变信号与样本对应关系；使用分数质量搬运而非0/1硬置换。只有一次固定排列，不能作随机化显著性结论。','', '| 外层折 | 联合普通−CLS普通 BA pp | 原保护−打散 BA pp |','|---|---:|---:|']
for row in rows[1:]:
 mm=row['metrics'];lines.append(f'| {row["scope"]} | {100*(mm["plain_correction"]["BA"]-mm["cls_plain_correction"]["BA"]):+.4f} | {100*(mm["utility_correction"]["BA"]-mm["mass_shuffled_correction"]["BA"]):+.4f} |')
lines+=['','## 小类训练侧信号','', '下表为三个外层训练部分的逐折范围，百分比均在对应真实类别内部按患者等权归一。训练拟合正确率是在已用于拟合的样本上计算，有乐观性；不能当作外层迁移证据。','', '| 类别 | 内层repair质量%范围 | keep质量%范围 | 原保护训练repair正确率%范围 | 原保护训练keep正确率%范围 |','|---|---:|---:|---:|---:|']
def fmt(values):
 values=[100*x for x in values if x is not None];return '无信号' if not values else f'{min(values):.2f}–{max(values):.2f}'
for c in cfg['audit_classes']:
 rr=[r for r in audit['inner'] if r['class_id']==c];lines.append('| '+str(c)+' | '+fmt([r['repair_weighted_fraction'] for r in rr])+' | '+fmt([r['keep_weighted_fraction'] for r in rr])+' | '+fmt([r['training_fit_on_signals']['utility_correction']['repair'] for r in rr])+' | '+fmt([r['training_fit_on_signals']['utility_correction']['keep'] for r in rr])+' |')
lines+=['','## 父轮已有正确样本的错判去向','', '下表只描述父轮原保护相对尺度联合丢失的正确样本，不参与新对照拟合或选参。类别ID按原映射。','', '| 类别 | 原保护损害图片数 | 错判目的类：图片数 |','|---|---:|---|']
for a in audit['outer']:
 h=a['harmed_transitions']['utility_correction'];lines.append(f'| {a["class_id"]} | {h["images"]} | '+(', '.join(f'{c}: {n}' for c,n in h['destinations'].items()) or '无')+' |')
lines+=['','## 验收、资源与边界','',f'独立验收{v["state"]}：重算{v["new_predictions_recounted"]}条新预测及28组七臂指标，最大分数误差{v["max_score_error"]:.3g}、目标误差{v["max_objective_error"]:.3g}、梯度差{v["max_gradient_error"]:.3g}；检查模型封存、CLS无patch和加权搬运质量守恒。', '',f'本轮实际拟合{cost["correction_attempts"]}次，成功{cost["correction_completed"]}头，编码0、ridge0。进程主体{cost["process_wall_seconds"]:.3f}秒；历史已记录成本继续累计，历史未完整计量开销不隐去。新增目录与父轮合计由guardian保持低于19.9GB，并预留100MB本地开销，唯一存储硬限20GB。','', '本轮不更改目标、容差、正则或排列追分，不添加模块/数据/种子，不访问原验证集/E/test，不扩展六阶段。所有负结果、失败和审计证据保留。公开聚合结果与源码，患者身份、图片、特征、模型与逐图分数不公开。','', '详细证据：[七臂指标](results/summary.csv)、[逐类指标](results/per_class.csv)、[信号质量](results/signal_mass.csv)、[完整旧信号审计](results/EXISTING_SIGNAL_AUDIT.json)、[独立验收](results/VERIFICATION.json)。']
(p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
