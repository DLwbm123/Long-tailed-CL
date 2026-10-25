from pathlib import Path
import json,statistics
r=Path('/tmp/q1093');out=r/'results';c=json.loads((r/'SOURCE_COSTS.private.json').read_text());t=json.loads((r/'TARGET_CONTROL_EXIT.private.json').read_text());assert all(v['control']['exit_code']==0 for v in c)
cost={'source_jobs':len(c),'source_CPU_seconds':sum(v['control']['worker_CPU_seconds'] for v in c),'source_job_wall_seconds_sum':sum(v['control']['attempt_wall_seconds'] for v in c),'source_peak_allocated_bytes':max(v['control']['peak_allocated_bytes'] for v in c),'source_peak_RSS_bytes':max(v['control']['peak_RSS_bytes'] for v in c),'target':t,'source_job_costs':c,'all_source_jobs_exit0':True,'IO_is_not_simultaneous_allocated_storage':True}
(out/'COST_SUMMARY.json').write_text(json.dumps(cost,indent=2))
cur=json.loads((out/'FULL_TRAIN_CURVES.json').read_text());tr={};plat={}
for ds in ['NIH','ISIC','HK']:
 rows=[v for v in cur if v['dataset']==ds];z=[max([v for v in rows if v['seed']==seed],key=lambda v:v['stage'])['curves'][-1] for seed in [74002,74003,74004]];tr[ds]={k:statistics.mean(v[k] for v in z)*100 for k in ['crossfit_query_macro_BA','crossfit_query_tail_BA']};plat[ds]=sum((max(v['objective_mean'] for v in x['curves'][-3:])-min(v['objective_mean'] for v in x['curves'][-3:]))/max(1e-12,abs(statistics.mean(v['objective_mean'] for v in x['curves'][-3:])))<.01 for x in rows)
(out/'TRAIN_FIT_SUMMARY.json').write_text(json.dumps({'final_training':tr,'last3_descriptive_platform_counts':plat},indent=2))
a=json.loads((out/'ANALYSIS.json').read_text());g=json.loads((out/'GATE.json').read_text())
lines=['# R63完整结果：跨折训练判别很高，开发尾类仍未改善','','全三域三seed完整退出，原协议不变。45stage、52260有效AdamW更新、208230 micro backward、495条epoch曲线与90终点评测全部核验；结案0新增模型前向/优化。H0/R28 BOTH均false，各域final/tail联合改善均0/3，保留H0。固定R63关闭，不扫fold/rank/lr/anchor/temperature/epoch或挑frozen读出替代候选。','','## 完整结果','','单位%，阶段平均宏BA / 最终宏BA / 最终TRAIN<250尾BA。三seed均值。','','|域|support_lora候选|frozen_support诊断|对H0变化pp|对R28变化pp|','|---|---|---|---|---|']
def fmt(x):return ' / '.join(f'{x[k]:.6f}' for k in ['mean_BA','final_BA','tail_BA'])
for ds in a:lines.append('|'+ds+'|'+fmt(a[ds]['support_lora']['mean'])+'|'+fmt(a[ds]['frozen_support']['mean'])+'|'+fmt(g['H0'][ds]['deltas_pp'])+'|'+fmt(g['R28'][ds]['deltas_pp'])+'|')
lines+=['','## 全seed候选','','|域|seed|平均|最终|尾|','|---|---|---|---|---|']
for ds in a:
 for z in a[ds]['support_lora']['seeds']:lines.append(f"|{ds}|{z['seed']}|{z['mean_BA']:.6f}|{z['final_BA']:.6f}|{z['tail_BA']:.6f}|")
lines+=['','## 训练与泛化','','|域|最终TRAIN跨折宏BA|最终TRAIN跨折尾BA|开发尾BA|末3objective描述平台stage数|','|---|---|---|---|---|']
for ds,z in tr.items():lines.append(f"|{ds}|{z['crossfit_query_macro_BA']:.4f}|{z['crossfit_query_tail_BA']:.4f}|{a[ds]['support_lora']['mean']['tail_BA']:.4f}|{plat[ds]}|")
lines+=['','两折当前pass按metadata组件不交，但都反复参与同一LoRA训练；TRAIN跨折不是独立患者holdout。很高的训练尾判别没有转化为开发尾收益，不能说支持/查询分离已解决泛化，也不能唯一归因于记忆、噪声或分布。平台只是末3目标相对范围<1%的描述，不是收敛证明。','','HK平均/最终对H0约+6.43/+1.84pp，但尾-4.53pp；ISIC平均/最终提高但尾-4.86pp。NIH最终/尾下降，不选择有利域。冻结诊断重复seed来自同一模型，不是独立人群。训练用半支持、正式推理全支持，样本数/不确定性不同。固定失败不否定全部支持条件化方法。','','## 工程与费用','']
lines.append(f"原20h时钟保持，总历时{t['total_elapsed_seconds']:.6f}s（{t['total_elapsed_seconds']/3600:.2f}h）；目标CPU{t['worker_CPU_seconds']:.6f}s，源全部job CPU{cost['source_CPU_seconds']:.6f}s。目标peak {t['peak_allocated_bytes']}bytes，源peak {cost['source_peak_allocated_bytes']}bytes，分别低于44/24GB。目标退出留存{t['allocated_bytes']}bytes。源全部{len(c)}job退出码0。")
lines+=['','原FP32输入parity、source上限16000row、组划分prefix、support detach/20刷新、完整query/support覆盖、冻结原参数、strictrestore、全3seed TRAIN seal后真实VAL均核验。按域完成后只清理声明的再生token/base；保留head/prototype/base-tail/响应/曲线/计数/META/费用。IO/历史peak/删除/当前分账，不改旧预算，不为诊断保留大token。','','## 后续','','正式安全交付后执行有限计划A（保存输出诊断，0模型前向/反向/优化），再按证据决定有界B诊断及新的完整主线实验，不由单次负cos或有利域直接启动投影。','','全部是反复历史开发数据；无TEST、独立患者或临床效果证据。公开自有source/protocol、全class/seed/arm计数、曲线、费用摘要；不公开图像/身份/逐例响应/特征/token/权重/原日志/私有配置。']
(r/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'source_jobs':len(c),'source_CPU':cost['source_CPU_seconds'],'source_peak':cost['source_peak_allocated_bytes'],'platform':plat,'train':tr}))
