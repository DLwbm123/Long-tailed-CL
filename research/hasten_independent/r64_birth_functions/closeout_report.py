from pathlib import Path
import json,statistics
r=Path('/tmp/q1096');out=r/'results';c=json.loads((r/'SOURCE_COSTS.private.json').read_text());t=json.loads((r/'CONTROL_EXIT.private.json').read_text());assert all(v['control']['exit_code']==0 for v in c)
cost={'source_jobs':len(c),'source_CPU_seconds':sum(v['control']['worker_CPU_seconds'] for v in c),'source_job_wall_seconds_sum':sum(v['control']['attempt_wall_seconds'] for v in c),'source_peak_allocated_bytes':max(v['control']['peak_allocated_bytes'] for v in c),'source_peak_RSS_bytes':max(v['control']['peak_RSS_bytes'] for v in c),'target':t,'source_job_costs':c,'all_source_jobs_exit0':True,'IO_is_not_simultaneous_allocated_storage':True}
(out/'COST_SUMMARY.json').write_text(json.dumps(cost,indent=2))
cur=json.loads((out/'FULL_TRAIN_CURVES.json').read_text());tr={};plat={}
for ds in ['NIH','ISIC','HK']:
 rows=[v for v in cur if v['dataset']==ds];z=[max([v for v in rows if v['seed']==seed],key=lambda v:v['stage'])['curves'][-1] for seed in [74002,74003,74004]];tr[ds]={k:statistics.mean(v[k] for v in z)*100 for k in ['crossfit_query_macro_BA','crossfit_query_tail_BA']};plat[ds]=sum((max(v['objective_mean'] for v in x['curves'][-3:])-min(v['objective_mean'] for v in x['curves'][-3:]))/max(1e-12,abs(statistics.mean(v['objective_mean'] for v in x['curves'][-3:])))<.01 for x in rows)
(out/'TRAIN_FIT_SUMMARY.json').write_text(json.dumps({'final_training':tr,'last3_descriptive_platform_counts':plat},indent=2))
a=json.loads((out/'ANALYSIS.json').read_text());g=json.loads((out/'GATE.json').read_text())
lines=['# R64完整结果：旧出生函数严格保持，整体与尾类仍下降','','全三域三seed完整退出，原协议不变。45stage、52260有效AdamW更新、208230 micro backward、495条epoch曲线与90终点评测全部核验；结案0新增模型前向/优化。H0/R28 BOTH均false，各域final/tail联合改善均0/3，保留H0。固定R64关闭，不扫fold/rank/lr/anchor/temperature/epoch或挑frozen读出替代候选。','','## 完整结果','','单位%，阶段平均宏BA / 最终宏BA / 最终TRAIN<250尾BA。三seed均值。','','|域|birth_functions候选|frozen_support诊断|对H0变化pp|对R28变化pp|','|---|---|---|---|---|']
def fmt(x):return ' / '.join(f'{x[k]:.6f}' for k in ['mean_BA','final_BA','tail_BA'])
for ds in a:lines.append('|'+ds+'|'+fmt(a[ds]['birth_functions']['mean'])+'|'+fmt(a[ds]['frozen_support']['mean'])+'|'+fmt(g['H0'][ds]['deltas_pp'])+'|'+fmt(g['R28'][ds]['deltas_pp'])+'|')
lines+=['','## 全seed候选','','|域|seed|平均|最终|尾|','|---|---|---|---|---|']
for ds in a:
 for z in a[ds]['birth_functions']['seeds']:lines.append(f"|{ds}|{z['seed']}|{z['mean_BA']:.6f}|{z['final_BA']:.6f}|{z['tail_BA']:.6f}|")
lines+=['','## 训练与泛化','','|域|最终TRAIN跨折宏BA|最终TRAIN跨折尾BA|开发尾BA|末3objective描述平台stage数|','|---|---|---|---|---|']
for ds,z in tr.items():lines.append(f"|{ds}|{z['crossfit_query_macro_BA']:.4f}|{z['crossfit_query_tail_BA']:.4f}|{a[ds]['birth_functions']['mean']['tail_BA']:.4f}|{plat[ds]}|")
lines+=['','TRAIN跨折不是独立holdout。R64严格保持旧出生函数，但会保留出生错误；旧query只能通过新类别负竞争影响活跃函数，不能改善旧函数。正式VAL保存分数的36对相邻stage、21120旧查询行、990个旧expert张量比较均tensor-exact，0新模型前向。保持旧函数没有保证整体准确率或尾类收益；新增类别竞争仍可改变argmax。','','## 工程与资源','']
lines.append(f"原20h时钟保持，总历时{t['total_elapsed_seconds']:.6f}s（{t['total_elapsed_seconds']/3600:.2f}h）；目标CPU{t['worker_CPU_seconds']:.6f}s，源全部job CPU{cost['source_CPU_seconds']:.6f}s。目标peak {t['peak_allocated_bytes']}bytes，源peak {cost['source_peak_allocated_bytes']}bytes；目标退出留存{t['allocated_bytes']}bytes。源全部{len(c)}job退出0。")
lines+=['','45stage/52260updates/208230micro/495curves/90readouts，所有原参数冻结、恢复、foldprefix、旧expert与VAL旧列不变、query10N/新类support10Nnew、全3seed seal后真实VAL、输入parity、三域再生清理回执核验通过。结案0模型前向/反向/optimizer。保留所有科学/恢复/失败/费用，原数据与共享依赖未改。','','共享主机I/O等待incident保留：NIH第二seed中DataLoader曾D态/folio_wait，主机I/O pressure较高、swap近满但可用内存较多；后续恢复，未杀进程/重启/改变原钟或科学配置。观察不证明唯一根因，等待时间包含在原总历时，不当作方法性能失败原因。','','## 解释边界与后续','','本配置相对H0、R28全部域final/tail联合改善0/3，BOTHfalse；保留H0，不择冻结诊断或有利seed，不扫描expert温度/融合/冻结时点/rank/lr/epoch救援。新增保留容量和推理tail次数、旧类梯度被移除、支持刷新曝光不同，与R63不是单因素比较。全部为反复历史开发，没有TEST/独立患者/临床证据。CIFAR HASTEN复现的上游投影问题与R64无关，R64没有使用该投影，不据此改R64结果。下一轮先综合R62–R64训练/开发与保存响应证据，冻结可证伪的新风险假设与完整协议，不盲训或重复基线调优。']
(r/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'source_jobs':len(c),'source_CPU':cost['source_CPU_seconds'],'source_peak':cost['source_peak_allocated_bytes'],'platform':plat,'train':tr}))
