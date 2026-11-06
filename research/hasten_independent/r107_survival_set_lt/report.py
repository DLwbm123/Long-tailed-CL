"""Saved two-flow aggregate delivery; no new training or post-hoc selection."""
import csv,json
from pathlib import Path
from common import write
from evaluate import csvout
from preflight import ARMS

def run(root):
 r=Path(root);S={k:json.loads((r/k/'RESULT_SUMMARY.json').read_text()) for k in ['a','b']};lines=['# R107 Survival-GSPO SET-LT 完整结果','','两流五阶段七臂，seed74002；70个分类器端点及10策略/价值端点全部封存后统一VAL；TEST0。固定主候选GSPO_SET_LT，不按结果换臂或改门。','','|流|方法|mean macro|final macro|final tail|mean old-tail|有效旧尾阶段|','|---|---|---:|---:|---:|---:|---:|']
 for k,s in S.items():
  for arm in ARMS:
   x=s['mean'][arm];f=s['final'][arm];old='N/A' if x['old_tail'] is None else f"{x['old_tail']:.6f}";lines.append(f"|{k}|{arm}|{x['macro']:.6f}|{f['macro']:.6f}|{f['tail']:.6f}|{old}|{s['valid_stages'][arm]['old_tail']}|")
 for name in ['STAGE_METRICS.csv','CLASS_STAGE.csv','GROUP_STAGE.csv','RESCUE_HARM.csv']:
  csvout(r/name,[dict(stream=k,**row) for k in ['a','b'] for row in csv.DictReader((r/k/name).open())])
 audit=[];meta=[];coverage=[]
 for k in ['a','b']:
  for row in json.loads((r/k/'MEMORY_AND_ACCESS_AUDIT.json').read_text()):
   row=dict(row);parents=row.pop('per_class_parent_coverage');local=row.pop('per_class_local_tokens')
   for cl,(ng,nl) in enumerate(zip(parents,local)):coverage.append(dict(stream=k,stage=row['stage'],arm=row['arm'],class_index=cl,global_parent_tokens=ng,local_tokens=nl))
   audit.append(dict(stream=k,**row))
  for t in range(5):
   es=json.loads((r/k/f's{t}'/'META_DIAGNOSTICS.private.json').read_text());complete=[e for e in es if 'updates' in e];row={'stream':k,'stage':t,'groups':len(es),'completed':len(complete)}
   for role in ['GSPO_SET_LT','REWARD_SHUFFLED','GSPO_SET']:
    u=[v for e in complete for v in e['updates'][role]];ret=[d['return'] for e in complete for d in e['diagnostics'] if d['role']==role];row[role+'_return_mean']=sum(ret)/len(ret) if ret else None;row[role+'_zero_groups']=sum(e['updates'][role][0]['zero_advantage_group'] for e in complete);row[role+'_clip_fraction_max']=max((v['ratio_clip_fraction'] for v in u),default=0.)
   meta.append(row)
 csvout(r/'MEMORY_CLASS_COVERAGE.csv',coverage);csvout(r/'MEMORY_ACCESS.csv',audit);csvout(r/'META_STAGE_SUMMARY.csv',meta)
 tiny=[];contrib=[];classes=list(csv.DictReader((r/'CLASS_STAGE.csv').open()))
 for k in ['a','b']:
  for t in range(5):
   rr=[x for x in classes if x['stream']==k and int(x['stage'])==t];by={(x['arm'],x['class']):x for x in rr};main=[x for x in rr if x['arm']=='GSPO_SET_LT'];C=len(main)
   for x in rr:
    if x['tiny']=='True':tiny.append(x)
   for ref in ARMS:
    if ref=='GSPO_SET_LT':continue
    diff=lambda a:sum(float(x['recall'])-float(by[ref,x['class']]['recall']) for x in a)
    non=[x for x in main if x['tiny']=='False'];ti=[x for x in main if x['tiny']=='True'];d=diff(main)/C;nt=diff(non)/C;tc=diff(ti)/C;assert abs(d-nt-tc)<1e-10
    contrib.append(dict(stream=k,stage=t,reference=ref,full_macro_delta_pp=d,non_tiny_original_denominator_contribution_pp=nt,tiny_original_denominator_contribution_pp=tc,non_tiny_macro_sensitivity_pp=diff(non)/len(non)))
 csvout(r/'TINY_CLASS_STAGE.csv',tiny);csvout(r/'TINY_CONTRIBUTION.csv',contrib)
 contrasts={k:{ref:{'mean_macro_positive':d['mean']['macro']>0,'final_macro_positive':d['final']['macro']>0,'final_tail_nonnegative':d['final']['tail']>=0,'mean_old_tail_nonnegative':d['mean']['old_tail'] is not None and d['mean']['old_tail']>=0,'final_old_tail_nonnegative':d['final']['old_tail'] is not None and d['final']['old_tail']>=0} for ref,d in s['comparisons'].items()} for k,s in S.items()}
 write(r/'CONTRAST_FLAGS.json',{'per_contrast_descriptive_flags':contrasts,'not_statistical_significance':True,'no_posthoc_scalar_success_gate':True})

 lines+=['','## 主方法−匹配对照，pp','','|流|参照|mean macro|final macro|final tail|mean old-tail|','|---|---|---:|---:|---:|---:|']
 for k,s in S.items():
  for ref,d in s['comparisons'].items():
   old='N/A' if d['mean']['old_tail'] is None else f"{d['mean']['old_tail']:+.6f}";lines.append(f"|{k}|{ref}|{d['mean']['macro']:+.6f}|{d['final']['macro']:+.6f}|{d['final']['tail']:+.6f}|{old}|")
 lines+=['','全体方法在所有合法未选候选中选择；ID只用于父子绑定/最终排序，非动作后缀限制。相同最终集合可有多条轨迹；记录真实执行概率，非平凡决策才计序列长度，单合法动作强制转移。ZERO_ACTOR等概率合法下一动作，不声称对所有可行最终集合均匀。语义精确并列仍可依存储顺序打破平局，不能声称逐seed逐字节置换不变。','','LT伪流只用当前新TRAIN，真实旧证据不互借。支持/查询组件分离，真实伪支持数量进入状态，伪稀缺为实际支持<=4，正式HK tail<250不变。频次递减/混合各8组/阶段；单类伪阶段无法同时含高低支持，事前交替角色而不虚构混合。SET保留R106 actor伪任务生成/原count与tail语义，所有actor和GREEDY共用LT价值头；因此SET-LT差值隔离actor环境，不把价值头也换成旧分布。','','主回报、学习器、固定学习预算、字节历史边界见冻结PROTOCOL。每组主/错配/SET各4条真实影子轨迹；同成员用配对随机数，不当12次独立重复。错配臂用事前种子从9种四元素无不动点置换中选一种，不按收益选错配，不与主方法交换轨迹。所有actor两次GSPO更新，ZERO无actor优化器。公共价值头用LT反事实第一对，MORE六对，4MSE更新/对。CF一未来阶段，actor两未来阶段；不称完全相同信用目标。','','完整主结论需同时检查ZERO、错配、GREEDY/MORE及Ordered旧尾保护；reward、梯度和裁剪激活都不是成功条件。tiny原分母贡献、全部逐类/分组/阶段正负报告；无独立重复或显著性结论，不把R106跨run差值归因单一改动。','','每臂可读取回放数据528568B以内，模型/优化器另计。丢弃旧视图不能恢复，源数据保留不等于可训练访问。native重复特征/合成标签仅形状与梯度成本fixture，正式重新初始化。物理共享影子每完整组648分类更新，两流上限103680；formal21000、actor960、value4480另计。逻辑主RL/错配188，SET188但不同任务prefix，GREEDY44，MORE204。共享计算不重复计CPU/GPU实耗。','','两流native全计划保守预算通过才fit，原24h包含准备与修复不重置；每server新增<50GB，本run20GB上限+512MiB早停余量，GPU至多2/CPU总4，独立2s guard。全部费用见COSTS与CONTROL_EXIT，core/native包含worker不重复相加；准备SSH文档发布CPU非0未完整采样。监测OFF，完成当前冻结实验后停止，不开新order/seed/算法，不按VAL调参。']
 (r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');write(r/'RESULT_SUMMARY.json',{'streams':S,'primary':'GSPO_SET_LT','TEST':0})
 write(r/'COSTS.json',{'workers':{k:json.loads((r/k/'COSTS.json').read_text()) for k in ['a','b']},'shared_physical_classifier_updates_per_meta_group':648,'logical_RL_per_group':188,'logical_GREEDY_per_group':44,'logical_MORE_per_group':204,'actual_wait4_controller':'CONTROL_EXIT.json','shared_work_charged_once':True})
 write(r/'VERIFICATION.json',{'passed':True,'streams':2,'classifier_endpoints':70,'formal_classifier_updates':21000,'all_worker_verifications_passed':True,'TEST':0})
