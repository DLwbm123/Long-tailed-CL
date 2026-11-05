"""Saved aggregate-only closeout. No model, encoder, optimizer or dataset access."""
import csv,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
load=lambda n:json.loads((R/n).read_text())
rows=lambda n:list(csv.DictReader((R/n).open()))
write=lambda n,x:(R/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
S=load('RESULT_SUMMARY.json')['streams'];P=load('SAVED_POLICY_AUDIT.json');M=load('SAVED_MEMORY_AUDIT.json');C=load('COSTS.json');X=load('CONTROL_EXIT.json')
assert X['exit_code']==0 and M['passed'] and M['endpoints']==70 and load('VERIFICATION.json')['passed']
assert all(v['equal_predictions'] and v['equal_memory_files'] for v in P['stages'])
meta=rows('META_SUMMARY.csv');out=[]
for stream in ['a','b']:
 for stage in range(5):
  a=[v for v in meta if v['stream']==stream and int(v['stage'])==stage];assert len(a)==16
  out.append(dict(stream=stream,stage=stage,groups=len(a),complete=sum(v['status']=='COMPLETE' for v in a),gspo_return_mean=statistics.mean(float(v['gspo_return']) for v in a),pg_return_mean=statistics.mean(float(v['pg_return']) for v in a),cf_absolute_effect_mean=statistics.mean(float(v['cf_abs_mean']) for v in a),gspo_zero_groups=sum(v['gspo_zero']=='True' for v in a),pg_zero_groups=sum(v['pg_zero']=='True' for v in a)))
with (R/'META_STAGE_SUMMARY.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
cls=rows('CLASS_STAGE.csv');stage=rows('STAGE_METRICS.csv');tiny=[];ex=[]
for stream in ['a','b']:
 for t in range(5):
  for arm in S[stream]['final']:
   a=[r for r in cls if r['stream']==stream and int(r['stage'])==t and r['arm']==arm];assert a
   good=[r for r in a if r['tiny']=='False'];b=[r for r in cls if r['stream']==stream and int(r['stage'])==t and r['arm']=='BALANCED'];baseline={r['class']:r for r in b}
   for r in a:
    if r['tiny']=='True':tiny.append({k:r[k] for k in ['stream','stage','arm','original_label','legacy_index','count','correct','recall']})
   full=sum(float(r['recall'])-float(baseline[r['class']]['recall']) for r in a)/len(a)
   excl=sum(float(r['recall'])-float(baseline[r['class']]['recall']) for r in good)/len(good)
   contrib=sum(float(r['recall'])-float(baseline[r['class']]['recall']) for r in a if r['tiny']=='True')/len(a)
   assert abs(full-excl*len(good)/len(a)-contrib)<1e-10
   ex.append(dict(stream=stream,stage=t,arm=arm,class_count=len(a),non_tiny_class_count=len(good),macro_delta_vs_BALANCED_pp=full,non_tiny_macro_delta_vs_BALANCED_pp=excl,tiny_original_denominator_contribution_pp=contrib))
for name,a in [('TINY_CLASS_STAGE.csv',tiny),('TINY_CONTRIBUTION.csv',ex)]:
 with (R/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(a[0]),lineterminator='\n');w.writeheader();w.writerows(a)
C['controller_exit']=X;C['measured_worker_CPU_total_seconds']=sum(v['worker_CPU_seconds'] for v in X['workers'].values());C['measured_workers_plus_controller_CPU_seconds']=C['measured_worker_CPU_total_seconds']+X['controller_CPU_seconds'];C['science_optimizer_total']=sum(sum(v['counts'][k] for k in ['formal_classifier_optimizer','shadow_classifier_optimizer','policy_optimizer','shadow_value_optimizer']) for v in C['workers'].values());C['engineering_optimizer_total']=sum(sum(v['counts'][k] for k in ['native_classifier_optimizer','native_policy_optimizer','native_value_optimizer']) for v in C['workers'].values());C['GPU_kernel_seconds']=None;C['software_preparation_CPU']='nonzero incomplete sampling; successful checks were repeated, last sampled CPU .335683157s is not their total';write('COSTS.json',C)
V=load('VERIFICATION.json');V.update(saved_memory_cross_stage_subset_audit=M['passed'],saved_policy_likelihood_and_zero_group_aggregate_audit=True,all10_formal_GSPO_PG_memories_and_predictions_equal=True,new_closeout_forward_optimizer=0,tiny_contribution_identity_max_error_pp=1e-10);write('VERIFICATION.json',V)
lines=['# R106 完整结果分析','','两条五阶段任务流、七臂均完成；70个分类器与10个策略/价值端点全部先封存，再统一VAL。所有worker和controller exit0。没有新增训练、调参或重采样救援。','','## 主结论','','Survival-GSPO在分层order0超过BALANCED和同价值头确定性选择，但在Ordered没有超过这两个关键参照，不能声称已获得跨任务流稳定优势。GSPO与匹配动作级PG在两流全部阶段选出同一正式记忆，最终预测完全相同，无法支持GSPO相对PG的独立贡献。','','|流|方法|平均macro%|最终macro%|最终tail%|','|---|---|---:|---:|---:|']
for stream in ['a','b']:
 for arm in S[stream]['final']:
  s=S[stream];lines.append(f"|{'Ordered' if stream=='a' else 'order0'}|{arm}|{s['mean'][arm]['macro']:.6f}|{s['final'][arm]['macro']:.6f}|{s['final'][arm]['tail']:.6f}|")
lines+=['','## 五阶段完整主方法差值','','各单元为GSPO−参照macro/tail百分点；无tail类别的阶段记N/A，不补零。','','|流|阶段|vs BALANCED|vs 同价值GREEDY|','|---|---:|---:|---:|']
for stream in ['a','b']:
 for t in range(5):
  rr={r['arm']:r for r in stage if r['stream']==stream and int(r['stage'])==t};g=rr['SURVIVAL_GSPO'];values=[]
  for ref in ['BALANCED','VALUE_GREEDY']:
   a=float(g['macro'])-float(rr[ref]['macro']);b=f"{float(g['tail'])-float(rr[ref]['tail']):+.6f}" if g['tail'] else 'N/A';values.append(f'{a:+.6f}/{b}')
  lines.append(f"|{stream}|{t}|{values[0]}|{values[1]}|")
lines+=['','## 微小类别及旧尾部','','以下列出最终GSPO−BALANCED的原分母贡献；排除tiny的重算只作敏感性分析，不替换主指标。tiny指原先legacy 1/2/12三类，其VAL分母小，不能当独立重复。全部七臂五阶段tiny正确数/分母公开于TINY_CLASS_STAGE.csv。','','|流|最终macro差pp|非tiny20类macro差pp|tiny按原23类分母贡献pp|','|---|---:|---:|---:|']
for stream in ['a','b']:
 a=next(v for v in ex if v['stream']==stream and v['stage']==4 and v['arm']=='SURVIVAL_GSPO');lines.append(f"|{stream}|{a['macro_delta_vs_BALANCED_pp']:+.6f}|{a['non_tiny_macro_delta_vs_BALANCED_pp']:+.6f}|{a['tiny_original_denominator_contribution_pp']:+.6f}|")
lines+=['','|流|tiny原label|VAL正确/分母 GSPO|BALANCED|同价值GREEDY|MORE|','|---|---:|---:|---:|---:|---:|']
for stream in ['a','b']:
 a=[r for r in tiny if r['stream']==stream and int(r['stage'])==4 and r['arm']=='SURVIVAL_GSPO']
 for r in a:
  vals=[]
  for arm in ['SURVIVAL_GSPO','BALANCED','VALUE_GREEDY','VALUE_GREEDY_MORE']:
   z=next(v for v in tiny if v['stream']==stream and v['stage']==r['stage'] and v['original_label']==r['original_label'] and v['arm']==arm);vals.append(f"{z['correct']}/{z['count']}")
  lines.append(f"|{stream}|{r['original_label']}|"+'|'.join(vals)+'|')
lines+=['','|流|旧尾有效阶段数|GSPO−BAL mean/final old-tail pp|GSPO−GREEDY mean/final old-tail pp|','|---|---:|---:|---:|']
for stream in ['a','b']:
 s=S[stream];f=lambda ref:f"{s['comparisons'][ref]['mean']['old_tail']:+.6f}/{s['comparisons'][ref]['final']['old_tail']:+.6f}";lines.append(f"|{stream}|{s['valid_stages']['SURVIVAL_GSPO']['old_tail']}|{f('BALANCED')}|{f('VALUE_GREEDY')}|")
lines+=['','order0最终整体尾部改善并不等于旧尾全部改善：对同价值GREEDY，GSPO最终old-tail仍下降。Ordered对BALANCED的旧尾平均和最终均明显下降。完整old/new×head/tail分组见GROUP_STAGE.csv；不隐藏缺失组或负阶段。','','## 为什么GSPO与PG同分','','saved-only检查发现：160个训练组均非零优势组，策略梯度非零；每角色每组4条轨迹末端记忆均不同。两角色同成员使用配对随机种子，每个角色内部四成员独立，不能把跨角色8条轨迹算8次独立重复。所有记录更新均未触发ratio裁剪；最终策略参数max差约0.001至0.0055，但640对影子成员及10个正式阶段的采样记忆仍相同。正式分类器输入、更新随机种子和预算也配对，所以正式记忆相同后预测相同是合理结果。','','这排除了“全零奖励/根本没有actor更新”的解释，但没有证明actor已学到跨阶段稳定决策。序列和动作级surrogate在旧策略附近首步梯度一致，后续差异很小；共同随机数下本次没有转化为不同离散选择。不能由此证明两种目标普遍等价，也不能把order0相对确定性策略的收益全归于RL：本轮没有额外的均匀随机记忆对照，随机选择与学到的策略仍未分离。不以该发现追加事后对照或调步数。详见SAVED_POLICY_AUDIT.json。','','## 内存、历史访问与费用','','70个实际记忆文件及统计表均核：每臂528568B以内；每阶段旧token严格属于该臂上一阶段已存集合，未见类统计count0/birth−1，父全局依赖与类覆盖通过。模型/优化器状态另计：每臂最终分类器+AdamW序列化约442KB，公共价值/双actor及优化器合计463195B；它们不是额外回放数据。所有历史特征/像素由阶段访问边界禁止恢复，不代表从保护源盘删除原图。实际文件字节见SAVED_MEMORY_AUDIT.json、模型状态存储见STATE_STORAGE_SUMMARY.json。','','物理科学optimizer共104840：正式分类器21000、影子分类器78720、actor640、价值4480。工程optimizer共2112单列；native/profile计入worker费用不再相加。两流共1280条真实影子轨迹、64640个动作采样，两个复用更新产生129280动作更新暴露，不算额外采样。公共CF/前缀计算实际计一次；MORE逻辑影子步数204/组，高于GSPO188/组，但不冒称wall严格匹配。','',f"controller wall {X['controller_wall_seconds']:.6f}s（约{X['controller_wall_seconds']/60:.2f}min）；原FIRST到退出{X['original_elapsed_seconds']:.6f}s（约{X['original_elapsed_seconds']/60:.2f}min，含准备原钟未重置）。两个worker CPU合计{C['measured_worker_CPU_total_seconds']:.6f}s，加controller已采样CPU总{C['measured_workers_plus_controller_CPU_seconds']:.6f}s；core计时含在worker中。峰group RSS{X['peak_sampled_group_RSS_bytes']}B，退出时新增实际磁盘{X['allocated_bytes']}B。最后结案文档另增少量磁盘与CPU，最终原钟见私有交付收据。",'','准备/失败的软件检查/导入/SSH/文档/发布CPU非零未完整采样；没有GPUkernel实耗计时，不把2.73h保守准入预算与14min实际workerwall差值称方法效率收益。工程失败记录保留，无科学重启。','','## 证据边界与停止','','这是冻结的单seed开发实验，非患者独立或统计显著证据。新方法的通用冻结前端/严格证据记忆与此前全历史强监督配置不同，绝对准确率不能用于归因比较。结果支持“未来学习生存反馈值得研究”的局部信号，但不足支持完整主方法、GSPO独立贡献或顶会创新成立；负阶段和强对照全部保留。','','本轮完整交付后停止；监测保持OFF，不自动追加order/seed/数据集、超参搜索或救援。']
(R/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')
report=(R/'REPORT_ZH.md').read_text();prefix='> 完整结论：order0有正收益，Ordered未胜关键强对照；GSPO与PG正式记忆/预测相同，尚无GSPO独立贡献证据。完整逐阶段、tiny/旧尾、saved-only策略与内存核验见[ANALYSIS_ZH.md](ANALYSIS_ZH.md)。\n\n'
if not report.startswith('> 完整结论'):(R/'REPORT_ZH.md').write_text(prefix+report)
write('CLOSEOUT_VERIFICATION.json',{'passed':True,'saved_only':True,'new_encoder_forward_optimizer':0,'actual_memory_endpoint_audit':70,'policy_endpoint_aggregate_audit':10,'stage_metric_rows':len(stage),'class_stage_rows':len(cls),'tiny_stage_rows':len(tiny),'tiny_contribution_rows':len(ex),'meta_stage_rows':len(out),'all_worker_exit0':True,'all_classifier_seals':70,'no_posthoc_retrain':True,'monitor':'OFF'})
print('saved-only closeout passed; no new forwards/optimizer')
