"""Saved aggregates only: no model loading, data access, forward, optimizer or relaunch."""
import csv,json,time
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text())
write=lambda n,x:(R/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
S=read('RESULT_SUMMARY.json')['streams'];X=read('CONTROL_EXIT.json');V=read('VERIFICATION.json');P=read('SAVED_POLICY_AUDIT.json');C=read('COSTS.json');A=list(csv.DictReader((R/'MEMORY_ACCESS.csv').open()));st=list(csv.DictReader((R/'STAGE_METRICS.csv').open()));ti=list(csv.DictReader((R/'TINY_CONTRIBUTION.csv').open()))
assert X['exit_code']==0 and V['passed'] and P['passed'] and len(A)==70 and len(st)==70
assert all(int(x['memory_bytes'])<=528568 and x['old_views_from_own_previous_memory_only']=='True' for x in A)
assert all(v['ZERO_last_weight_bias_exactly_zero'] and v['reward_derangement_all_no_fixed_points'] for v in P['stages'])
lines=['# R107 主方法与学习必要性分析','','## 结论','','完整两流五阶段七臂均已完成，无科学失败或重启。GSPO_SET_LT未超过ZERO_ACTOR；奖励错配对照也未被稳定超过。Ordered旧尾相对本轮BALANCED改善，但不足支持学到的RL规则优于随机记忆。两流均不能认定主方法的策略学习增量成立。','','## 全七臂结果','','单位%；a=Ordered，b=分层order0。平均macro含全部五阶段；tail/old-tail只平均实际存在该组的阶段，不补零。','','|流|方法|mean macro|final macro|final tail|final old-tail|','|---|---|---:|---:|---:|---:|']
for k,s in S.items():
 for arm in s['final']:
  f=s['final'][arm];lines.append(f"|{k}|{arm}|{s['mean'][arm]['macro']:.6f}|{f['macro']:.6f}|{f['tail']:.6f}|{f['old_tail']:.6f}|")
lines+=['','## 主方法−各匹配对照，百分点','','|流|参照|mean macro|final macro|final tail|mean old-tail|final old-tail|','|---|---|---:|---:|---:|---:|---:|']
for k,s in S.items():
 for ref,d in s['comparisons'].items():lines.append(f"|{k}|{ref}|{d['mean']['macro']:+.6f}|{d['final']['macro']:+.6f}|{d['final']['tail']:+.6f}|{d['mean']['old_tail']:+.6f}|{d['final']['old_tail']:+.6f}|")
lines+=['','Ordered：主方法虽比BALANCED平均macro+1.370583、最终macro+5.604261，但比ZERO_ACTOR平均macro−2.196477、最终macro−5.780495。对奖励错配最终macro仅+0.058490，finaltail与oldtail反而更低；因此不能将相对均衡规则的收益归因于正确长期回报。对同价值GREEDY平均macro较高而最终macro/tail较低，亦非稳定优势。','','order0：主方法最终macro比ZERO−1.065317、错配−1.237807；比同价值GREEDY+6.125619，但该贪心控制本轮明显较弱，单赢它不能替代未训练与错配检验。MORE最终macro落后主方法7.218990，但finaltail仍略高0.217890。对BALANCED平均macro−3.397221，最终macro−2.291997。','','## Ordered旧尾类改善了吗','','本轮主方法finalold-tail43.617019%；相对BALANCED+9.497149pp，meanold-tail+16.382334pp（有效2阶段），不再是R106对均衡规则的负差。但对ZERO finalold-tail仍−9.741014pp，mean−9.027055pp；对同价值GREEDY final+0.297261而mean−2.198887。改善是参照相关的，不能声称旧尾保持问题已被RL解决。','','作为历史描述，R106主方法finalold-tail38.551066%，R107为43.617019%，绝对+5.065953pp；但所有控制的集合构造、价值监督与演员伪环境均有明确变化，不能把跨run差值归因于单一模块。R107的同实验ZERO对照比跨run绝对提高更有归因意义。','','## 全五阶段宏差／尾差','','每个单元为GSPO_SET_LT−参照macro/tail百分点。N/A为空尾组。','','|流|阶段|vs BALANCED|vs ZERO|vs SHUFFLED|vs GREEDY|','|---|---:|---:|---:|---:|---:|']
for k in ['a','b']:
 for t in range(5):
  rows={x['arm']:x for x in st if x['stream']==k and int(x['stage'])==t};m=rows['GSPO_SET_LT'];vals=[]
  for ref in ['BALANCED','ZERO_ACTOR','REWARD_SHUFFLED','VALUE_GREEDY']:
   d=float(m['macro'])-float(rows[ref]['macro']);tail=f"{float(m['tail'])-float(rows[ref]['tail']):+.6f}" if m['tail'] else 'N/A';vals.append(f'{d:+.6f}/{tail}')
  lines.append(f'|{k}|{t}|'+ '|'.join(vals)+'|')
lines+=['','## Tiny原分母贡献','','完整三类tiny仍计入主macro/tail，分母与正确数见TINY_CLASS_STAGE.csv；下表不删除类别，只拆分最终macro差值的原23类分母贡献。','','|流|参照|final macro差pp|非tiny原分母贡献pp|tiny原分母贡献pp|','|---|---|---:|---:|---:|']
for k in ['a','b']:
 for ref in ['BALANCED','ZERO_ACTOR','REWARD_SHUFFLED','VALUE_GREEDY','VALUE_GREEDY_MORE']:
  x=next(x for x in ti if x['stream']==k and x['stage']=='4' and x['reference']==ref);assert abs(float(x['full_macro_delta_pp'])-float(x['non_tiny_original_denominator_contribution_pp'])-float(x['tiny_original_denominator_contribution_pp']))<1e-10
  lines.append(f"|{k}|{ref}|{float(x['full_macro_delta_pp']):+.6f}|{float(x['non_tiny_original_denominator_contribution_pp']):+.6f}|{float(x['tiny_original_denominator_contribution_pp']):+.6f}|")
lines+=['','本轮对ZERO/错配的最终macro差值全部来自非tiny部分；负结果不能解释为仅差一张tiny图像。Ordered对GREEDY负差中tiny占−2.173913pp，其余20类按原分母合计仅+0.090487pp，进一步说明不能选择性强调单一汇总指标。','','## 策略与匹配检查','','保存状态核验：ZERO最后层权重/偏置始终为0；全部160组奖励错配为无不动点置换；所有主方法训练组优势非零、梯度非零、每组4条末端记忆均不同。所有记录GSPO裁剪仍未激活，不能据此要求调参直到裁剪生效。','','Ordered中SET与SET-LT的策略参数已有差异，但五阶段正式记忆与预测完全相同，故本轮没有呈现长尾伪环境对正式选择的增量。order0自阶段1起正式记忆不同，但SET-LT对SET平均macro−0.700072、最终macro+1.343509、最终tail−0.542205，方向混合。不同环境的梯度存在不等于已发生有利迁移。详见SAVED_POLICY_AUDIT.json。','','这轮的新关键证据是：未训练合法随机策略很强，现有长期回报训练没有稳定胜过它或错误配对回报。它否定的是当前冻结设置下的“已证明学到更好保留规则”解释，不是所有RL证据管理都不可能有效；也不能通过继续只比较较弱确定性规则来回避该结果。','','## 正确性、字节与费用','','两worker和controller均exit0，70分类器及10策略/价值端点在统一VAL前封存；两流各10500正式分类更新，共21000。saved-only重建macro/micro/tail最大误差分别2.13e−14/2.84e−14pp；所有模型/优化器finite，70个实际记忆及父依赖/类覆盖/旧token属于自身上一记忆/未见统计空值均通过。TEST0。每臂528568B上限；ID无后缀限制及强制动作排除检查通过。源码所披露的确定性语义并列tie break限制仍保留。','','物理科学optimizer130120：影子103680、正式21000、actor960、价值4480。工程两worker共2744optimizer，另有1次独立CPU合成classifier+非空AdamW序列化检查，共2745；该外部检查在native准入后、正式初始编码期间执行，不当作原生GPU前置检查。两流1920影子轨迹、111705非平凡动作采样，两次复用更新为223410动作更新暴露，不是额外采样。原生/影子前缀/CF共享计算实际只计一次。','',f"两个worker已采样CPU共{sum(v['worker_CPU_seconds'] for v in X['workers'].values()):.6f}s，controller{X['controller_CPU_seconds']:.6f}s；controllerwall{X['controller_wall_seconds']:.6f}s（{X['controller_wall_seconds']/60:.2f}min）；原FIRST到worker/controller退出{X['original_elapsed_seconds']:.6f}s（{X['original_elapsed_seconds']/60:.2f}min）。峰groupRSS{X['peak_sampled_group_RSS_bytes']}B，退出新增磁盘{X['allocated_bytes']}B；保存后的审计/文档小幅增加另计。",'','core与native/profile费用已在worker内，不重复相加。独立CPU恢复检查CPU1.128562s另列；savedpolicy审计coreCPU/墙钟见其JSON。重复软件检查、导入、SSH、文档、发布CPU非0未完整采样；没有GPUkernel时长，不以保守预算差称方法效率提升。等待用户询问与保存后结案的原钟另在最终私有收据披露，不重置FIRST。','','## 本轮结案','','COMPLETE_NEGATIVE_FOR_LEARNED_POLICY_INCREMENT：训练工程完整成功，主方法未通过未训练/错配必要归因证据；这是完整结果而非工程失败。只有单开发seed、多轮开发VAL，不能推断统计显著、患者独立泛化或创新成立。全部七臂与正负阶段保留，不改主候选、不调参或恢复端点。监测保持OFF，本轮交付后停止，不自动追加新实验。']
(R/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')
pre='> 完整结论：主方法未超过未训练策略或稳定超过奖励错配；Ordered旧尾相对均衡改善，但仍逊于未训练策略。见[完整分析](ANALYSIS_ZH.md)。\n\n';report=(R/'REPORT_ZH.md').read_text()
if not report.startswith('> 完整结论'):(R/'REPORT_ZH.md').write_text(pre+report)
C['controller_exit']=X;C['measured_workers_plus_controller_CPU_seconds']=sum(v['worker_CPU_seconds'] for v in X['workers'].values())+X['controller_CPU_seconds'];C['science_optimizer_total']=130120;C['engineering_worker_optimizer_total']=2744;C['external_CPU_engineering_optimizer']=1;C['external_restore_check']=read('ENGINEERING_RESTORE_CHECK.json');C['saved_policy_audit_CPU_seconds']=P['saved_only_CPU_seconds'];C['GPU_kernel_seconds']=None;write('COSTS.json',C)
V.update(saved_policy_zero_and_derangement_audit_passed=True,new_closeout_forward_optimizer=0);write('VERIFICATION.json',V)
write('CLOSEOUT_VERIFICATION.json',{'passed':True,'saved_only':True,'new_encoder_forward_optimizer':0,'classifier_endpoints':70,'memory_endpoint_rows':len(A),'policy_value_endpoints':10,'formal_classifier_updates':21000,'all_worker_exit0':True,'main_not_changed_after_results':True,'TEST':0,'monitor':'OFF'})
print('saved-only closeout passed')
