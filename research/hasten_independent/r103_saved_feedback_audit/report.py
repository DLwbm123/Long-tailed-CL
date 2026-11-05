"""Render fixed descriptive conclusions from aggregate CSVs, never refit anything."""
import csv,json,time
from pathlib import Path

def run(root):
    root=Path(root)
    def rd(name):
        with (root/(name+'.csv')).open() as f:return list(csv.DictReader(f))
    def pick(rows,**kw):return [r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
    def num(x):return None if x=='' else float(x)
    zeros=rd('ZERO_GROUP_DECOMPOSITION');pattern=rd('CANDIDATE_PATTERN_SUMMARY');trans=rd('OOF_FULL_UTILITY_TRANSITIONS');gain=rd('GAIN_CONCENTRATION');contrib=rd('CLASS_GAIN_CONTRIBUTIONS');uniform=rd('UNIFORM_VS_CANDIDATE_PATTERN');routes=rd('ROUTING_DECISION_QUALITY');unique=rd('UNIQUE_EVIDENCE_COVERAGE')
    lines=['# R103 封存反馈、候选互补性与迁移审计','','本轮为已结束R103的saved-only后验机制分析，科学来源ec5502cf886e7e453e98609f9fcd6dde0df6d5fd。不是新性能实验、独立复现或因果干预。R103主候选GRPO_CF及原有负结论不变；没有R104训练、模型前向、图像/GPU/TEST读取或监测。','', '## 一页核心结果','','**建议：暂停继续修改三选一路由/GRPO配方，先将固定多视图复现写成待执行方案。原因是奖励稀疏主要具有结构性，OOF相对效用变化多是消失/出现而非收益反转，固定UNIFORM的正结果有别于单例驱动的MULTI，但也仍集中于小分母，且Ordered未成立。**','','### Q1：全同组主要来自哪里？','', '下表是实际历史抽样的聚合重建，不是端点替代历史。每流每GRPO臂81920组，按128outer×128query×5stage去重；同组4次内更新只算一次。结构性由保存查询计划与候选正确性确定，总全同比例由每outer保存的精确聚合日志确定，采样性为两者差额。不能恢复哪一条informative查询恰好抽到同奖励动作，也不能计算真实旧策略下的逐组解析期望。','','|流|臂|历史全同组率|结构性组率|采样性组率|结构性占全同组|','|---|---|---:|---:|---:|---:|']
    for stream in ['a','b']:
        for arm in ['GRPO_CF','GRPO_IN']:
            rr=pick(zeros,stream=stream,arm=arm);st=sum(float(v['structural_fraction']) for v in rr)/5;sa=sum(float(v['sampling_zero_fraction']) for v in rr)/5
            lines.append(f"|{stream}|{arm}|{100*(st+sa):.4f}%|{100*st:.4f}%|{100*sa:.4f}%|{100*st/(st+sa):.4f}%|")
    lines += ['','a=Ordered，b=分层order0。GRPO_CF全同组中结构性分别占95.55%/93.73%。采样性仍占全部组4.28/6.04个百分点，不能说采样完全无关；但仅增大G无法改变全对/全错候选。这是观察计数分解，不是未保存old policy的条件期望分解。固定初始q_ref的解析参考另列CSV，明确不是训练历史或最终策略。全同组只令GRPO surrogate为0，外置KL/解析损害仍可能有梯度。','', '### Q2：多少唯一图像/分量真正提供了正确性差异？','','|流|最后阶段OOF差异图像|自然频率占比|类别等权占比|五阶段OOF差异图像去重并集|','|---|---:|---:|---:|---:|']
    for stream in ['a','b']:
        p=pick(pattern,stream=stream,stage=4,role='TRAIN_OOF',group='all',pattern='INFORMATIVE')[0];u=pick(unique,stream=stream,role='TRAIN_OOF')[0]
        lines.append(f"|{stream}|{p['count']}/6821|{100*float(p['micro_fraction']):.4f}%|{100*float(p['class_balanced_fraction']):.4f}%|{u['informative_images']}|")
    lines += ['','当前保存的identity_component在这些数据中与图像一一对应，所以表中图像与可辨识分量数量相同。它不是患者ID证据，不能称这些都是独立患者。两个流使用同一批TRAIN/VAL，不能相加成独立队列；跨stage图像不重复计为新证据。UNIQUE_EVIDENCE_COVERAGE另报FULL/VAL去重并集。','', '### Q3：OOF→FULL相对效用变化主要是什么？','','最后阶段同一6821 TRAIN查询、各bank自己的教师基线：','','|流|候选|真正符号反转(±1↔∓1)|效用消失(±1→0)|效用出现(0→±1)|正收益消失中教师也变对|正收益消失总数|','|---|---|---:|---:|---:|---:|---:|']
    for stream in ['a','b']:
        for action in ['CENTER','MULTI']:
            rr=pick(trans,stream=stream,stage=4,group='all',action=action);total=lambda kind:sum(int(v['count']) for v in rr if v['change_type']==kind);gone=next(v for v in rr if v['r_OOF']=='1' and v['r_FULL']=='0')
            lines.append(f"|{stream}|{action}|{total('sign_reversal')}|{total('disappeared')}|{total('appeared')}|{gone['teacher_wrong_to_correct']}|{gone['count']}|")
    lines += ['','收益消失中约85%—87%对应教师从错变对、局部候选保持正确，不是局部候选由对变错。不能把所有机会集合变化叫排序反转，更不能仅凭机会重合低就认定换bank是VAL失败主因。变化同时含上游拟合规模、查询是否参与拟合、置信度等差异。候选优劣次序的含平局3×3矩阵与逐类/旧新头尾净效用全部公开。路由OOF→FULL概率L1与动作变化只有原聚合统计，另行引用并标aggregate_only，未用缺失概率重新构建。','', '### Q4：UNIFORM是否能在三个候选全错时判对？','','能，但本批结果有明确范围：10个VAL stage中，Ordered第4阶段发生1次，属于new classifier17/legacyC15；这一模式该类有3张，UNIFORM判对1张。两流最终阶段的000模式中都没有救回。TRAIN表也存在少数类似记录。这个观察证明硬三选一Oracle不是概率融合的普遍上界，但不能拿它解释分层最终UNIFORM的全部收益。全部111模式中UNIFORM均正确，未发现需要用并列/数值解释的异常。VAL没有保存候选logits，故只能核保存预测关系；仅在TRAIN有logits处重算原等权概率规则。','', '### Q5：多视图收益分散还是小分母主导？','','|分层最终读出|总正贡献pp|总负贡献pp|净macro差pp|C1贡献pp|','|---|---:|---:|---:|---:|']
    for mode in ['MULTI','UNIFORM','GRPO_IN','GRPO_CF']:
        v=pick(gain,stream='b',stage=4,mode=mode)[0];lines.append(f"|{mode}|{float(v['positive_sum_pp']):+.6f}|{float(v['negative_sum_pp']):+.6f}|{float(v['net_macro_pp']):+.6f}|{float(v['legacyC1_contribution_pp']):+.6f}|")
    lines += ['','分层最终UNIFORM对C1仍0/1，其+3.234387pp不能归因同一单例。逐类真实配对如下：','','|legacy类|T正确数|UNIFORM正确数|救回|损害|原23类分母macro贡献pp|','|---|---:|---:|---:|---:|---:|']
    for cl in [13,16,20]:
        v=next(v for v in contrib if v['stream']=='b' and v['stage']=='4' and v['mode']=='UNIFORM' and int(v['legacy_index'])==cl);lines.append(f"|C{cl}|{v['teacher_correct']}/{v['count']}|{v['method_correct']}/{v['count']}|{v['rescue']}|{v['harm']}|{float(v['macro_contribution_pp']):+.6f}|")
    u=pick(gain,stream='b',stage=4,mode='UNIFORM')[0]
    lines += ['',f"这三类配对确认救回4张/损害0，合计+{float(u['legacyC13_C16_C20_contribution_pp']):.6f}pp，其余类别合计+{float(u['net_macro_pp'])-float(u['legacyC13_C16_C20_contribution_pp']):.6f}pp。约占净增益{100*float(u['specified_share_net']):.2f}%，但占总正贡献{100*float(u['specified_share_positive']):.2f}%，两种分母不能混用。全体类别则救回{u['total_rescue']}张、损害{u['total_harm']}张，净多正确15张，并非只有这4张变化。仍是少量小分母图像证据，分量可辨识不等于患者独立。",'', 'MULTI/GRPO_IN的C1单例贡献均+4.347826pp，按原分母扣除后其他类别分别+0.478726/−0.249709pp；不是删除类别重算主指标。Ordered最终UNIFORM仍−0.364861pp，不能隐藏。所有正负类及最大1/3类占总正贡献比例见CSV；净差非正或很小时，指定类占净差比例不作稳定增益份额解释。','', '### Q6：下一步最有依据的投入？','','暂缓三候选GRPO配方调整和新序贯训练。先将T/CENTER/MULTI/UNIFORM固定多视图配对复现作为唯一候选后继，仅形成未启动方案。现有证据支持局部互补信号存在，不支持它已跨流、跨优化重复稳定，更不支持采样策略梯度必要性。即便未来固定融合复现成功，也不能改写R103主候选或将其归因GRPO。','', '## 路由质量（最终VAL，保存动作）','','|流|臂|选局部|局部且改变教师类别|救回|损害|可救机会利用率|','|---|---|---:|---:|---:|---:|---:|']
    for stream in ['a','b']:
        for mode in ['SUP_CF','RAW_EXACT_CF','GROUP_EXACT_CF','GRPO_CF','GRPO_IN']:
            v=pick(routes,stream=stream,stage=4,group='all',mode=mode)[0];rate='N/A' if v['rescue_utilization']=='' else f"{100*float(v['rescue_utilization']):.2f}%";lines.append(f"|{stream}|{mode}|{v['local_selected']}|{v['local_prediction_changed']}|{v['rescue']}|{v['harm']}|{rate}|")
    lines += ['','选局部不一定改变类别，改类别不一定改善。救回与损害图像数不能直接代替macro，原分母贡献另报。GRPO和GROUP_EXACT分层五stage预测全部相同，Ordered最终相同、中间stage1/2不同6/26张；不扩展为参数/概率/轨迹相同。','', '## 验证范围、缺失和成本','','全部90已保存读出的micro/macro/tail/old-tail、逐类计数、空组和原有效stage均值分母已独立复核。8模式互斥穷尽、微观/类别等权统计、3×3原计数及类别加权/边际、action-order含平局、rescue−harm与净正确数及macro贡献恒等式、000/111融合关系、327680组四次复用去重均通过。','', '验证状态是“全部可做分析完成，缺失字段覆盖partial”，不是无条件全量passed。没有保存逐行训练后router概率、真实old概率或逐组采样动作/零组标志；不加载权重forward补齐。原训练历史条件期望和训练端点逐行Pzero无法提供。固定已知q_ref解析结果仅为参考，不冒充真实初始FP32模型输出或训练轨迹。历史lineage依赖封存运行审计，当前ID/分量排除另已复核，不重演原训练。','', '新optimizer/backward/model_forward/image_read/encoder/VAL_model_scoring/TEST均0。确实读取保存的VAL预测和标签做后验解释，不能说完全未接触VAL。旧F是训练反馈，不恢复为holdout。分层最终R102与R103教师1张读出差异的既有记录原样保留；所有新配对使用R103本轮T。','', 'COSTS.json仅记录新增分析/两次验证及元数据处理，历史教师/训练成本只引用原科学commit不重复加。两次验证的第二次是补齐类别加权转移、action-order组、tail/oldtail贡献与机会分母覆盖，不改变原分析或科学预测。准备/文档/SSH/发布CPU非0未完整采样。CPU-only最多2计算线程，无新GPU读取，输出小于512MiB。','', '可直接观察的是表内模式/效用/贡献，有限解释是反馈不互换或信息稀疏，不能判断唯一VAL失败原因、临床泛化、未来多seed性能或RL普遍有效性。完成后停止，monitor OFF，无后继launcher。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();run(args.output)
