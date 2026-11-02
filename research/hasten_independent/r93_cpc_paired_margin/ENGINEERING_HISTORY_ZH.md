# R93 最新状态：等价实现优化后预算通过，正式训练已启动

2026-11-02。用户明确要求优化实现且不改方法、训练步数或对照，通过预算即直接启动。本轮沿用原FIRST/deadline，没有以新实验名重置24h；以下旧预算拒绝记录保持历史，不代表最新运行状态。

## 唯一性能修复与等价证据

旧pair函数每支持对、每类重新扫描全部S，用Torch标量标签筛选成员；当C23/S6821时，该重复元数据工作显著放大每次梯度更新的wall时间。现在只在每个不可变stage数据上一次分组/排序，缓存class→有序ID列表。支持采样、成员和顺序、m与offset、双侧state、网络/损失/精度/梯度/优化器、lambda0/1和6144steps全部不变。

六stage有序池与原定义相同；6264个训练/评估配对情境的成员选择一致；两端stage共40组双侧state/p0/coverage比较完全相同；lambda0/1和cap4/FULL完整参数梯度检查的state/loss/grad最大差均为0。等价检查是工程验证，不是效益结果，未进行正式optimizer/encoder/VAL。详见OPTIMIZATION_EQUIVALENCE.json、METADATA_CACHE_AMENDMENT.md。

## 同口径预算

五cap固定第二pass取最慢，全尺寸C23/B1713/S6821。最坏pair从0.912116秒降到0.136254秒（约6.69倍）；最坏合成eval-context从3.568342降到0.516917秒。缓存构建首次成本包含第一pass，六stage一次构建由原3600秒准备/保存/分析余量覆盖。safety4、全部6144×4pair、eval120context和AdamW开销不减。

新全计划保守投影17431.130150秒（4.842小时），检查时原钟剩余82903.641615秒（23.029小时），准入通过。估计不是完成保证，独立守卫继续约束原deadline、CPU4、RAM、磁盘和共享锁。未缩epoch/更新/held模型/对照，不使用GPU或新图像编码。

## 已确认正式启动

正式attempt从原seed重新初始化CONTROL/MARGIN×main/held全部模型及AdamW，不复用合成优化参数、不恢复任何已学习optimizer。启动后STATUS BUILD，stage0 CONTROL/main已128正式更新，pairbackward512/supportepisode1024，actor/encoder/VALTEST0，进程入口中性，资源正常。这是启动快照，不是最终计数或效果。

工程attempt3 workerCPU41.853483秒/wall32.215002秒，峰RSS606658560B/group593186816B；其中equivalence CPU29.271573/wall21.168596、optimized profile CPU5.544989/wall1.491653已包含，不重复加。前三attempt合计workerCPU78.453886秒；工程真实TRAIN pairbackward累计10（原native2+等价8），合成profile pairbackward20、合成optimizer2，另最初软件合成backward单列。正式worker后续费用另加，当前正式counter不包含旧工程attempt。准备/导入/SSH/guard/发布CPU非0未全采样。

结果仍未知。完成需6144steps/24builder阶段端点/120固定F情境每arm、原稳定性与识别/tiny门、所有正负及成本交付；0actor/VALTEST。协议中的F是开发TRAIN反馈，不称独立验证，FULL S不等于全TRAIN部署。未宣称GRPO收益或最终创新。

---

## 以下为优化前工程拒绝及完整历史收据

# R93 配对支持间隔一致性：工程预算不准入

2026-11-02。独立候选工程检查正常exit0，状态NOT_ADMITTED_BUDGET。新数学软件及真实TRAIN原型模块梯度/恢复检查通过；事前保守预算未通过，正式构建器训练没有启动，不是方法效益负结果。R91/D92保持关闭。

## 本次候选和冻结比较

从相同原始初始化训练两个监督构建器：lambda0配对对照与lambda1配对margin一致性，原R91结构/CE/原型辅助/学习率等不变。相同实际支持数、不同合法支持成员，两个支持集构建全类原型；在同B查询上约束分类间隔/20的平方差。无actor，NOOP/half/full完整读出，0VAL。main与永久held两个构建器、六stage，每model256updates、每update4pairs，共6144正式优化步计划；同预算两臂，不缩步数/删held救援。完整冻结方案见PROTOCOL.md。

该项是通用一致性正则的机制诊断候选，不宣称新颖性、GRPO改进或顶会主线已成立。问题是支持成员扰动的分类边界稳定性，而非增加alpha档位或重启旧graph/metric/transport分支。

## 检查结果

- 软件：同一支持对一致性精确为零、逐query共同全类分数平移不变、类别排列等变、双侧有限非零梯度，PASS。
- 实际TRAIN：固定初始stage cap4支持对，两次完整forward/backward，参数未更新；梯度有限、非零，strictrestore PASS。0新encoder、0optimizer、0VALTEST。
- 第一次pair 0.182750463s，预定第二次 0.172506094s；不是取最小。按初始C13/查询行数外推最坏C23/B1713，sizefactor 5.454724398，再safety4、全6144×4pairs及3600s分析余量。
- 预计全计划 **96101.429410s（26.695小时）**，检查时剩余 86261.492748s（23.962小时），不满足原24h预算，因此未正式fit。

该估计是保守尺寸外推，不是测得该方法必需26.7小时，也不能据此声称科学不可行。当前没有通过原完整预算，不放宽safetyfactor或改epoch/控制来制造准入。若后续有明确工程计时假设，必须在0正式science边界单独记录同原clock的修复范围和全部成本，不能悄悄替换本收据；目前没有执行新profile或重启。

## 实际计数和成本

正式builder更新0，pairbackward0，supportepisode0，正式查询训练行0，actor0，evaluation0，encoder0，VALTEST0。工程真实TRAIN pairbackward2（每pair两侧builder前向）；软件另有合成双侧梯度检验，非正式优化。

workerwall 12.069312s、CPU 8.885831s、peakRSS 478081024B、2s采样grouppeak 470773760B，disk 65847B；原clock到进程退出 141.608445s。原生/软件/计时均含worker，不重复加。准备实现、SSH、文档、guard及发布CPU非0未完整采样。没有模型效益指标，没有已训练参数端点可供比较。工程收据和失败准入证据均保留。

## D92解释勘误（数值不变）

此前D92报告“F选相对uniform仅9/30正”应准确读为“B选出的动作在F上仅9/30情境U增益为正”。真正F oracle是同域最大值，必不低于均匀均值。D92自身指整个held组同时变化而非逐个真实类别原型。新分享分析核实FULL S下仅held组修正宏6/6负、全类共同宏6/6正；总体均值不能外推稳定独立原型去噪。

## 后续边界

不需要用户逐轮确认，持续研究授权保留；但单测通过不等于预算通过。当前未启动正式训练，不能写CPC新方法正在fit。不得以结果选择支持cap/块、增加GRPO步数、恢复R91或更改原门补救。安全源码、协议与工程预算摘要交付；无私有数据/图/逐例/特征/模型/优化器/rawlogs/privatepaths公开。


## 后续同钟直接最坏尺寸计时（仍未准入，最终工程结案）

在0正式science边界事前固定DIRECT_WORST_BUDGET_AMENDMENT.md：不改lambda、模型、6144更新、角色、门或安全4，直接测最坏C23/B1713而不再从C13外推。合成S6821行（大于实际S3405），每cap两pass，固定用第二pass再取五cap最大，不挑最快。无F/VAL读取。

第二pass五cap完整pair耗时依次0.828478/0.837918/0.837655/0.840454/0.912116秒；合成AdamW单步0.007998秒、4模型完整合成eval-context3.568342秒。全计划加安全4及3600秒余量投影95173.983948秒（26.437小时），剩余84553.820684秒（23.487小时），仍不准入。它是该实现下的保守全计划投影，非已消耗26小时。保留首估计和拒绝，不选择更小计时、减少安全或步数。按附注不再重复此profile来强行准入。

attempt2正常exit0，工程合成pairbackward10、合成optimizer1、合成eval-context1，全部单列，不属于正式训练；正式optimizer/encoder/VALTEST仍0。workerCPU27.714572秒/wall24.114175秒，内部profileCPU19.370618/wall12.293178含worker，不重复加。两attempt worker累计CPU36.600403秒，原clock至attempt2退出1849.164866秒；峰RSS547835904B/group557608960B、磁盘147269B。准备、SSH、guard、发布非0CPU未全采样。

NFS归档copy2在内容复制后附加xattr时报EIO；原campaign内容保留，其他旧工程收据使用不复制xattr的copyfile保留。未删除历史或模型数据。读取已退出PID的ps返回无进程，不是实验故障；没有重启旧worker。

代码可见pair每次对全部S逐类重建成员池，可能是可优化的纯元数据开销，但本轮未以新计时继续调整准入，未声称已证明该开销占比或已加速。若未来做工程优化，应验证与固定配对IDs/state/梯度等价，不能改变样本/方法配置或以新编号绕过本轮预算。

实际结论保持：R93没有正式训练或效益结果；本次通用配对一致性候选在现冻结实现/保守预算下未准入，非科学无效。所有原收据及成本保留。
