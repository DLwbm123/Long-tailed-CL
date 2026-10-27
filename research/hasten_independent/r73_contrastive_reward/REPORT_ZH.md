# R73：HK74002 三档对比类别奖励的完整负结果

复用R67同一seed六阶段监督checkpoint，无新增监督训练。四分支只改变二元/三档奖励并各有采样/同目标精确期望，全部六stage完成退出0；各stage后训练独立初始化、不跨stage携带RL增量。全部24head封存后才读取真实HKVAL。

三档reward：correct+1、固定SFT相反fold视觉原型最相似错误类-1、其他wrong0。每stage两fold各一张非自身hardnegative表，训练开始前固定并保存，不读VAL/TEST/futureheads或随postmodel更新。G8保持原oldpolicy有放回采样，不强制positive/不重采样；group-popstd优势和PPOclip不变。Exact对所有45个三类reward-count组合及类内old条件动作概率积分同一clipped normalizedgroup目标；CPU6561组穷举loss/gradient匹配，binaryexact与旧binomialexact相同。不是weightedCE或额外rewardmodel。

| 方法，百分比 | 阶段平均BA | 最终BA | 最终尾类BA |
|---|---:|---:|---:|
| H0 |55.630100|56.454053|36.857416|
| 共同监督起点 |66.840041|64.876782|44.924938|
|binary_grpo|66.115954|63.655574|44.737644|
|hardnegative_grpo|65.776356|63.472624|44.482559|
|binary_exact|65.456019|63.954743|44.161723|
|hardnegative_exact|64.852939|62.510861|40.726164|

预先指定 hardnegative_grpo 对原binary_grpo平均/最终/尾类-0.339598/-0.182950/-0.255085个百分点；对SFT-1.063685/-1.404158/-0.442379个百分点。采样主候选失败，没有通过增加wrong动作奖励区分而取得收益。

hardnegative_exact 对 binary_exact 平均-0.603080、最终-1.443881、尾类-3.435559个百分点，下降更明显。主候选虽高于其exact控制，不能据此称胜过原方法或监督起点。原binary两对照同seed指标与历史一致，不是新seed验证。

| 采样分支 | 零优势组率 | 全错但有优势的组数 | 正确动作数/467472 | hardwrong动作数 | TRAIN平均KL |
|---|---:|---:|---:|---:|---:|
|binary_grpo|74.7578%|0|418725|24403|0.046914|
|hardnegative_grpo|74.3043%|415|419781|23936|0.037232|

三档采样415/58434组（0.7102%）全部动作错误但存在非零优势，二元为0；干预确实产生了原目标没有的梯度信号。整体零优势率仅从74.7578%降到74.3043%，不能把全部历史zero组视为allwrong。三档reward_sum395845=419781correct-23936hardwrong；二元418725=correct数，所有动作三分类counts和奖励恒等式已核。三档exact平均TRAINKL0.023516，binaryexact0.019433；exact没有采样组，其0计数不能当zero组率0。更多informativewronggroups不足以说明梯度方向有利，且固定几何近邻不等于临床语义。

六stage每个两fold表共12张均核非自身/有效class范围/固定TRAIN来源；模型原参数冻结、49152LoRA、strictrestore/fold、固定TARS权重、old/ref、未缩放KL、raw推理、封存和真实VAL分母均核。saved-onlyverify→analyze通过944optimizer/3696micro backward/24branch72curve30真实VAL读数（24新+6只读SFT），0新模型评分/优化。奖励保留原TARSgamma.75/clip.2/KL.04/.01anchor/自然query/naturalpolicy/2epoch及其余冻结设置，无方向保护或逆系数补偿。

原钟2587.522530秒（43.13分钟），CPU3548.444895秒，workerRSS峰7.438GB，保守聚合峰7.460GB，新增磁盘峰55.628MB。三类别exact的45count组合计算费用包含在内，不假称等同旧二项exact计算量。仅A1000/1，30GB内存限制内；root/remote prefix交换实际0，losslessFP32parity最大2.2352e-7，域末清声明再生prefix、全部学习产物保留。未恢复用户停止的R65，历史SFT1180/4620只读来源不计新增。

单HK单seed反复历史开发、极少类VAL分母可1或2，不能三域BOTH/三seed/TEST/独立患者或临床确认。采样/精确均未支持此奖励改进，不更换主候选、不追改门/挑域seed/仅报高于H0，不扫rewardstrength/G/LR/epoch/weightcap救援。与此前R67–R72负结果共同保留，后继必须重新综合判断主线，而非持续声称少一项工程修复即可使GRPO有效。

公开自有source/protocol/全stage-class计数/曲线/固定hardnegative表/诊断/费用和正负报告；不公开原图/身份/逐例数组/features/token/权重/原日志/私有配置。
