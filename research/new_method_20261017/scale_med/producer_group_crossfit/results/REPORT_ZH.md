# 整个监督producer排query组：完成结果

结论：**CONDITIONAL_ADAPTATION_REUSE_GAP_SIGNAL**。预设两项点门和调整区间门均通过，支持在这两个固定producer条件下，适配后响应复用差距大于冻结encoder/仅拟合读出的差距。**它不是性能成功，也不是encoder纯因果或完整CL成功。** method_success=false；不复活旧P/C/D，不追加步数/种子/alpha/保护项。

|响应表|BA %|各类正确数0/1/2/3|
|---|---:|---|
|F_seen|84.991051|259/20/125/1689|
|F_out|64.713626|237/13/64/1626|
|T_seen|99.294146|307/20/169/2012|
|T_out|56.634201|226/6/54/1876|

四表均覆盖同2555行，分母307/20/170/2058。seen由用过该组的producer评分，out由另一未用该组的producer评分；F/T规则相同。原681开发、future/test均未评分。

## 预设机制量与完整风险

- T seen−out BA差42.659945pp；F seen−out为20.277425pp。差分ΔBA=**22.382520pp**，预设组bootstrap调整区间**[15.027638,30.173198]pp**。
- 类2T seen−out=67.647059pp，F为35.882353pp；差分Δclass2=**31.764706pp**，调整区间**[18.867925,45.270430]pp**。
- seed74292/10000draw，两项Bonf分位[.05/4,1−.05/4]，相同类内已知组draw用于四表。只条件于两固定producer，不估训练随机性或未来数据表现。
- **T_out BA56.634201低于F_out64.713626，差−8.079425pp。** 类2 54/170低于64/170（−5.882353pp）；类1 6/20低于13/20（−35pp）；类0少11例，类3多250例。训练端几乎完美不能掩盖未见组的损害。
- T的seen到out类2救回1/损害116，F为5/66；类1T损害14，F损害7。所有类、两个fold、CE、margin分位与R/H在aggregate完整公开，不挑最好fold。

## 与原文、Pro和此前结果的关系

cross-fitted remapping原文§3.3要求整个训练算法排fold；本次从原预训练权重重置，mapper/初始ridge/组权/采样/CE更新全部只用support。两个T各800步、固定终点；所有encoder封存后才一次生成各2555编码，全部读出/响应表封存后才评分，没有把旧全2555监督encoder上的小支持ridge叫OOF。

Pro要求区分保留训练头、固定参考必要性和相对简单控制的竞争力。旧P/C及D已失败，本轮使用同CE终点头，仍观察到明显组外损失，进一步削弱“换部署头是唯一瓶颈”的解释。它与原文的训练响应记忆问题一致，但不是该文GLMC、仿射remapper或理论Gaussian条件的复现。F本身也有20.28pp复用差距，不能把T的42.66pp全部归encoder。

两个fold图数1329/1226而组数逐类相差≤1，类1两半8组、图数12/8。它们的训练量/头/mapper/优化难度不同；四表虽同样本配对，仍不是单个样本纳入/排除的随机干预。原预训练提取链无本任务encoder更新，不认证预训练语料无重叠；这些内部query组在研究历史中并非全新数据。

raw T logit尺度明显大于F，margin及CE不能直接跨头尺度解释为更好表示。SADP的具体中间层/归一化位置仍未唯一，本轮未估其AGOP，不能因本轮信号就直接上方向保护训练。

## 下一步决策

优先区分未见组损害来自“适配表示相对参考读出也损害”，还是“同表示换回support参考读出可恢复”。下一独立CPU诊断可在这两个真正排组的已封存producer上，完整做F/T特征×F/T头的2×2交叉读出，只对其query组评估；F头/mapper仍仅由对应support拟合，无新训练/拟合/方向选择。必须先独立冻结配对表、margin分解、停止门和预算；不能挑单个交叉组合升级性能主候选。此前D的交叉读出只在已反复暴露开发，不替代本次组排除证据。

如果只是校准差异，后续任何校准实验也须独立协议、按组嵌套OOF拟合/选择与明确full-model运输控制；不能直接把OOF两半头接到旧全量C，或把仿射校准包装成原创。若固定参考读出下query风险同样下降，就没有充分依据只做logit修正。当前不自动追加旧P/C训练、remapper搜索或未来扩类。

## 验收、成本与隐私

2×800步/51200监督槽、2个support-onlyridge、2×2555单次FEATURE_SEAL、两完整T artifact各10076821<=19058800、全部ENCODER/MODEL封存/有限值/逐项回读/组排除/R-H恒等式通过。driver6phase只有预定inject exit1，其余exit0，guard未强制，driver/worker/coord/guard均已退休。没有重新编码、评估中拟合或访问开发/future/test。

本次driver包含全部6phase，**不再加PRECHECK_COST**；worker正文不再次加，guard等待墙钟不加。活动墙钟3765.321996s、knownworker/wait4 CPU3856.921073s、协调/driver/guard CPU145.327452s一次并账。历史CPU缺口、准备/网络/报告整理CPU仍未完整计量。

首actual至guard退休3780.189177s<10800，RSS峰4167516160<12GiB；round335358062+16MiB<1GiB，window13618263533+1001040384，报告增长另16MiB<17/20GB。未复制原像素/模型/环境，未删除历史资产。

科学源3d5eff2015b504621ff5670b78b165b5145088ba、协议e92901a；公开源码/协议/设计核查/所有聚合/成本/失败摘要。feature、模型、逐例响应、组身份、原私有日志和第三方PDF私有。DELIVERY完成以双端及远端SHA/匿名报告回执为准。

参考：[cross-fitted remapping原文](https://openreview.net/forum?id=3Fq8NvlSTg)、[SADP原文](https://openreview.net/forum?id=LYEiAGvdJw)、[用户Pro分析](https://chatgpt.com/share/6ad747bb-b77c-83ea-b079-368b6c3552f5)。原文阅读范围、非复现边界详见PROTOCOL.md和DESIGN_REVIEW_ZH.md。
