# CT11-H：固定Task4神经头与解析读出诊断

CT3–10均已STOP；本研究独立，不重启历史worker。根据CT10-O的结构事实，检验固定模型上的训练神经分类头是否表现出与解析ridge不同的当前类适应/旧类保留。双方都有可能较差，不预设头优于ridge，不把该诊断当作新训练方法。

固定模型：CT6 beta10、CT9 Task4 beta1，HK/ISIC×原seed/order1993/1994/1995，共12个已锁Task4末状态。仅8已见类，HK是8/23前缀，beta1也仅作用Task4。原正常Task1，无独立S0。

新增N10/N1各6个预测单元，均原神经head+head_few raw sum、全8类，无HN/温度/校准/最佳head/ensemble。复用CT6 C2/C3及CT9 B1A/B1T共24个原ridge预测；总36阶段/288逐类行。主要诊断对比N10−C3、N1−B1T；固定次要N10−C2、N1−B1A、N1−N10。报告BA、old/current及restricted、HM、tail、零召回、误分与逐类，不按结果改选读出。2000次seed54001类分层component配对bootstrap，固定三个模型不重抽seed，所有对比未作多重校正。

数据：仅原val、各seed已见8类、sample_id升序、batch48/drop_last=False。所有读出使用同一个既有eval probe（两pool.batchwise_prompt=False、原batched linear），不是重新运行原CE训练的batchwise=True路由。标签仅在前向后参与指标；train/旧fit/未来fit/test/reserved/oracle图像及特征访问0。不拟合统计/W，不训练，不更改模型/优化器。

全部12父状态先通过原普通restore与hash、dataset/seed/task/epoch/known/seen、beta/源码/原协议、RNG/loader/synthesis核验。CT9恢复保留原e7f9源码与原PROTOCOL_LOCK字节，不把新诊断源码伪装成训练源。新的DIAGNOSTIC_LOCK独立绑定12状态、24预测、24W及代码/协议。工程合格后锁定再统一评价。相同前向的联合特征乘原A/T W必须复现24个原ridge分数（预设atol=rtol=1e-6，argmax必须完全一致），不允许在val上调容忍或修方法。不一致保留BLOCKED_REPRODUCIBILITY，已执行访问计入账本。

工程还核验probe标签盲train=False/weight=None、输出8候选且有限、网络state/mode/RNG不变、pointwise标志、布局及标签映射一致；当前调用返回的神经分数原本就由原extract计算，复用该路径，不新增head选择。保存私有新logits/provenance，公开仅聚合及哈希。

单worker，已有4090D；0新neural epoch/optimizer step/checkpoint/analytic fit。GPU无小时截止，仅12状态有限矩阵；CPU报告≤600秒，活动新盘≤1GiB且空闲≥1GiB。逐个拉取checkpoint后严格恢复、验证可恢复源仍在归档，再删除临时副本；所有原始资产不动。源NFS只读，服务仅GET且限固定12状态，600秒传输/660秒ACK，失败不自动重试。预计10–30分钟，实际以回执为准。禁止测试访问与根盘回退。

先工程后执行完整矩阵；真实失败保留已完成和消耗，修复必须独立绑定，不重跑取最好。完成后交付锁、12父恢复审计、24ridge复现、12新预测锁、36/288指标、配对区间、资源和中文结论。NEXT_DECISION=STOP，不追加训练、搜索beta、改变路由或把神经头替换成正式成绩。
