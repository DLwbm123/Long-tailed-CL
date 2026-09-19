# CT9-P：固定FD系数单任务干预

独立于CT3–8已STOP研究，参考CT6 344d87d、CT7 f83bf4d、CT8 8aff618。用户授权自主有限后续实验；本计划先固定范围，不因结果选择其他beta。

假说：同一CT3-P Task3末状态起，将FD系数由10降为固定1，可增加Task4适应，可能同时恶化旧类记忆。该系数是预先固定的一数量级干预，不由val或梯度比自动计算，不搜索。研究只检验条件于已用beta10训练的Task1–3前缀的单任务效应，不是beta1全程方法。

矩阵：HK与ISIC，各原seed/order1993/1994/1995，六条Task4续训，每条10epoch；60task-epochs、2750正式steps（HK140/10/120；ISIC1280/600/600），6新末checkpoint。正常Task1、无独立S0，不重训前缀，不冻结adapter。HK只到8类，不得称完整23类Final；ISIC为8类终端，但新系数仅Task4生效。

新读出B1A=beta1+原A、B1T=同轨迹+原T；固定控制CT6 Task4 C2/C3（beta10）和CT5 Task4 F1。12新预测单元/96逐类，含三控制30/240行。主B1T−C3；次B1A−C2、B1T−F1、B1A−F1、B1T−B1A。不是依据结果选择最佳bank。主BA、old/current/HM/tail、逐类/零召回/最差seed、误分与梯度审计。2000次seed52001按类component配对bootstrap，固定三模型不重采样seed。报告全部正负结果及区间，不以任一seed好坏终止矩阵，不选择epoch。

先通过原FDStudent.restore_new严格恢复六个beta10 Task3父状态，验证模型、args、原源码、manifest、RNG、loader/synthesis、A/T计数与W。显式fork后仅新FD系数变1，保存新的checkpoint schema与源码/协议绑定；普通新状态restore必须拒绝错误beta/源/协议，不能把beta10标签留在新状态中。Task4教师是对应Task3自身副本，无梯度且独立。FD函数、同增强张量、pointwise路由和可微student保持原实现。原real CE范围、reduction、/3、assignment、pull、theta、batch48、采样器、优化器/调度、预处理、10epoch不变。无ConCM/Risk/新模块。任务末A/T都从各自immutable pre-task银行出发，只提交一次，不反馈训练。

train仅当前Task4 fit（remapped6/7），旧/未来fit/test/reserved/oracle访问0。所有6checkpoint与12W锁后统一val，沿用原sample_id升序/batch48/已见8类候选；不能用val拟合或改beta。报告只读锁定分数。父状态或数据/源码不一致、非有限、资源不满足则准确BLOCKED，不替代父状态、不重训前缀。

工程：六父普通恢复、教师/RNG/A/T检查；CPU微型autograd检查beta10回归与beta1差项、零FD初态梯度与checkpoint拒绝错误锁。实际首正式epoch验证有限loss、adapter梯度、teacher固定，后续每epoch原审计保留。无需额外工程optimizer step。

单GPU worker，预计1–2小时（以实际吞吐为准），用户无GPU小时截止；只60epoch/2750steps。CPU解析/报告≤2小时，活动盘≤2GiB、空闲≥1GiB，新归档≤3GiB。逐父拉取，使用后删除已经验证可恢复副本；6新末checkpoint经my-gpu实际NFS归档确认后删除hb副本。600秒传输/660秒ACK有界，不自动无限重试；保留失败和用时。滚动checkpoint用于明确同锁恢复，禁止盲重跑。既有根/tmp/p22root/p25root等只读，不启动旧worker或旧服务。新中性根/tmp/p28root、入口/tmp/p28.py，my-gpu新/tmp/p28archive服务；使用现有算力，不购买或终止他人进程。

交付新协议/源码/谱系、工程、分项训练与资源账、Task4固定30/240指标与预定区间、中文结论。开发性val，不宣称独立确认；不把梯度相关性当因果性能证据。完成后NEXT_DECISION=STOP，不自动扩到HK Task5–11、不搜索其他beta或再训练新模型。
