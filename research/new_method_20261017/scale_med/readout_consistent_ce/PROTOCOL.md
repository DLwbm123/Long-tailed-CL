# READOUT_CONSISTENT_CE：训练—部署读出一致性的独立实验

2026-10-20持续自主研究授权下，在任何新预检/训练前封存；不是重启或调参救援DECISION_ANCHOR。动机来自已封存READOUT_MARGIN_AUDIT：同D特征F读出23/38、自身读出18/38，5新增损害中2例固定参考已错、3例换读出后错。该诊断仅描述、非因果，不把F_on_D追认成功。本实验明确改变训练/部署关系，非旧fixed-U固定特征读出重参数化；initial_bias/group_out均训练临时原双头后丢弃/拟合ridge，未测试本固定部署读出。研究的是已知监督优化原则，非原创认证。

假设：直接使用部署时相同的固定训练F mapper/初始ridge头/截距，对非线性encoder的既有bias实施CE适配，能比联合训练头及事后重拟合更好保留并改进初始风险。本轮只研究初始2555训练/681反复暴露开发，不读任何future/test/原始图像；不改变原类1未来J门，也不称SCALE-Med整体成功。

## 候选与匹配控制
P唯一主：F mapper、ridge alpha.01的前4类头及自由截距始终固定。训练与推理都用score=mapper_F(g_theta)@W_F+b_F；仅102912既有encoder bias更新，无投影惩罚/新模块/温度/梯度缩放。
C：同F mapper、同W_F/b_F初始化和同CE，但允许W/b联合更新（额外6144个float64研究参数）；推理用该训练终点头及同F mapper，训练后不择head checkpoint。
Q：同P终点encoder，重新拟合自身mapper/ridge，诊断控制不能升主。
R：同C终点encoder，重新拟合自身mapper/ridge，主要读出控制。
F：原冻结G。L：decision_anchor既定全支持保护臂已选原部署输出，强简单历史参照，只复用其初始封存预测/aggregate，不重训编码。
P/C都从相同原历史预训练encoder初始化，不读旧E/I/O/Bias/D/L/R learned encoder作为初始化；只读取旧protection文件中由2555冻结F训练特征产生的mapper/teacher_W（不是学习后encoder）。C初始W从F复制，不先训练头再传P。原native双head一律冻结且不参与此新loss。Q/R不影响训练选择。

## 训练与终点
相同原APART FP32双分支G1535、PERM/层/池/reader像素；原FP32单位化后float64单位化，不改精度或缓存生产。固定共同采样沿已核seed74250，800步每类8不同组各1图=32槽，P/C各25600槽，共51200。沿用此清单用于控制采样差异，不按结果选择seed。micro8 checkpoint、完整32 CE后反传。AdamW lr.003/wd0/betas(.9,.999)/eps1e-8，无scheduler/clip/早停最优或搜索。每100步仅恢复点；所有800步端点先ENCODER_SEAL，不按P/C局部表现停止或改另一臂。P的F读出exact不变；C仅读出参数和同bias允许更新，其他模型参数边界不变。未来encoder永久冻结，但本轮不访问未来。

两个encoder各一次3236行编码后FEATURE_SEAL，不重跑无seal残缺编码。F旧g只读。P/C以F坐标部署，不需重新拟合头；Q/R各在自己2555训练拟合原mapper/A/B并让已暴露681仅选alpha[.0001,.01,1]，同分取大，总6个新ridge解全报。不拟合额外alpha或调整P/C。全部头先MODEL_SEAL才汇总救回/损害/风险。
P/C/Q/R四个完整artifact各<=19058800bytes，含absolute bias 411648bytes、mapper/ids/PERM/W/b/当前坐标三角A/B/元数据。P/C的W是固定参考/训练CE头，**不宣称它由本artifact A/B的ridge正规方程导出**；A/B仍在对应当前特征/部署mapper坐标计算并计入容量。Q/R须正规方程残差<=1e-7与A/B重建回读一致。P/C回读完整分数/偏差override一致，F/C初始头与参考回放，不能只存W豁免持久成本。固定teacher推理只需一个适配encoder，不并行F encoder；训练头和持久头同一对象，无免费额外inference。

## 开发点门与家族
预定P-F BA>=1pp、class2>=5pp；class0/1/3成本各<=1pp。P对C/Q/R/L的BA均严格>0、class2均>=0；P-R class2还>=5pp。点门失败固定STOP_READOUT_CONSISTENT_DEVELOPMENT，不择C/Q/R/L或F_on_D升主、不加lambda/alpha/seed/步数救援。所有点和逐类R/H/去向完整报告，已暴露开发不作独立确认。
仅点门全过才计算P-F/C/Q/R/L各BA与class2共10项组配对bootstrap，10000draw、seed74271，相同draw复用，Bonf分位[.05/20,1-.05/20]，全部调整下界>0才DEVELOPMENT_READOUT_CONSISTENCY_SIGNAL；否则STOP_READOUT_CONSISTENCY_EVIDENCE_INSUFFICIENT。无论如何method_success=false，无future/test/J，不用新class2问题替代原方法目标。

## 执行前鉴定
科学/源码先封存。合成验证固定/可训练头目标和梯度、自由截距/平移等价、风险门成功/失败/缺失、10项JSON、R/H恒等式。相同正式包装器预定失败注入四回执齐全。实际固定32训练图预检：原batch32 G回放<=1e-10；micro8/checkpoint相对完整32梯度atol1e-5/rtol1e-3，披露特征舍入；P/C初始bias梯度一致，C head梯度非零finite；P头梯度关闭且一步后exact不变；C head一步更新、非bias/原双head不变；两臂一步finite、override回读/初始参数恢复exact，probe弃用。完整artifact模板<=cap及回读，真实学习后再核。任一失败停止未决并保留，不自动重试或放宽容差。

## 硬预算与隐私交付
独立首次synthetic至最终退出共享<=3h=10800s，失败/子阶段不重置；此为独立假设，不延长旧D原4h时钟。新增远端+本地/Git/helper16MiB<1GiB，单文件512MiB、worker RSS12GiB；预计<512MiB，旧窗17GB软停/20GB硬限保留。独立协调每2秒核deadline/空间/RSS，超限本轮terminate、5秒kill并保存回执，不干预他人。只GPU0，启动前显存峰值余量和真实NFS挂载/容量/小写读probe；像素/模型/环境复用不复制，所有cache/tmp规定NFS，PYTHONDONTWRITEBYTECODE，主子argv/GPU入口中性。退休PID不监测、不重启。

driver仅正式TRAIN退出0后自动FIT，预检通过才能启动正式；验收2×800/6ridge/4artifact/2FEATURE_SEAL/全部头封存/finite/parity/容量/退出退休。读取多回执，不凭PID/0MiB。费用预检+driver串行活动墙钟，worker wait4包含子代只计一次，不重复正文/父阶段。所有失败/NA和历史缺口保留。唯一scale-med30min跟进，健康静默，不创建第二入口。完成公开源码、冻结协议、全聚合/中文结果/成本失败摘要，GitHub经既定代理push核SHA和匿名HTTP200，私有DELIVERY双端。图像/特征/模型/身份/逐例/原私有日志/第三方PDF不公开或下载。
