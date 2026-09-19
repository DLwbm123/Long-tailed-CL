# CT6-F：固定FD的完整任务续训

2026-09-19独立研究，基于CT3-P交付6f6e15f与CT5-F交付2aaf822。用户授权自主管理后续有限实验、取消GPU小时上限。旧研究均STOP，不重训/改写。假说：原beta10的FD与T记忆兼容性在完整任务上仍优于正常Task1冻结参照F1；允许负结果，不预设成功。

六个CT3-P FD Task3末状态严格恢复，含模型、A/T充分统计、原args、RNG与loader/synthesis状态。Task4教师必须是该Task3模型自身。保留普通restore_new全部校验，原科学依赖SHA不变；新编排单独源码锁。每任务原优化器/调度重新建立，继承流程同原task_setup。不重训Task1–3，没有独立S0。

固定HK三seed各Task4–11（末20→23类）、ISIC三seed各Task4，1993/1994/1995原顺序。27新任务×10epoch=270task-epochs，5570正式steps：HK950/1120/1020，ISIC1280/600/600。C2=FD+A，C3=同轨迹FD+T，共用六条新续训、27checkpoint、54新阶段预测；原36个C2/C3前缀预测复用。合计90/918新方法全程指标；加锁定F1/P/C0对照共225/2295。

原完整real loss+10*pointwise联合特征FD；同一当前增强图像，上一任务模型固定无梯度教师，学生adapter有梯度。原CE路由/reduction/除3/assignment/pull/采样/batch48/FP32/优化配方不变。A/T从各自immutable task-start bank估计，只当期fit，末轮一次提交。无ConCM/Risk/新模块/head选择/参数搜索/early-stop。父末checkpoint、失败证据与负结果完整保存。

访问：train仅当期fit，旧/未来fit与test/reserved/oracle全部0。CT3原代码的prefix guard扩展为真实seen边界，不能误将HK末任务当两类。全部54W及27checkpoint锁定后才允许新增val预测；相同sample_id升序/batch48/原pointwise推理与CBRidge lambda.001。现有前缀预测与F1/P/C0分数只读复用，不重评取最好。

主比较C3−F1；次C2−F1、C3−C2，补充C3−P、C3−C0固定控制。逐阶段/终端BA、全阶段平均、old/current/HM/tail、逐类分母/正确数、零召回、最差seed、错误分解、遗忘。2000次seed50001按类component配对bootstrap，固定三模型共享规范样本抽样，不重采样seed。公开全部seed及负结果，区分开发性val与独立确认，不据均值单独宣布成功。

工程：六父严格restore和RNG检查，Task4固定教师hash与参数隔离、A/T shape/count/finite、父W复现；训练复用CT3已验证FD梯度/教师锁审计，首正式epoch实际吞吐记录，不额外探测训练或重复更新。缺父/坏状态/非有限/访问越界立即BLOCKED。只精确同锁恢复技术故障，保留失败消耗。

资源：一个GPU worker，GPU小时无截止，仅固定5570steps。原任务边界重新建立优化器属于原语义。预计1.5–3GPU小时，以实际首epoch吞吐更新。CPU解析/报告额外累计≤2小时，归档传输单独墙钟账；运行盘额外≤2GiB且空闲≥1GiB。逐单元归档到my-gpu既有真实NFS项目目录，归档总量≤10GiB、提前写读探针，完整校验后才删除hb可恢复副本。六父归档始终保留，最终27checkpoint均保留；不购置算力，不删除唯一资产或终止他人进程。

完成固定矩阵、验证与公开聚合交付后NEXT_DECISION=STOP。不得由本计划自动搜索强度或再增数据/方法/test访问。
