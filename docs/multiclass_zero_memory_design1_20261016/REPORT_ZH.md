# 多类 softmax 零点记忆：启动报告

RUNNING_CPU_ONLY，尚无本轮终态结果。

科学源 `851359b5873f97082c000252a427172cc49c6f42` 已执行前代理remoteSHA/精确DESIGN匿名HTTP200、NAS nfs/free/probe、目标compile及中性父子ps验证。启动 1792114225.8290746，CPU截止 1792119725.829075，不重置。初始unit active、0/4完成仅启动快照，不当PASS。

本轮仅改变损失机制：OVR变为多类softmax CE及其共同零点二阶近似。四臂/低秩数据/48步/lr1/16/每类16d+16容量/原moment48/局部域.125/严格分类门不变。损失梯度尺度会变化，同数值lr不等优化进度等价；新增训练后一次无更新risk/gradient mapping诊断，成本全计，不能据此额外迭代。

0医学images/features/checkpoint/val/test/GPU，2线程CPU、case1800秒/总5500秒。各失败完整保留，不在健康源重试。启动报告和helper为独立reporting，不覆盖科学source。

前19轮74CPU累计5546.389817275573秒；本轮未闭合费用不预填0。GPU历史400036.74851310626不变。此前8af319a低秩分类主门失败，全部正负费用已公开，不能被本轮改判。
