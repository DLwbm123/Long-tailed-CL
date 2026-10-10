# OrganAMNIST首轮：评价工程失败，完整比较未成立

两臂训练均212更新，共424；两个零更新预检PASS，0策略更新。数据8343 train/6491官方val准备READY，test数组未打开。候选5stage评价完成，但base评价在五阶段推理之后、写metrics之前触发KeyError: historical_metrics。整轮INCOMPLETE，不能称完整配对或方法成功。

根因：新Organ official_validation未从仅适用于既有医学base的历史召回检查中排除，意外要求不存在的历史baseline文件；新数据集本来就没有该历史比较。不是训练发散或科学门槛失败，不放宽ISIC/HK已有历史检查，不在原轮暗中重跑。保留全部输出、原源码与状态。

|指标|base|候选（单臂，仅供失败归档）|配对Δ|
|---|---:|---:|---|
|final_balanced_accuracy|NA|95.327242|NA|
|average_incremental_balanced_accuracy|NA|98.267324|NA|
|final_tail_recall|NA|95.725878|NA|
|forgetting|NA|3.463548|NA|
|final_old_recall|NA|96.049148|NA|
|final_new_recall|NA|92.078665|NA|

候选逐阶段11类聚合已归档RESULTS.json；base未落盘metrics，因此不得伪造其指标或借候选单臂论证效果。不将NA填零、不按这些未配对结果改方法/参数。后续修复须独立尝试、冻结源、成本单列，只复用当前完整训练阶段模型，不重复424训练更新，原24h截止不重置。

6个GPU子进程费用全部保留（5成功+1评价失败），GPU驻留3616.210987秒，累计闭合160848.437431秒；失败评价449.133797秒已包含一次。CPU准备635.432053、CPU检查0.436503秒与协调器墙钟另列。首次冻结前CPU总和fixture失败另留回执，不声称全流程无工程失败。

仅GPU0/1，0test/策略更新。公开只有源码/协议/匿名聚合/标量/完整失败摘要/成本；不发布影像、权重、逐样本索引分数特征、统计/中心/原型矩阵、私有路径或原始日志。单seed/order开发val，非独立患者验证。
