# 医学策略提议执行诊断：完成报告

两条新增诊断完整完成；seed74002、shuffled、每任务两轮。原 strict 和 uniform 轨迹复用首轮。
医学仅使用既有 development validation，test/reserved 未访问；没有新增 CIFAR 评估。
诊断执行最终策略均值提议，accepted 仍表示原安全门是否通过，不表示是否执行。

|数据|方法|平均 BA %|最终 BA %|尾类召回 %|任务遗忘 pp|原门通过|非均匀执行|
|---|---|---:|---:|---:|---:|---:|---:|
|ISIC|cf_prototype|60.2669|57.4294|57.6213|10.3068|NA|0/6|
|ISIC|cf_group|60.0151|57.1246|57.0116|10.6665|2/6|2/6|
|ISIC|proposal|60.2796|57.9178|57.9731|9.9720|2/6|6/6|
|HK|cf_prototype|62.4319|61.1644|46.1954|5.9467|NA|0/12|
|HK|cf_group|62.4319|61.1644|46.1954|5.9467|0/12|0/12|
|HK|proposal|62.3856|61.0578|45.9921|5.9873|0/12|12/12|

## 对照差值

以下差值单位为百分点；BA/尾类越高越好，任务遗忘越低越好。
- ISIC proposal − cf_group：average_incremental_balanced_accuracy +0.2644，final_balanced_accuracy +0.7933，final_tail_recall +0.9615，task_forgetting -0.6944。
- ISIC proposal − cf_prototype：average_incremental_balanced_accuracy +0.0126，final_balanced_accuracy +0.4884，final_tail_recall +0.3518，task_forgetting -0.3347。
- HK proposal − cf_group：average_incremental_balanced_accuracy -0.0464，final_balanced_accuracy -0.1066，final_tail_recall -0.2033，task_forgetting +0.0406。
- HK proposal − cf_prototype：average_incremental_balanced_accuracy -0.0464，final_balanced_accuracy -0.1066，final_tail_recall -0.2033，task_forgetting +0.0406。

## 解释

本轮是端到端干预，执行不同动作后后续状态与提议会分叉，不能称逐步相同提议的离线比较。
是否改善指标与是否满足原安全门分别报告；代理门失败不等于已证实临床伤害，但不能声称旧知识保护成功。
单种子、两轮训练和反复使用的开发集仅支持初筛；没有独立确认、充分收敛或统计优越声明。
全部正负结果保留，不按本轮结果更换阈值、种子、checkpoint或重新运行。

## 成本与交付

首轮 48145.345917 秒；本轮 2614.809568 秒；累计 50760.155486 秒。
成本使用已封口 suite 回执，包含启动、导入、训练与评估；旧成本不重置。
仅交付聚合指标、诊断、协议与源码。原始图像、样本身份、私有配置、checkpoint、特征及原始日志不公开。
