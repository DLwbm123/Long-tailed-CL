# 三数据集 ordered / shuffled：原始完整方案验证

同一设置内使用相同划分、初始化、训练步数和两轮适配器训练；每臂只运行 seed74002。
完整方案 cf_group；cf_linear 为闭式头基线，cf_prototype 增加多原型证据，cf_weighted 使用直接梯度权重控制。
医学数据使用原有开发验证集；CIFAR 使用冻结模型的官方测试集。无验证集选模型、选参数或提前停止。
单种子、短训练预算只能支持本轮设置内的比较，不能作为充分收敛、独立临床确认或论文表格复现的结论。

|数据/顺序/方法|平均 Acc %|最终 Acc %|最终 BA %|任务遗忘 pp|策略更新|非均匀权重轮数|
|---|---:|---:|---:|---:|---:|---:|
|ISIC_shuffled_cf_group|59.918|56.271|57.125|10.666|48|2/6|
|HK_shuffled_cf_group|84.003|82.686|61.164|5.947|96|0/12|
|CIFAR100LT_shuffled_cf_group|86.995|84.530|84.530|3.892|96|0/12|
|ISIC_ordered_cf_group|62.769|55.932|56.810|12.702|48|0/6|
|HK_ordered_cf_group|86.576|83.392|61.726|6.706|96|0/12|
|CIFAR100LT_ordered_cf_group|87.624|84.640|84.640|2.912|96|0/12|
|ISIC_shuffled_cf_linear|60.031|56.610|57.429|9.612|0|0/6|
|HK_shuffled_cf_linear|84.041|82.744|61.237|6.028|0|0/12|
|CIFAR100LT_shuffled_cf_linear|87.012|84.630|84.630|3.780|0|0/12|
|ISIC_ordered_cf_linear|62.769|55.932|56.810|12.702|0|0/6|
|HK_ordered_cf_linear|86.616|83.333|61.697|6.719|0|0/12|
|CIFAR100LT_ordered_cf_linear|87.620|84.660|84.660|2.848|0|0/12|
|ISIC_shuffled_cf_prototype|60.189|56.610|57.429|10.307|0|0/6|
|HK_shuffled_cf_prototype|84.003|82.686|61.164|5.947|0|0/12|
|CIFAR100LT_shuffled_cf_prototype|86.995|84.530|84.530|3.892|0|0/12|
|ISIC_ordered_cf_prototype|62.769|55.932|56.810|12.702|0|0/6|
|HK_ordered_cf_prototype|86.576|83.392|61.726|6.706|0|0/12|
|CIFAR100LT_ordered_cf_prototype|87.624|84.640|84.640|2.912|0|0/12|
|ISIC_shuffled_cf_weighted|60.031|56.610|57.429|9.612|48|1/6|
|HK_shuffled_cf_weighted|84.003|82.686|61.164|5.947|96|0/12|
|CIFAR100LT_shuffled_cf_weighted|86.995|84.530|84.530|3.892|96|0/12|
|ISIC_ordered_cf_weighted|62.769|55.932|56.810|12.702|48|0/6|
|HK_ordered_cf_weighted|86.576|83.392|61.726|6.706|96|0/12|
|CIFAR100LT_ordered_cf_weighted|87.624|84.640|84.640|2.912|96|0/12|

非均匀权重轮数为零时，不能宣称 RL 权重对训练实际生效；策略有更新与最终动作被接受分开统计。
原型证据折叠为线性头中的先验，并未增加非线性分类边界；旧类共同平移仍是近似。
many >100、medium 20–100、few <20 按冻结训练样本数定义，空组记 null。每类召回及任务准确率矩阵见 RESULTS.json。
