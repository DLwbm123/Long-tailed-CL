# D51：类别专属文本残差——真实 TRAIN 学习 pilot 不支持

本轮实际优化新的参数化结构，未再以旧模型零步方向代替学习试验。给每个已到达初始类增加512维zero-init残差，直接加入原Text_encode输出、再进入原共享映射及paired/hierarchy损失；原Image/Text adapters和所有loss系数保持。该结构区别R37出生路由保持、R24 detached logit head、R42首初始化矩阵、D49风险替换及D50视觉辅助CE。没有新增系数、教师或融合选择。

严格TRAIN fit/holdout：每类manifest奇偶双fold互换，每fold支持/查询各最多64行。每个domain×seed×fold×arm均从相同原生随机核心全新训练2epoch，仅fit行参与优化、类均值/协方差及父层级初始化；没有载入任何已训练checkpoint。零残差初始输出与control exact，新增参数不消耗RNG，候选初始全部核心状态与control exact；训练batch/步数/种子相同，优化器一次标准step。

|域|seed|两fold平均最终宏BA变化pp|两fold平均尾BA变化pp|
|---|---:|---:|---:|
|NIH|74002|-4.186063|-19.367816|
|NIH|74003|-0.361530|-27.198276|
|NIH|74004|-1.632184|-8.160920|
|ISIC|74002|-9.952352|-9.340659|
|ISIC|74003|-6.192050|-0.549451|
|ISIC|74004|-7.829670|-6.318681|
|HK|74002|+3.846886|+1.720653|
|HK|74003|+0.145016|-6.029208|
|HK|74004|-0.718292|-5.148810|

指标为固定原生.5 NCM+hyper融合上的候选减同次native control，不选择融合。NIH0/3种子达到事前宏/尾两项各+1pp；ISIC/HK平均保护门也false。拒绝固定class-text residual结构，不启动完整训练，不扫幅度/正则/seed/epoch/融合。HK74002局部改善不替代全域门，不把pilot当作H0/R28完整增量比较。

三域完整36记录，成功部分552个实际optimizer step；原attempt1在第一native control完成18步后，候选第一次前向的CPU/CUDA适配器位置检查失败，候选0优化，未完成分数未纳入36记录；原钟内把新增adapter移到目标GPU后再核zero-init parity，候选/折分/步数未变。保留失败原日志及实际成本，含失败重复native总570个新优化步，不能只报成功部分。

NCM prototype分支所有配对counts完全相同；候选residual梯度finite非零、训练后范数非零，全部原finite/step/parent审计通过，安全类计数与三个branch均保留。这些工程事实不构成效果成功。

原600s/300MB总时钟保留，attempt2exit0，总93.561197s/565248bytes，成功attempt峰值RSS2566455296bytes。捕获中性完整controller/campaign/ISIC worker ps及其GPU中性python显示；未捕获所有domain worker完整ps，不扩大声称。所有PID退休，不重跑。

小样本、2epoch和仅初始阶段限制明显；不否定所有类别参数化或原始论文之外的结构改进。TRAIN fold/seeds非患者独立，未使用VAL/TEST，无独立确认或临床结论。公开自有源代码/协议/全安全类计数/正负结果；不公开权重、特征、逐例响应、身份、图像、原日志或私有配置。
