# R59 完整结果：共享角度量与全部真实类分布证据

三域三seed、45完整训练stage、每stage10epoch、52110实际optimizersteps、135保存读出、495完整TRAIN LOO曲线核验通过。closeout新增模型前向/优化均0。原4h6GB封存，attempt1exit0总1626.785019s（27.11min）/退出新增实际200187904bytes，CPU1656.116933s/峰RSS2390609920bytes。

**候选distribution_shared对H0/R28 BOTH门false，保留H0。固定R59不采用，不选择HK/seed/其他读出替代；不扫描scale、正则、epoch、kernel带宽或最近k救援。**

## 三域三臂均值：平均/最终/尾BA（%）

|数据集|候选distribution_shared|distribution_fixed|centroid_shared|
|---|---|---|---|
|NIH|28.255438/19.333333/8.571429|27.351411/20.333333/7.619048|27.303792/20.333333/8.571429|
|ISIC|48.092563/43.697462/49.358974|46.541649/40.551936/45.352564|51.137192/45.739825/57.371795|
|HK|58.635087/55.124069/41.277032|59.077700/56.354850/45.070190|52.983529/51.735783/30.248270|

## 全部参照差（平均/最终/尾，pp）

|参照|数据集|三指标差|final/tail联合改善seed|
|---|---|---|---|
|H0|NIH|-4.099500/-7.000000/-3.492063|0/3|
|H0|ISIC|-7.289132/-8.557976/-9.829060|0/3|
|H0|HK|2.143325/-2.385404/3.454533|0/3|
|R28|NIH|-4.162698/-6.777778/-3.174603|0/3|
|R28|ISIC|-7.381233/-8.386906/-9.134615|0/3|
|R28|HK|2.162389/-2.453476/3.362705|0/3|

## 全seed保留

|数据集|arm|seed|平均BA|最终BA|尾BA|
|---|---|---|---|---|---|
|NIH|distribution_shared|74002|28.280423|19.333333|8.571429|
|NIH|distribution_shared|74003|28.205467|19.333333|8.571429|
|NIH|distribution_shared|74004|28.280423|19.333333|8.571429|
|NIH|distribution_fixed|74002|27.351411|20.333333|7.619048|
|NIH|distribution_fixed|74003|27.351411|20.333333|7.619048|
|NIH|distribution_fixed|74004|27.351411|20.333333|7.619048|
|NIH|centroid_shared|74002|27.353395|20.333333|8.571429|
|NIH|centroid_shared|74003|27.204586|20.333333|8.571429|
|NIH|centroid_shared|74004|27.353395|20.333333|8.571429|
|ISIC|distribution_shared|74002|48.023118|43.489129|49.358974|
|ISIC|distribution_shared|74003|48.127285|43.801629|49.358974|
|ISIC|distribution_shared|74004|48.127285|43.801629|49.358974|
|ISIC|distribution_fixed|74002|46.541649|40.551936|45.352564|
|ISIC|distribution_fixed|74003|46.541649|40.551936|45.352564|
|ISIC|distribution_fixed|74004|46.541649|40.551936|45.352564|
|ISIC|centroid_shared|74002|51.137192|45.739825|57.371795|
|ISIC|centroid_shared|74003|51.137192|45.739825|57.371795|
|ISIC|centroid_shared|74004|51.137192|45.739825|57.371795|
|HK|distribution_shared|74002|58.641020|55.106502|41.277032|
|HK|distribution_shared|74003|58.632121|55.132852|41.277032|
|HK|distribution_shared|74004|58.632121|55.132852|41.277032|
|HK|distribution_fixed|74002|59.077700|56.354850|45.070190|
|HK|distribution_fixed|74003|59.077700|56.354850|45.070190|
|HK|distribution_fixed|74004|59.077700|56.354850|45.070190|
|HK|centroid_shared|74002|52.983074|51.759938|30.248270|
|HK|centroid_shared|74003|52.963973|51.687474|30.248270|
|HK|centroid_shared|74004|53.003542|51.759938|30.248270|

## 曲线与判别含义

|数据集|末三epoch描述平台数|终点LOO NLL低于epoch0|最终TRAIN LOO宏/尾BA%|度量有效维数|
|---|---|---|---|---|
|NIH|18/18|18/18|21.746987/24.910500|226.162206|
|ISIC|9/9|9/9|50.704515/69.823232|464.836304|
|HK|18/18|18/18|59.165969/49.025005|452.781433|

45/45阶段目标下降，末三epoch有<1%相对波动。正权度量仍有数百有效维数，不是R57同一种双因子尺度失衡或近单方向乘积；不同有效维数定义不能作统一容量百分比分解。平台不证明全局收敛，也不保证加epoch有效。

完整TRAIN LOO曲线没有独立留出集。LOO训练仍共用支持样本且共同拟合度量，不等患者crossfit；与R55原重代入风险定义不同，不得直接相减宣称泛化问题已解决。NIH的完整LOO TRAIN宏/尾本身较低，因此不能把失败统一归为“TRAIN记住了但开发没泛化”。

条件消融：同learned度量下分布证据相对单中心在HK尾提高（41.277032 vs30.248270），ISIC更低（49.358974 vs57.371795），NIH尾持平（8.571429）。学习度量相对固定分布在ISIC改善，却在HK降低（尾41.277032 vs45.070190）。这没有提供跨域一致收益，不否定所有分布建模，不择有利域，也不做因果贡献百分比分解。

完整真实历史记忆、逐类质量1、自身支持剔除、双边梯度均有核验，却未实现NIH/ISIC改善。因此“只是忘了均衡/真实回放/双因子秩塌缩”不足以解释全部历史失败；也不能宣称预训练表征无价值。固定角核分布和共享逐维度几何的任务判别不足仍需区分，不能以提高核尺度/自由度代替新依据。

## 旧尾→新头终点错误：仅描述给定类别集合的竞争

|数据集|arm|seed|旧尾行数|旧尾正确|旧尾判为最后task新头|
|---|---|---|---|---|---|
|NIH|distribution_shared|74002|90|8|1|
|NIH|distribution_fixed|74002|90|4|0|
|NIH|centroid_shared|74002|90|7|1|
|NIH|distribution_shared|74003|90|8|1|
|NIH|distribution_fixed|74003|90|4|0|
|NIH|centroid_shared|74003|90|7|1|
|NIH|distribution_shared|74004|90|8|1|
|NIH|distribution_fixed|74004|90|4|0|
|NIH|centroid_shared|74004|90|7|1|
|ISIC|distribution_shared|74002|50|25|5|
|ISIC|distribution_fixed|74002|50|23|6|
|ISIC|centroid_shared|74002|50|29|4|
|ISIC|distribution_shared|74003|50|25|5|
|ISIC|distribution_fixed|74003|50|23|6|
|ISIC|centroid_shared|74003|50|29|4|
|ISIC|distribution_shared|74004|50|25|5|
|ISIC|distribution_fixed|74004|50|23|6|
|ISIC|centroid_shared|74004|50|29|4|
|HK|distribution_shared|74002|144|74|8|
|HK|distribution_fixed|74002|144|76|8|
|HK|centroid_shared|74002|144|74|5|
|HK|distribution_shared|74003|144|74|8|
|HK|distribution_fixed|74003|144|76|8|
|HK|centroid_shared|74003|144|74|5|
|HK|distribution_shared|74004|144|74|8|
|HK|distribution_fixed|74004|144|76|8|
|HK|centroid_shared|74004|144|74|5|

旧尾为最后task前已到达且TRAIN<250，新头为最后task到达且TRAIN>=250。全部27终点混淆矩阵公开；此分组不是训练破坏或因果分解，重复曝光不是独立患者。

## 后续主线与边界

固定R59关闭，当前无活动训练，持续研究/唯一心跳保持。后继不在R57强弱约束或R59核尺度间打转。先核对医学特征、标签定义与目标决策接口，区分类别语义/任务构造、固定表示类别重叠、支持混合与判别边界。必要机制工作不重复旧诊断、不以短pilot淘汰；有不同且有据结构后另立完整三域三seed协议直接实验。

新增feature记忆191686656bytes加标签/ID/序列化，额外历史曝光/算法记忆不同于HASTEN；不优化comparator。Spatial/CLS+patch和D13暂停保持。所有重复历史开发集，无TEST/独立患者/临床证据。公开自有source/protocol/安全全classseed正负/曲线/混淆；特征/权重/身份/逐例响应/原日志/私有路径不公开。
