# 互补旧类跨维协方差检验：补齐固定2×2来源对照

## 证据与问题

仅恢复新类跨维项，相对类内对角方案四切片总体BA均增加，ISIC总正确+48/+64、HK+2/+8；但相对完整岭只有ISIC Qgate通过，双查询主门槛仍均失败，ISIC Qfit对类内对角旧宏BA下降2.8111pp。本轮补齐只恢复旧类跨维项的互补格，区分新旧来源及共同作用，不据单个成功查询选模型。

## 唯一主要改动

保持K为seen总类数，G_between=μᵀμ/K；C_old和C_new分别为旧H和新S各类内散布之和除完整K，不能分别用组类数重归一化。候选G_O=G_between+C_old+diag(diag(C_new))；W=(G_O+0.001I)^−1(μᵀ/K)，score=8zW，无bias、无列归一化，原μ不额外单位化。

所有均值、完整G对角/trace、支持和表示不变。相对完整岭仅移除新类跨维项，相对类内对角仅恢复旧类跨维项。新类均值和方差仍保留，不能称不需要新数据。无λ/强度/秩/温度/坐标旋转选择扫描，无类重加权或重采样。消融依赖既有特征坐标，不称原逐样本LS等价解或新方法。

## 共享实现与验证

复用analytic.append/ridge，将现有new_covariance_ridge改名为group_covariance_ridge，并同步文件内全部调用；不存在其他调用方。旧/新来源共享同一矩阵构造，按本轮标志选择旧H的已算特征累计group_moments。组合S=μᵀμ+S_group−μ_groupᵀμ_group，对角覆盖为原fullS；旧bank不改。old_class_cross_covariance_only=true与前五标志互斥，前轮新来源与其他默认分支保持原公式。

核对group_moments计数和均值对应完整moments中选定类别，避免组错位。其余组的边际方差必须有限非负，不裁剪/λ回退/自动重跑。原finite1536/nonempty/uniformp/labels/头1536×6或15/相对残差<=1e−7及访问守卫保留。

每数据集3个二维工程解：无within解析、复制选定旧类样本类平衡解不变、已知旧/新跨维项时旧来源闭式解（验证原S不改、不是完整方程）。共6工程+2正式矩阵解，0逐元素正式解、0梯度含工程。测试明确选择class0作为旧组；既有新组测试选择class1。

## 冻结数据与资源

seed74002、原类序/划分、raw单位features、H64/169、新S2290/95、类内均匀且所有类质量1/K不变。先支持拟合保存封存，再query/ref；teacher只零残差模型形状。旧S、E、原图、新标签、GPU禁止。

两CPU任务各900秒、总3600秒，两线程。2618支持+2155查询=4773特征前向，额外对已算64+169=233行旧H特征累计矩阵，不增加特征推理但记录计算。两拟合头32256个float64元素258048字节252KiB，0bias。仍构造fullG，不宣称Gram节省。0适配器教师编码器记忆更新，历史成本不重计、客户端准备null。

## 主要参考、来源对照及判定

主参考raw_ridge。全部14参考为raw_centroid、MM、HH、identity_ETF、teacher_centroid、teacher_ETF、coordinate_median、centered_centroid、raw_ridge、raw_diagonal_ridge、raw_prototype_ridge、raw_isotropic_ridge、raw_within_diagonal_ridge、raw_new_covariance_ridge。历史只复用已交付聚合，不重跑三格。

固定D=raw_within_diagonal_ridge（旧关新关），O=本轮（旧开新关），N=raw_new_covariance_ridge（旧关新开），F=raw_ridge（旧开新开）。全部格的对角、均值、支持和正则相同。

报告旧来源差O−D和F−N，新来源差N−D和F−O，以及加性偏离F−N−O+D。对每个查询、每个seen类recall/CE和all/old/new宏BA/CE都计算，单位BA/recall为pp、CE为原单位；组平均须与逐类均值一致。此为固定条件下的描述性非加性，不是独立人群因果、统计显著性或可相加的“重要性”证明。不能事后用某一格代替主要指标。

主门槛仍每数据集双query总体BA严格正、旧/新各>=−1pp。完整逐类计数与全seen交叉熵、全部参考差值和微计数保留，不自动选优或部署。对简化参考改善不等于保留完整岭。

诊断报告旧/新trace（除K）、旧/新散布和旧/新非对角部分相对完整within的Frobenius范数比、头范数、残差及原within占比；trace可加而范数比不可加、不是平方能量。来源不同的样本数/类别数不同，不能把观察差值归为总体数据质量因果。

## 完成和证据边界

退役后收集两结果、PROGRAM、bank记录及日志，公开全部14参考、固定2×2对照、失败缺失及成本，再决定有信息下一步。不扩预算、不重跑已完成格。现有query已参与设计，患者分组未知，仅开发机制证据，不独立医学、跨种子或全阶段持续学习验证。
