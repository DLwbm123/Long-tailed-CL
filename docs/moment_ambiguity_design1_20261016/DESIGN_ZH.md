# 同一二阶矩歧义：前瞻CPU反例审计

序列轮fff17f5接口全通过，但HK stage3/4全部失败，最坏匿名类15只有5个合成支持。该观察不能确认因果。本轮不再改采样或调参数，检验仅存均值/协方差对非线性未来查询的不可辨识性；不读旧节点，不重跑旧fixture。

参考A是rank4正则单纯形的5个等权单位顶点，参考B是其确定性旋转，仍5个等权点。嵌入d32和d1536，均值长度mu固定0及.6，正交残差尺度sqrt(1-mu²)，每点unit。两参考有相同样本数、均值、协方差和完整二阶矩；不是仅trace相同。旋转让固定第一查询在A投影为1,-1/4四次，在B为sqrt(3/8)两次、-1/sqrt6三次。

原查询族子集A=B=0、v=2q、beta8，不做query optimizer。沿该方向的精确softplus均值分别[sp(16a)+4sp(-4a)]/5及[2sp(16a sqrt(3/8))+3sp(-16a/sqrt6)]/5。给定同一个moment-only表示，同一查询的预测t无法区分两参考；三角不等式max(|t-LA|,|t-LB|)>=|LA-LB|/2。该下界是至少一个参考上的误差，不是两者都坏，也不是模型所有查询误差上界。中点取等验证实现。

canonical共享表示mu/4有效方向及lambda/15固定槽/tail0，等价的低秩各向同性生成law不受有效子空间基旋转影响。分别Gaussian与block_spherical解码，每次16点8对；8固定seed批只验证逐批三角下界，不平均改门。block spherical在本反例中单位norm及一二阶矩期望都精确，仍不自动恢复单纯形高阶结构。无memory/query optimizer，不增加此前正式训练预算。

3独立CPU：低维算术、d32、d1536。每规模mu0/.6各4固定查询（vertex/bisector/orthogonal/mean_axis），共16精确行；零/正gap全部报，2方法8批对应256采样行。0医学images/features/checkpoint/val/test/GPU，不称方法验证或GPU-ready。反例不证明此前HK具体失败原因或医学不可能，只界定moment-only统一保证的范围。

原case1800/总5500秒CPU资源界不变；全部wall/user/system/RSS逐次保留，三case不是四。只匿名标量，支持/方向/矩阵不导出。先完整交付再考虑表示或任务相关查询范围，不能用反例放宽旧门或直接医学GPU。
