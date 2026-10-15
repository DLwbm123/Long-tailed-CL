# 固定子空间半径的谱记忆：前瞻CPU协议

上一轮5225e12整体门失败，ISIC8/8、HK4/8、小规模6/8。HK最大误差在合成count2/4类，这些类raw协方差几乎精确，但Gaussian采样和单位归一化下损失仍大；方差上升是有限8批观察，不证明原因。本轮只检验径向分布，不调整容量、方向数、强度、更新或查询预算。

共用上一轮完整谱拟合：均值、15方向和值、尾方差，每类16d+16 float64，同原moment。k=min(15,d-1,nclass-1)，不按查询选秩；原moment48/lr.02保持。候选与Gaussian控制共享同一谱拟合，仅解码不同。

对Gaussian g，p=V^Tg，g_perp=g-Vp。候选raw=mu+V diag(sqrt(lambda))*sqrt(k)*p/||p||+sqrt((d-k)*tau)*g_perp/||g_perp||。Gaussian正交投影独立，因此两个子空间方向均匀且独立，各块E=0、协方差与Gaussian控制相同；高阶矩不同。每次16点8对g/-g，同PRNG种子。最后单位归一化可改变均值/协方差。固定块半径不保证总半径恒定（各lambda不同）、不保证降方差或真实类分布正确。不加径向调节参数。

零方差块跳过，正方差块投影norm<=1e-12拒绝不重抽；rawnorm<=1e-12拒绝。同原谱eigh数值域检查，存储/临时工作区和生成成本完整计。单支持/两支持合成退化性质不能冒充医学稀少类改善；当前支持全合成，统计可能透露低样本原支持，不授权医学原始特征回放。

4CPUcase：raw矩cubature/对称/rank1两点/零方差/退化拒绝/query梯度FD/可复现；原small32d512K4、ISIC1536d9434K8、HK1536d1038K23，seed263/264/265公开比例，原非高斯unit合成支持。0医学images/features/checkpoints/val/test/GPU。

四方法random、原moment48、Gaussian spectral、block_spherical主候选。48查询仍4随机+2共同stress组各8，rank8/beta8/Fro2/vnorm2/stress12Adam .01只评价不更新memory；方法集合变更不可跨轮max直接配对。冻结查询后8个新16点批，两个高维每批candidate max<=.9各moment/random且mean不增，加算术PASS才完整有限信号。8批平均只是诊断，不挑批/类/门。全部逐批逐类正负、采样方差、raw谱geometry、归一化后mean/trace、实际资源费用必须公开。

每case1800/整体5500秒仅CPU资源门，不改正式训练预算；失败保留不重试。未验证完整序列、分类训练或生产预算，不GPU-ready。先完整交付，再机制分析；不无限堆技巧或加预算追门。
