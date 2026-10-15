# 同存储预算的谱分布记忆：前瞻CPU协议

生成矩拟合轮2e247f2整体失败，ISIC0/8；高维相同生成目标甚至略差于旧目标posthoc控制，没有最优证书，不能归因容量不足。本轮不再调整采样或目标系数，重新分配同一存储预算检验方向协方差结构。

原16中心向量+16权重，共16d+16 float64/类；候选均值d、15协方差特征向量15d、15特征值、1剩余方差，同为16d+16。k=min(15,d-1,nclass-1)，方向/值固定15槽零填充，不能按查询或val挑rank。current synthetic全支持中心化协方差一次eigh，保留最大k方向，余d-k方向用剩余trace/(d-k)等方方差。只裁数值负特征值，<-1e-10拒绝。原始协方差PSD、trace和重建Fro误差均记录；不恢复非高斯高阶结构。

生成raw=mu+C_model^(1/2)g，再单位归一化；16点为8对g/-g。原始mean/cov公式可查，归一化后的mean/cov并不保证相同，rawnorm<=1e-12拒绝不重抽。存储无旧support/样本IDs，但小样本统计可能识别原特征；当前仅合成检查，不将通过视为医学原始特征回放许可。

四方法random、原ball moment48（不降基线）、isotropic(mu+traceC/d)、spectral。等方控制存储更小，非容量匹配；与原moment容量精确相等。candidate没有优化器更新，baseline48/lr.02不变，eig构造及所有解码/查询成本按实际驻留保留，不称计算等价。

4独立CPU：raw矩cubature/PSD trace/保留d-1时完整协方差/零方差/排列law/query FD/可复现；原32d512K4、1536d9434K8、1536d1038K23，seed263/264/265、公开类比例全合成非高斯unit支持。0医学images/features/checkpoints/val/test/GPU。48查询=4随机+2四方法共同stress各8，rank8/beta8/Fro2/vnorm2/stress12Adam .01；查询只评价不更新memory。方法集合变更不能跨轮max直接配对。

冻结查询后8新16点批。主spectral每批两高维max<=.9各原moment/random且mean不增，加算术PASS才完整有限信号；8批平均只诊断，不改门、不选类/批。全正负逐批和逐类误差、等方对照、raw协方差重建与trace/normalized偏差、方差、全部存储和构造资源必须公开。

每case1800/总体5500秒资源上限，失败保留不重试。未检验序列、分类训练/生产预算，不GPU-ready。先完整交付再机制分析，不增加容量/正式seed/预算档位追门。
